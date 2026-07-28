package rfdetr

import (
	"fmt"
	"image"
	"sort"
	"sync/atomic"
	"time"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// Keypoint channel indices (GroupPose-style, 8 channels per keypoint)
const (
	kpChannelX        = 0
	kpChannelY        = 1
	kpChannelFindable = 2
	kpChannelVisible  = 3
	kpChannelLXX      = 4
	kpChannelLXY      = 5
	kpChannelLYY      = 6
	kpChannelClass    = 7
	kpPredDim         = 8
)

// KpEngine RF-DETR keypoint detection engine
type KpEngine struct {
	session  *ort.Session
	config   Config
	runCount uint64
}

// NewKpEngine creates a new RF-DETR keypoint detection engine
func NewKpEngine(cfg Config) (*KpEngine, error) {
	ortlog.Infow("creating RF-DETR keypoint engine",
		"modelPath", cfg.ModelPath,
		"inputSize", cfg.InputSize,
		"confThreshold", cfg.ConfThreshold,
		"numClasses", cfg.NumClasses,
		"numThreads", cfg.NumThreads,
		"useCuda", cfg.UseCuda)

	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}

	if err := oc.New(); err != nil {
		ortlog.Errorw("failed to initialize ONNX config", "error", err)
		return nil, fmt.Errorf("initialization failed: %w", err)
	}

	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	oc.Destroy()
	if err != nil {
		ortlog.Errorw("failed to create ONNX session", "modelPath", cfg.ModelPath, "error", err)
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	detectedSize, dynamicBatch := detectInputSizeAndDynamicBatch(cfg.ModelPath)
	if cfg.InputSize != 0 && cfg.InputSize != detectedSize {
		ortlog.Warnw("input_size mismatch, overriding with detected value",
			"configured", cfg.InputSize,
			"detected", detectedSize,
			"modelPath", cfg.ModelPath)
	}
	cfg.InputSize = detectedSize
	if !cfg.DynamicBatch {
		cfg.DynamicBatch = dynamicBatch
	}
	ortlog.Infow("input size resolved", "inputSize", cfg.InputSize, "dynamicBatch", cfg.DynamicBatch, "modelPath", cfg.ModelPath)

	ortlog.Infow("RF-DETR keypoint engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)

	return &KpEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *KpEngine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying RF-DETR keypoint engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict executes keypoint detection on a single image
func (e *KpEngine) Predict(img image.Image) ([]KpResult, error) {
	startedAt := time.Now()

	preprocessStart := time.Now()
	inputTensor, params, err := preprocess(img, e.config.InputSize, e.session, e.config.PreprocessConfig)
	if err != nil {
		return nil, fmt.Errorf("preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()
	preprocessElapsed := time.Since(preprocessStart)

	if len(e.session.InputNames) == 0 {
		return nil, fmt.Errorf("model has no input")
	}
	inputName := e.session.InputNames[0]

	inputValues := map[string]*ort.Value{
		inputName: inputTensor,
	}

	runStart := time.Now()
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer ort.DestroyValues(outputValues)
	runElapsed := time.Since(runStart)

	// Resolve output tensors
	var boxesName, logitsName, keypointsName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "pred_boxes":
			boxesName = name
		case "pred_logits":
			logitsName = name
		case "pred_keypoints":
			keypointsName = name
		}
	}

	// Fallback: if named outputs not found, try shape-based detection
	if boxesName == "" || logitsName == "" || keypointsName == "" {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("keypoint model requires pred_boxes, pred_logits and pred_keypoints outputs, got %v", e.session.OutputNames)
	}

	boxesOut := outputValues[boxesName]
	logitsOut := outputValues[logitsName]
	keypointsOut := outputValues[keypointsName]
	if boxesOut == nil || logitsOut == nil || keypointsOut == nil {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("keypoint model output is nil")
	}

	// Get shapes
	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}
	kpShape, err := keypointsOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get keypoints shape: %w", err)
	}

	if len(boxesShape) != 3 || len(logitsShape) != 3 || len(kpShape) != 4 {
		boxesOut.Destroy()
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v logits=%v keypoints=%v", boxesShape, logitsShape, kpShape)
	}

	numDetections := int(boxesShape[1])
	numClasses := int(logitsShape[2])
	totalKeypoints := int(kpShape[2])
	kpDim := int(kpShape[3])

	// Get raw data
	boxesRaw, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	logitsRaw, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}
	kpRaw, err := ort.GetTensorData[float32](keypointsOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get keypoints data: %w", err)
	}

	// Copy data before destroying tensors
	boxesData := make([]float32, numDetections*4)
	copy(boxesData, boxesRaw[:numDetections*4])
	logitsData := make([]float32, numDetections*numClasses)
	copy(logitsData, logitsRaw[:numDetections*numClasses])
	kpData := make([]float32, numDetections*totalKeypoints*kpDim)
	copy(kpData, kpRaw[:numDetections*totalKeypoints*kpDim])

	boxesOut.Destroy()
	logitsOut.Destroy()
	keypointsOut.Destroy()

	postprocessStart := time.Now()
	results := e.postprocess(boxesData, logitsData, kpData, numDetections, numClasses, totalKeypoints, kpDim, params)
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(1, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

type kpCandidate struct {
	classID int
	score   float32
	cx, cy  float32
	w, h    float32
	index   int
}

func (e *KpEngine) postprocess(
	boxes []float32, logits []float32, kpData []float32,
	numDetections, numClasses, totalKeypoints, kpDim int,
	params imageParams,
) []KpResult {
	candidates := make([]kpCandidate, 0, numDetections)

	for i := 0; i < numDetections; i++ {
		boxOffset := i * 4
		logitOffset := i * numClasses

		maxScore := float32(0.0)
		classID := -1
		for c := 0; c < numClasses; c++ {
			s := sigmoid(logits[logitOffset+c])
			if s > maxScore {
				maxScore = s
				classID = c
			}
		}

		if maxScore < e.config.ConfThreshold {
			continue
		}

		candidates = append(candidates, kpCandidate{
			classID: classID,
			score:   maxScore,
			cx:      boxes[boxOffset+0],
			cy:      boxes[boxOffset+1],
			w:       boxes[boxOffset+2],
			h:       boxes[boxOffset+3],
			index:   i,
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > e.config.MaxDetections {
		candidates = candidates[:e.config.MaxDetections]
	}

	kpStride := totalKeypoints * kpDim

	results := make([]KpResult, 0, len(candidates))
	for _, cand := range candidates {
		box := boxCxcywhToXyxy(cand.cx, cand.cy, cand.w, cand.h, params.origW, params.origH)

		// Decode keypoints for this detection
		kpOffset := cand.index * kpStride
		keyPoints := decodeKeypoints(kpData[kpOffset:kpOffset+kpStride], totalKeypoints, kpDim, params.origW, params.origH)

		results = append(results, KpResult{
			ClassID:   cand.classID,
			Score:     cand.score,
			Box:       box,
			KeyPoints: keyPoints,
		})
	}

	return results
}

// decodeKeypoints decodes raw keypoint predictions into pixel coordinates.
// Raw keypoint layout (GroupPose-style, 8 channels):
//
//	0: x (normalized), 1: y (normalized), 2: findable_logit, 3: visible_logit,
//	4: Lxx, 5: Lxy, 6: Lyy, 7: class_logit
func decodeKeypoints(kpData []float32, totalKeypoints, kpDim, origW, origH int) []KeypointResult {
	keyPoints := make([]KeypointResult, 0, totalKeypoints)

	for k := 0; k < totalKeypoints; k++ {
		offset := k * kpDim
		if offset+kpPredDim > len(kpData) {
			break
		}

		xNorm := kpData[offset+kpChannelX]
		yNorm := kpData[offset+kpChannelY]
		findableLogit := kpData[offset+kpChannelFindable]
		visibleLogit := kpData[offset+kpChannelVisible]

		findableScore := sigmoid(findableLogit)
		visibleScore := sigmoid(visibleLogit)

		keyPoints = append(keyPoints, KeypointResult{
			X:        int(xNorm * float32(origW)),
			Y:        int(yNorm * float32(origH)),
			Score:    visibleScore,
			Visible:  visibleScore > 0.5,
			Findable: findableScore > 0.5,
		})
	}

	return keyPoints
}

func (e *KpEngine) logPredictTimings(batchSize int, preprocessElapsed, runElapsed, postprocessElapsed, totalElapsed time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("rf-detr kp timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}
