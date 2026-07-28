package edgecrafter

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

// PoseEngine is the EdgeCrafter pose estimation engine.
// It expects ONNX models exported in "raw" mode with outputs:
//   - pred_logits    [N, num_queries, num_classes]
//   - pred_keypoints [N, num_queries, num_body_points, 2]
type PoseEngine struct {
	session  *ort.Session
	config   Config
	runCount uint64
}

// NewPoseEngine creates a new EdgeCrafter pose estimation engine.
func NewPoseEngine(cfg Config) (*PoseEngine, error) {
	ortlog.Infow("creating EdgeCrafter pose engine",
		"modelPath", cfg.ModelPath,
		"inputSize", cfg.InputSize,
		"confThreshold", cfg.ConfThreshold,
		"numClasses", cfg.NumClasses,
		"numBodyPoints", cfg.NumBodyPoints,
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

	ortlog.Infow("EdgeCrafter pose engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)

	return &PoseEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources.
func (e *PoseEngine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying EdgeCrafter pose engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict runs pose estimation on a single image.
func (e *PoseEngine) Predict(img image.Image) ([]PoseResult, error) {
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

	// Resolve outputs: pred_logits and pred_keypoints
	var logitsName, keypointsName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "pred_logits":
			logitsName = name
		case "pred_keypoints":
			keypointsName = name
		}
	}
	if logitsName == "" || keypointsName == "" {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("EdgeCrafter pose model requires pred_logits and pred_keypoints outputs, got %v", e.session.OutputNames)
	}

	logitsOut, ok0 := outputValues[logitsName]
	keypointsOut, ok1 := outputValues[keypointsName]
	if !ok0 || !ok1 || logitsOut == nil || keypointsOut == nil {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("EdgeCrafter pose model output %q or %q does not exist", logitsName, keypointsName)
	}

	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}
	keypointsShape, err := keypointsOut.GetShape()
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get keypoints shape: %w", err)
	}

	if len(logitsShape) != 3 || len(keypointsShape) != 4 {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: logits=%v keypoints=%v", logitsShape, keypointsShape)
	}

	numQueries := int(logitsShape[1])
	numClasses := int(logitsShape[2])
	numBodyPoints := int(keypointsShape[2])

	logitsRaw, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}
	keypointsRaw, err := ort.GetTensorData[float32](keypointsOut)
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get keypoints data: %w", err)
	}

	logitsData := make([]float32, numQueries*numClasses)
	copy(logitsData, logitsRaw[:numQueries*numClasses])
	kptData := make([]float32, numQueries*numBodyPoints*2)
	copy(kptData, keypointsRaw[:numQueries*numBodyPoints*2])

	logitsOut.Destroy()
	keypointsOut.Destroy()

	postprocessStart := time.Now()
	results := e.postprocess(logitsData, kptData, numQueries, numClasses, numBodyPoints, params)
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(1, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

// PredictBatch runs pose estimation on a batch of images.
func (e *PoseEngine) PredictBatch(imgs []image.Image) ([][]PoseResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	if !e.config.DynamicBatch {
		results := make([][]PoseResult, len(imgs))
		for i, img := range imgs {
			res, err := e.Predict(img)
			if err != nil {
				return nil, fmt.Errorf("image %d prediction failed: %w", i, err)
			}
			results[i] = res
		}
		return results, nil
	}

	startedAt := time.Now()

	preprocessStart := time.Now()
	inputTensor, paramsList, err := preprocessBatch(imgs, e.config.InputSize, e.session, e.config.PreprocessConfig)
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

	var logitsName, keypointsName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "pred_logits":
			logitsName = name
		case "pred_keypoints":
			keypointsName = name
		}
	}
	if logitsName == "" || keypointsName == "" {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("EdgeCrafter pose model requires pred_logits and pred_keypoints outputs, got %v", e.session.OutputNames)
	}

	logitsOut, ok0 := outputValues[logitsName]
	keypointsOut, ok1 := outputValues[keypointsName]
	if !ok0 || !ok1 || logitsOut == nil || keypointsOut == nil {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("EdgeCrafter pose model output %q or %q does not exist", logitsName, keypointsName)
	}

	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}
	keypointsShape, err := keypointsOut.GetShape()
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get keypoints shape: %w", err)
	}

	if len(logitsShape) != 3 || len(keypointsShape) != 4 {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: logits=%v keypoints=%v", logitsShape, keypointsShape)
	}

	batchSize := int(logitsShape[0])
	numQueries := int(logitsShape[1])
	numClasses := int(logitsShape[2])
	numBodyPoints := int(keypointsShape[2])

	logitsRaw, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}
	keypointsRaw, err := ort.GetTensorData[float32](keypointsOut)
	if err != nil {
		logitsOut.Destroy()
		keypointsOut.Destroy()
		return nil, fmt.Errorf("failed to get keypoints data: %w", err)
	}

	logitsData := make([]float32, len(logitsRaw))
	copy(logitsData, logitsRaw)
	kptData := make([]float32, len(keypointsRaw))
	copy(kptData, keypointsRaw)

	logitsOut.Destroy()
	keypointsOut.Destroy()

	if batchSize != len(imgs) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(imgs))
	}

	postprocessStart := time.Now()
	logitStride := numQueries * numClasses
	kptStride := numQueries * numBodyPoints * 2
	results := make([][]PoseResult, batchSize)
	for b := 0; b < batchSize; b++ {
		results[b] = e.postprocess(
			logitsData[b*logitStride:(b+1)*logitStride],
			kptData[b*kptStride:(b+1)*kptStride],
			numQueries, numClasses, numBodyPoints, paramsList[b],
		)
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(batchSize, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

type poseCandidate struct {
	classID int
	score   float32
	index   int
}

func (e *PoseEngine) postprocess(
	logits []float32,
	keypoints []float32,
	numQueries, numClasses, numBodyPoints int,
	params imageParams,
) []PoseResult {
	candidates := make([]poseCandidate, 0, numQueries)

	for i := 0; i < numQueries; i++ {
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

		candidates = append(candidates, poseCandidate{
			classID: classID,
			score:   maxScore,
			index:   i,
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	numSelect := e.config.NumSelect
	if numSelect <= 0 {
		numSelect = 60
	}
	if len(candidates) > numSelect {
		candidates = candidates[:numSelect]
	}

	results := make([]PoseResult, 0, len(candidates))
	for _, cand := range candidates {
		// Decode keypoints: [num_body_points, 2] -> (x, y) normalized
		kpts := make([]KeyPoint, numBodyPoints)
		for k := 0; k < numBodyPoints; k++ {
			kptOffset := cand.index*numBodyPoints*2 + k*2
			x := keypoints[kptOffset]
			y := keypoints[kptOffset+1]

			// Keypoints are in normalized [0,1] coordinates, scale to original image
			origX := max(0, min(params.origW, int(x*float32(params.origW))))
			origY := max(0, min(params.origH, int(y*float32(params.origH))))

			kpts[k] = KeyPoint{
				X:     origX,
				Y:     origY,
				Score: cand.score,
			}
		}

		// Compute bounding box from keypoints
		minX, minY := params.origW, params.origH
		maxX, maxY := 0, 0
		for _, kp := range kpts {
			if kp.X < minX {
				minX = kp.X
			}
			if kp.Y < minY {
				minY = kp.Y
			}
			if kp.X > maxX {
				maxX = kp.X
			}
			if kp.Y > maxY {
				maxY = kp.Y
			}
		}

		box := image.Rect(minX, minY, maxX, maxY)

		results = append(results, PoseResult{
			ClassID:   cand.classID,
			Score:     cand.score,
			Box:       box,
			KeyPoints: kpts,
		})
	}

	return results
}

func (e *PoseEngine) logPredictTimings(batchSize int, preprocessElapsed, runElapsed, postprocessElapsed, totalElapsed time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("edgecrafter pose timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}
