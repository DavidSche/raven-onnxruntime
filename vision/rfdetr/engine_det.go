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

type DetEngine struct {
	session  *ort.Session
	config   Config
	runCount uint64
}

func NewDetEngine(cfg Config) (*DetEngine, error) {
	ortlog.Infow("creating RF-DETR detection engine",
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

	// Always detect input size from model to ensure correctness,
	// even if cfg.InputSize is explicitly set (it may be wrong).
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

	ortlog.Infow("RF-DETR detection engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)

	return &DetEngine{
		session: session,
		config:  cfg,
	}, nil
}

func (e *DetEngine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying RF-DETR detection engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

func (e *DetEngine) Predict(img image.Image) ([]DetResult, error) {
	startedAt := time.Now()

	preprocessStart := time.Now()
	inputTensor, params, err := preprocess(img, e.config.InputSize, e.session)
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
	runElapsed := time.Since(runStart)

	var boxesName, logitsName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "pred_boxes":
			boxesName = name
		case "pred_logits":
			logitsName = name
		}
	}
	if boxesName == "" || logitsName == "" {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("detection model requires pred_boxes and pred_logits outputs, got %v", e.session.OutputNames)
	}

	boxesOut, ok0 := outputValues[boxesName]
	logitsOut, ok1 := outputValues[logitsName]
	if !ok0 || !ok1 || boxesOut == nil || logitsOut == nil {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("detection model output %q or %q does not exist", boxesName, logitsName)
	}

	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}

	if len(boxesShape) != 3 || len(logitsShape) != 3 {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v logits=%v", boxesShape, logitsShape)
	}

	numDetections := int(boxesShape[1])
	numClasses := int(logitsShape[2])

	boxesRaw, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	logitsRaw, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}

	boxesData := make([]float32, numDetections*4)
	copy(boxesData, boxesRaw[:numDetections*4])
	logitsData := make([]float32, numDetections*numClasses)
	copy(logitsData, logitsRaw[:numDetections*numClasses])

	boxesOut.Destroy()
	logitsOut.Destroy()

	postprocessStart := time.Now()
	results := e.postprocess(
		boxesData,
		logitsData,
		numDetections, numClasses, params,
	)
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(1, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

func (e *DetEngine) PredictBatch(imgs []image.Image) ([][]DetResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	if !e.config.DynamicBatch {
		results := make([][]DetResult, len(imgs))
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
	inputTensor, paramsList, err := preprocessBatch(imgs, e.config.InputSize, e.session)
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
	runElapsed := time.Since(runStart)

	var boxesName, logitsName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "pred_boxes":
			boxesName = name
		case "pred_logits":
			logitsName = name
		}
	}
	if boxesName == "" || logitsName == "" {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("detection model requires pred_boxes and pred_logits outputs, got %v", e.session.OutputNames)
	}

	boxesOut, ok0 := outputValues[boxesName]
	logitsOut, ok1 := outputValues[logitsName]
	if !ok0 || !ok1 || boxesOut == nil || logitsOut == nil {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("detection model output %q or %q does not exist", boxesName, logitsName)
	}

	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}

	if len(boxesShape) != 3 || len(logitsShape) != 3 {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v logits=%v", boxesShape, logitsShape)
	}

	batchSize := int(boxesShape[0])
	numDetections := int(boxesShape[1])
	numClasses := int(logitsShape[2])

	boxesRaw, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	logitsRaw, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}

	boxesData := make([]float32, len(boxesRaw))
	copy(boxesData, boxesRaw)
	logitsData := make([]float32, len(logitsRaw))
	copy(logitsData, logitsRaw)

	boxesOut.Destroy()
	logitsOut.Destroy()

	if batchSize != len(imgs) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(imgs))
	}

	postprocessStart := time.Now()
	results := make([][]DetResult, batchSize)
	boxStride := numDetections * 4
	logitStride := numDetections * numClasses
	for i := 0; i < batchSize; i++ {
		results[i] = e.postprocess(
			boxesData[i*boxStride:(i+1)*boxStride],
			logitsData[i*logitStride:(i+1)*logitStride],
			numDetections, numClasses, paramsList[i],
		)
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(batchSize, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

type detCandidate struct {
	classID int
	score   float32
	cx, cy  float32
	w, h    float32
}

func (e *DetEngine) postprocess(boxes []float32, logits []float32, numDetections, numClasses int, params imageParams) []DetResult {
	candidates := make([]detCandidate, 0, numDetections)

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

		candidates = append(candidates, detCandidate{
			classID: classID,
			score:   maxScore,
			cx:      boxes[boxOffset+0],
			cy:      boxes[boxOffset+1],
			w:       boxes[boxOffset+2],
			h:       boxes[boxOffset+3],
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > e.config.MaxDetections {
		candidates = candidates[:e.config.MaxDetections]
	}

	results := make([]DetResult, 0, len(candidates))
	for _, cand := range candidates {
		box := boxCxcywhToXyxy(cand.cx, cand.cy, cand.w, cand.h, params.origW, params.origH)
		results = append(results, DetResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     box,
		})
	}

	return results
}

func (e *DetEngine) logPredictTimings(batchSize int, preprocessElapsed, runElapsed, postprocessElapsed, totalElapsed time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("rf-detr det timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}
