package ltdetr

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
	ortlog.Infow("creating LTDETR detection engine",
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

	// Detect input size from model metadata
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

	ortlog.Infow("LTDETR detection engine created successfully",
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
		ortlog.Infow("destroying LTDETR detection engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

func (e *DetEngine) Predict(img image.Image) ([]DetResult, error) {
	return e.predictSingle(img)
}

func (e *DetEngine) PredictBatch(imgs []image.Image) ([][]DetResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	if e.config.DynamicBatch {
		return e.predictBatchDynamic(imgs)
	}

	results := make([][]DetResult, len(imgs))
	for i, img := range imgs {
		res, err := e.predictSingle(img)
		if err != nil {
			return nil, fmt.Errorf("image %d prediction failed: %w", i, err)
		}
		results[i] = res
	}
	return results, nil
}

func (e *DetEngine) predictSingle(img image.Image) ([]DetResult, error) {
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

	// LTDETR ONNX outputs: labels, boxes, scores
	var labelsName, boxesName, scoresName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "labels":
			labelsName = name
		case "boxes":
			boxesName = name
		case "scores":
			scoresName = name
		}
	}

	if boxesName == "" || scoresName == "" {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("LTDETR model requires boxes and scores outputs, got %v", e.session.OutputNames)
	}

	boxesOut := outputValues[boxesName]
	scoresOut := outputValues[scoresName]
	if boxesOut == nil || scoresOut == nil {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("LTDETR model output boxes or scores is nil")
	}

	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	scoresShape, err := scoresOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get scores shape: %w", err)
	}

	if len(boxesShape) != 3 || len(scoresShape) < 2 {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v scores=%v", boxesShape, scoresShape)
	}

	numDetections := int(boxesShape[1])

	boxesData, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	scoresData, err := ort.GetTensorData[float32](scoresOut)
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get scores data: %w", err)
	}

	boxesCopy := make([]float32, numDetections*4)
	copy(boxesCopy, boxesData[:numDetections*4])
	scoresCopy := make([]float32, numDetections)
	copy(scoresCopy, scoresData[:numDetections])

	boxesOut.Destroy()
	scoresOut.Destroy()

	var labelsCopy []int64
	if labelsName != "" {
		labelsOut := outputValues[labelsName]
		if labelsOut != nil {
			labelsRaw, err := ort.GetTensorData[int64](labelsOut)
			if err == nil {
				labelsCopy = make([]int64, numDetections)
				copy(labelsCopy, labelsRaw[:numDetections])
			}
			labelsOut.Destroy()
		}
	}

	postprocessStart := time.Now()
	result := e.postprocess(boxesCopy, scoresCopy, labelsCopy, numDetections, params)
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(1, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return result, nil
}

func (e *DetEngine) predictBatchDynamic(imgs []image.Image) ([][]DetResult, error) {
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

	var labelsName, boxesName, scoresName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "labels":
			labelsName = name
		case "boxes":
			boxesName = name
		case "scores":
			scoresName = name
		}
	}

	if boxesName == "" || scoresName == "" {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("LTDETR model requires boxes and scores outputs, got %v", e.session.OutputNames)
	}

	boxesOut := outputValues[boxesName]
	scoresOut := outputValues[scoresName]
	if boxesOut == nil || scoresOut == nil {
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, fmt.Errorf("LTDETR model output boxes or scores is nil")
	}

	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	scoresShape, err := scoresOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get scores shape: %w", err)
	}

	if len(boxesShape) != 3 || len(scoresShape) < 2 {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v scores=%v", boxesShape, scoresShape)
	}

	batchSize := int(boxesShape[0])
	numDetections := int(boxesShape[1])

	boxesData, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	scoresData, err := ort.GetTensorData[float32](scoresOut)
	if err != nil {
		boxesOut.Destroy()
		scoresOut.Destroy()
		return nil, fmt.Errorf("failed to get scores data: %w", err)
	}

	boxesCopy := make([]float32, len(boxesData))
	copy(boxesCopy, boxesData)
	scoresCopy := make([]float32, len(scoresData))
	copy(scoresCopy, scoresData)

	boxesOut.Destroy()
	scoresOut.Destroy()

	var labelsData []int64
	if labelsName != "" {
		labelsOut := outputValues[labelsName]
		if labelsOut != nil {
			labelsRaw, err := ort.GetTensorData[int64](labelsOut)
			if err == nil {
				labelsData = make([]int64, len(labelsRaw))
				copy(labelsData, labelsRaw)
			}
			labelsOut.Destroy()
		}
	}

	if batchSize != len(imgs) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(imgs))
	}

	postprocessStart := time.Now()
	results := make([][]DetResult, batchSize)
	boxStride := numDetections * 4
	scoreStride := numDetections
	labelStride := numDetections
	for i := 0; i < batchSize; i++ {
		var labelSlice []int64
		if len(labelsData) > (i+1)*labelStride {
			labelSlice = labelsData[i*labelStride : (i+1)*labelStride]
		}
		results[i] = e.postprocess(
			boxesCopy[i*boxStride:(i+1)*boxStride],
			scoresCopy[i*scoreStride:(i+1)*scoreStride],
			labelSlice,
			numDetections,
			paramsList[i],
		)
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(batchSize, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

type detCandidate struct {
	classID int
	score   float32
	x1, y1  float32
	x2, y2  float32
}

func (e *DetEngine) postprocess(boxes []float32, scores []float32, labels []int64, numDetections int, params imageParams) []DetResult {
	candidates := make([]detCandidate, 0, numDetections)

	for i := 0; i < numDetections; i++ {
		score := scores[i]
		if score < e.config.ConfThreshold {
			continue
		}

		boxOffset := i * 4
		x1 := boxes[boxOffset+0]
		y1 := boxes[boxOffset+1]
		x2 := boxes[boxOffset+2]
		y2 := boxes[boxOffset+3]

		classID := -1
		if labels != nil && i < len(labels) {
			classID = int(labels[i])
		}

		candidates = append(candidates, detCandidate{
			classID: classID,
			score:   score,
			x1:      x1,
			y1:      y1,
			x2:      x2,
			y2:      y2,
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
		box := boxXyxyToOrigScale(cand.x1, cand.y1, cand.x2, cand.y2, e.config.InputSize, params.origW, params.origH)
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

	ortlog.Infow("ltdetr det timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}
