package yolo26

import (
	"fmt"
	"image"
	"sync/atomic"
	"time"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// DetEngine YOLO26-det Engine
type DetEngine struct {
	session  *ort.Session
	config   Config
	runCount uint64
}

// NewDetEngine initializes the detection engine
func NewDetEngine(cfg Config) (*DetEngine, error) {
	ortlog.Infow("creating YOLO26 detection engine",
		"modelPath", cfg.ModelPath,
		"inputSize", cfg.InputSize,
		"confThreshold", cfg.ConfThreshold,
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

	ortlog.Infow("YOLO26 detection engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)

	return &DetEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *DetEngine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying YOLO26 detection engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict executes detection inference
func (e *DetEngine) Predict(img image.Image) ([]DetResult, error) {
	results, err := e.PredictBatch([]image.Image{img})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return results[0], nil
}

// PredictBatch executes batch detection inference
func (e *DetEngine) PredictBatch(imgs []image.Image) ([][]DetResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	startedAt := time.Now()

	// preprocess
	preprocessStart := time.Now()
	inputTensor, paramsList, err := preprocessBatch(imgs, e.config.InputSize, e.session, e.config.PreprocessConfig)
	if err != nil {
		return nil, fmt.Errorf("preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()
	preprocessElapsed := time.Since(preprocessStart)

	// get actual input name (compatible with different models)
	if len(e.session.InputNames) == 0 {
		return nil, fmt.Errorf("model has no input")
	}
	inputName := e.session.InputNames[0]

	// inference
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

	// get first output (compatible with different output names)
	if len(e.session.OutputNames) == 0 {
		return nil, fmt.Errorf("model has no output")
	}
	outputName := e.session.OutputNames[0]
	outputValue, ok := outputValues[outputName]
	if !ok || outputValue == nil {
		return nil, fmt.Errorf("output %q does not exist", outputName)
	}

	// Output Shape: [N, 300, 6]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}

	shape, err := outputValue.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get output shape: %w", err)
	}
	// NOTE: Do NOT call outputValue.Destroy() here. GetTensorData returns a
	// reference to ORT-managed memory (not a copy), so destroying the Value
	// before postprocess reads `data` would be a use-after-free.
	// The deferred ort.DestroyValues(outputValues) at the top of this function
	// will release the output after postprocess completes.

	if len(shape) != 3 {
		return nil, fmt.Errorf("unexpected output shape: %v", shape)
	}

	batchSize := int(shape[0])
	numObjects := int(shape[1])
	attributes := int(shape[2])
	if batchSize != len(imgs) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(imgs))
	}
	if attributes != 6 {
		return nil, fmt.Errorf("unexpected detection attributes: %d", attributes)
	}

	stride := numObjects * attributes
	postprocessStart := time.Now()
	results := make([][]DetResult, batchSize)
	for i := 0; i < batchSize; i++ {
		start := i * stride
		end := start + stride
		results[i] = e.postprocess(data[start:end], paramsList[i])
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(batchSize, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

func (e *DetEngine) logPredictTimings(batchSize int, preprocessElapsed, runElapsed, postprocessElapsed, totalElapsed time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("yolo26 det timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}

// postprocess performs post-processing and parses output results
func (e *DetEngine) postprocess(data []float32, params imageParams) []DetResult {
	results := make([]DetResult, 0)

	const stride = 6
	numDetections := len(data) / stride

	for i := 0; i < numDetections; i++ {
		offset := i * stride

		// [x1, y1, x2, y2, score, class_id]
		x1 := data[offset+0]
		y1 := data[offset+1]
		x2 := data[offset+2]
		y2 := data[offset+3]
		score := data[offset+4]
		classID := int(data[offset+5])

		if score < e.config.ConfThreshold {
			continue
		}

		// convert back to original image coordinates
		origX1 := max(0, int((x1-float32(params.padX))/params.scale))
		origY1 := max(0, int((y1-float32(params.padY))/params.scale))
		origX2 := min(params.origW, int((x2-float32(params.padX))/params.scale))
		origY2 := min(params.origH, int((y2-float32(params.padY))/params.scale))

		results = append(results, DetResult{
			ClassID: classID,
			Score:   score,
			Box:     image.Rect(origX1, origY1, origX2, origY2),
		})
	}

	return results
}
