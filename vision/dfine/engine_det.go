package dfine

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

// DetEngine D-FINE / D-FINE-seg detection Engine.
//
// Supports two model variants:
//   - D-FINE (from D-FINE repo): 2 inputs  (images, orig_target_sizes)
//   - D-FINE-seg (from D-FINE-seg repo): 1 input  (input/image)
//
// Both output labels [B,N] int64, boxes [B,N,4] float32, scores [B,N] float32.
type DetEngine struct {
	session  *ort.Session
	config   Config
	runCount uint64

	// detected at engine creation time
	needsOrigTargetSizes bool // true for D-FINE models with 2 inputs
}

// NewDetEngine initializes the D-FINE detection engine.
func NewDetEngine(cfg Config) (*DetEngine, error) {
	ortlog.Infow("creating D-FINE detection engine",
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

	// Detect if model needs orig_target_sizes (D-FINE models have 2 inputs)
	needsOrigTargetSizes := len(session.InputNames) >= 2

	ortlog.Infow("D-FINE detection engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames,
		"needsOrigTargetSizes", needsOrigTargetSizes)

	return &DetEngine{
		session:              session,
		config:               cfg,
		needsOrigTargetSizes: needsOrigTargetSizes,
	}, nil
}

// NeedsOrigTargetSizes returns true when the ONNX model expects the second
// input (orig_target_sizes), which indicates the model internally rescales
// boxes to the original image coordinates.
func (e *DetEngine) NeedsOrigTargetSizes() bool { return e.needsOrigTargetSizes }

// Destroy releases all resources.
func (e *DetEngine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying D-FINE detection engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict executes detection inference on a single image.
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

// PredictBatch executes batch detection inference.
func (e *DetEngine) PredictBatch(imgs []image.Image) ([][]DetResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	startedAt := time.Now()

	// Preprocess
	preprocessStart := time.Now()
	inputTensor, paramsList, err := preprocessBatch(imgs, e.config.InputSize, e.session, e.config.PreprocessConfig)
	if err != nil {
		return nil, fmt.Errorf("preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()
	preprocessElapsed := time.Since(preprocessStart)

	// Build input values
	inputName := e.session.InputNames[0]
	inputValues := map[string]*ort.Value{
		inputName: inputTensor,
	}

	// For D-FINE models that require orig_target_sizes
	if e.needsOrigTargetSizes && len(e.session.InputNames) >= 2 {
		origTargetName := e.session.InputNames[1]
		origTargetData := make([]int64, len(imgs)*2)
		for i, params := range paramsList {
			origTargetData[i*2+0] = int64(params.origH)
			origTargetData[i*2+1] = int64(params.origW)
		}
		origTensor, err := e.session.NewTensor([]int64{int64(len(imgs)), 2}, origTargetData)
		if err != nil {
			return nil, fmt.Errorf("failed to create orig_target_sizes tensor: %w", err)
		}
		defer origTensor.Destroy()
		inputValues[origTargetName] = origTensor
	}

	// Run inference
	runStart := time.Now()
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer ort.DestroyValues(outputValues)
	runElapsed := time.Since(runStart)

	// Resolve output names dynamically
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
		return nil, fmt.Errorf("D-FINE model requires boxes and scores outputs, got %v", e.session.OutputNames)
	}

	boxesOut, ok0 := outputValues[boxesName]
	scoresOut, ok1 := outputValues[scoresName]
	if !ok0 || !ok1 || boxesOut == nil || scoresOut == nil {
		return nil, fmt.Errorf("D-FINE model output boxes or scores is nil")
	}

	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	scoresShape, err := scoresOut.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get scores shape: %w", err)
	}

	if len(boxesShape) != 3 || len(scoresShape) < 2 {
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v scores=%v", boxesShape, scoresShape)
	}

	batchSize := int(boxesShape[0])
	numDetections := int(boxesShape[1])

	boxesRaw, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	scoresRaw, err := ort.GetTensorData[float32](scoresOut)
	if err != nil {
		return nil, fmt.Errorf("failed to get scores data: %w", err)
	}

	boxesCopy := make([]float32, len(boxesRaw))
	copy(boxesCopy, boxesRaw)
	scoresCopy := make([]float32, len(scoresRaw))
	copy(scoresCopy, scoresRaw)

	// Extract labels (optional for some model variants)
	var labelsData []int64
	if labelsName != "" {
		labelsOut := outputValues[labelsName]
		if labelsOut != nil {
			labelsRaw, err := ort.GetTensorData[int64](labelsOut)
			if err == nil {
				labelsData = make([]int64, len(labelsRaw))
				copy(labelsData, labelsRaw)
			}
		}
	}

	// Compute strides
	boxStride := numDetections * 4
	scoreStride := numDetections
	labelStride := numDetections

	if batchSize != len(imgs) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(imgs))
	}

	// Post-process each image in the batch
	postprocessStart := time.Now()
	results := make([][]DetResult, batchSize)
	for i := 0; i < batchSize; i++ {
		var labelSlice []int64
		if len(labelsData) >= (i+1)*labelStride {
			labelSlice = labelsData[i*labelStride : (i+1)*labelStride]
		}

		// For D-FINE models with orig_target_sizes, boxes are already in original
		// image coordinates. For single-input models, boxes are in model input
		// coordinates and need scaling.
		results[i] = e.postprocess(
			boxesCopy[i*boxStride:(i+1)*boxStride],
			scoresCopy[i*scoreStride:(i+1)*scoreStride],
			labelSlice,
			numDetections,
			paramsList[i],
			e.needsOrigTargetSizes, // if true, boxes are already in orig coords
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

func (e *DetEngine) postprocess(boxes []float32, scores []float32, labels []int64,
	numDetections int, params imageParams, boxesInOrigCoords bool) []DetResult {

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

	// Sort by score descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > e.config.MaxDetections {
		candidates = candidates[:e.config.MaxDetections]
	}

	results := make([]DetResult, 0, len(candidates))
	for _, cand := range candidates {
		var box image.Rectangle
		if boxesInOrigCoords {
			// D-FINE model with orig_target_sizes: boxes already in original coords
			box = image.Rect(
				max(0, int(cand.x1)),
				max(0, int(cand.y1)),
				min(params.origW, int(cand.x2)),
				min(params.origH, int(cand.y2)),
			)
		} else {
			// D-FINE-seg model (single input): scale from model input to original
			box = boxXyxyToOrigScale(cand.x1, cand.y1, cand.x2, cand.y2,
				e.config.InputSize, params.origW, params.origH)
		}

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

	ortlog.Infow("dfine det timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}
