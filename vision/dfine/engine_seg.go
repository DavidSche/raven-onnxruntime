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

// SegEngine D-FINE-seg segmentation Engine.
//
// The D-FINE-seg segmentation model has a single input (input) and outputs:
//   - labels  [B, K]      int64    – top-K class IDs
//   - boxes   [B, K, 4]   float32  – xyxy in model input coordinates
//   - scores  [B, K]      float32  – sigmoid confidence scores
//   - masks   [B, K, Hm, Wm] float32 – sigmoid-applied mask logits (optional)
//
// Supports dynamic batch via the ONNX model's dynamic_axes.
type SegEngine struct {
	session  *ort.Session
	config   Config
	runCount uint64
}

// NewSegEngine initializes the D-FINE-seg segmentation engine.
func NewSegEngine(cfg Config) (*SegEngine, error) {
	ortlog.Infow("creating D-FINE-seg segmentation engine",
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

	ortlog.Infow("D-FINE-seg segmentation engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)

	return &SegEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources.
func (e *SegEngine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying D-FINE-seg segmentation engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict executes segmentation inference on a single image.
func (e *SegEngine) Predict(img image.Image) ([]SegResult, error) {
	results, err := e.PredictBatch([]image.Image{img})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return results[0], nil
}

// PredictBatch executes batch segmentation inference.
// The ONNX model supports dynamic batch, so multiple images are processed
// in a single session.Run call for optimal throughput.
func (e *SegEngine) PredictBatch(imgs []image.Image) ([][]SegResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	startedAt := time.Now()

	// Preprocess batch
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

	// Run inference
	runStart := time.Now()
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer ort.DestroyValues(outputValues)
	runElapsed := time.Since(runStart)

	// Resolve output names: labels, boxes, scores, masks
	var labelsName, boxesName, scoresName, masksName string
	for _, name := range e.session.OutputNames {
		switch name {
		case "labels":
			labelsName = name
		case "boxes":
			boxesName = name
		case "scores":
			scoresName = name
		case "masks":
			masksName = name
		}
	}

	if boxesName == "" || scoresName == "" {
		return nil, fmt.Errorf("D-FINE-seg model requires boxes and scores outputs, got %v", e.session.OutputNames)
	}

	boxesOut, ok0 := outputValues[boxesName]
	scoresOut, ok1 := outputValues[scoresName]
	if !ok0 || !ok1 || boxesOut == nil || scoresOut == nil {
		return nil, fmt.Errorf("D-FINE-seg model output boxes or scores is nil")
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

	if batchSize != len(imgs) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(imgs))
	}

	// Copy tensor data to Go memory (must copy before DestroyValues)
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

	// Copy labels
	var labelsCopy []int64
	if labelsName != "" {
		labelsOut := outputValues[labelsName]
		if labelsOut != nil {
			labelsRaw, err := ort.GetTensorData[int64](labelsOut)
			if err == nil {
				labelsCopy = make([]int64, len(labelsRaw))
				copy(labelsCopy, labelsRaw)
			}
		}
	}

	// Copy masks
	var masksData []float32
	var maskH, maskW int
	if masksName != "" {
		masksOut := outputValues[masksName]
		if masksOut != nil {
			masksShape, err := masksOut.GetShape()
			if err == nil && len(masksShape) == 4 {
				maskH = int(masksShape[2])
				maskW = int(masksShape[3])
			}
			masksRaw, err := ort.GetTensorData[float32](masksOut)
			if err == nil {
				masksData = make([]float32, len(masksRaw))
				copy(masksData, masksRaw)
			}
		}
	}

	// Post-process each image in the batch
	postprocessStart := time.Now()

	boxStride := numDetections * 4
	scoreStride := numDetections
	labelStride := numDetections
	maskPlaneSize := maskH * maskW
	maskStride := numDetections * maskPlaneSize

	results := make([][]SegResult, batchSize)
	for i := 0; i < batchSize; i++ {
		boxSlice := boxesCopy[i*boxStride : (i+1)*boxStride]
		scoreSlice := scoresCopy[i*scoreStride : (i+1)*scoreStride]

		var labelSlice []int64
		if labelsCopy != nil && len(labelsCopy) >= (i+1)*labelStride {
			labelSlice = labelsCopy[i*labelStride : (i+1)*labelStride]
		}

		var maskSlice []float32
		if masksData != nil && len(masksData) >= (i+1)*maskStride {
			maskSlice = masksData[i*maskStride : (i+1)*maskStride]
		}

		results[i] = e.postprocessSeg(
			boxSlice, scoreSlice, labelSlice, maskSlice,
			numDetections, maskH, maskW,
			paramsList[i],
		)
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(batchSize, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

// segCandidate holds intermediate filtered detection data.
type segCandidate struct {
	classID int
	score   float32
	x1, y1  float32
	x2, y2  float32
	index   int // query index for mask lookup
}

// postprocessSeg filters and builds results for a single image in the batch.
func (e *SegEngine) postprocessSeg(
	boxes []float32, scores []float32, labels []int64, masks []float32,
	numDetections int, maskH, maskW int, params imageParams,
) []SegResult {
	candidates := make([]segCandidate, 0, numDetections)
	for i := 0; i < numDetections; i++ {
		score := scores[i]
		if score < e.config.ConfThreshold {
			continue
		}

		boxOffset := i * 4
		classID := -1
		if labels != nil && i < len(labels) {
			classID = int(labels[i])
		}

		candidates = append(candidates, segCandidate{
			classID: classID,
			score:   score,
			x1:      boxes[boxOffset+0],
			y1:      boxes[boxOffset+1],
			x2:      boxes[boxOffset+2],
			y2:      boxes[boxOffset+3],
			index:   i,
		})
	}

	// Sort by score descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > e.config.MaxDetections {
		candidates = candidates[:e.config.MaxDetections]
	}

	// Build results with masks
	noMasks := len(masks) == 0
	maskPlaneSize := maskH * maskW

	results := make([]SegResult, 0, len(candidates))
	for _, cand := range candidates {
		// Scale box from model input to original image coordinates
		box := boxXyxyToOrigScale(cand.x1, cand.y1, cand.x2, cand.y2,
			e.config.InputSize, params.origW, params.origH)

		var mask *image.Gray
		if !noMasks && maskPlaneSize > 0 {
			maskOffset := cand.index * maskPlaneSize
			if maskOffset+maskPlaneSize <= len(masks) {
				maskSlice := masks[maskOffset : maskOffset+maskPlaneSize]
				mask = resizeMask(maskSlice, maskH, maskW, box, params.origW, params.origH, e.config.MaskThreshold)
			}
		}

		results = append(results, SegResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     box,
			Mask:    mask,
		})
	}

	return results
}

func (e *SegEngine) logPredictTimings(batchSize int, preprocessElapsed, runElapsed, postprocessElapsed, totalElapsed time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("dfine seg timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}
