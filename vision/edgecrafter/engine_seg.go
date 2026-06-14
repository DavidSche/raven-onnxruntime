package edgecrafter

import (
	"fmt"
	"image"
	"sort"
	"strings"
	"time"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// SegEngine is the EdgeCrafter segmentation engine.
// It expects ONNX models exported in "raw" mode with outputs:
//   - pred_logits [N, num_queries, num_classes]
//   - pred_boxes  [N, num_queries, 4] (cxcywh normalized)
//   - pred_masks  [N, num_queries, H, W]
type SegEngine struct {
	session *ort.Session
	config  Config
}

// resolveSegOutputs resolves boxes, logits, and masks outputs from ONNX model output values.
func resolveSegOutputs(outputValues map[string]*ort.Value) (*ort.Value, *ort.Value, *ort.Value, error) {
	outputShapes := make(map[string][]int64)
	for name, v := range outputValues {
		if v == nil {
			continue
		}
		shape, err := v.GetShape()
		if err != nil {
			continue
		}
		outputShapes[name] = shape
	}
	ortlog.Debugw("seg engine output shapes", "shapes", outputShapes)

	var boxesOut, logitsOut, masksOut *ort.Value

	// Strategy 1: Match by name first, then validate shape
	for name, shape := range outputShapes {
		v := outputValues[name]
		switch name {
		case "pred_boxes":
			if len(shape) == 3 && shape[2] == 4 {
				boxesOut = v
			}
		case "pred_logits":
			if len(shape) == 3 {
				logitsOut = v
			}
		case "pred_masks":
			if len(shape) == 4 {
				masksOut = v
			}
		}
	}

	// Strategy 2: Fallback to shape-based detection for any missing outputs
	if boxesOut == nil || logitsOut == nil || masksOut == nil {
		for name, shape := range outputShapes {
			v := outputValues[name]
			switch len(shape) {
			case 3:
				if shape[2] == 4 && boxesOut == nil {
					boxesOut = v
				} else if logitsOut == nil {
					logitsOut = v
				}
			case 4:
				if masksOut == nil {
					masksOut = v
				}
			}
		}
	}

	if boxesOut == nil || logitsOut == nil || masksOut == nil {
		shapeInfo := make([]string, 0, len(outputShapes))
		for name, shape := range outputShapes {
			shapeInfo = append(shapeInfo, fmt.Sprintf("%s=%v", name, shape))
		}
		for _, v := range outputValues {
			v.Destroy()
		}
		return nil, nil, nil, fmt.Errorf("segmentation model requires boxes(3D,4), logits(3D) and masks(4D) outputs, got [%s]", strings.Join(shapeInfo, ", "))
	}

	return boxesOut, logitsOut, masksOut, nil
}

// NewSegEngine creates a new EdgeCrafter segmentation engine.
func NewSegEngine(cfg Config) (*SegEngine, error) {
	ortlog.Infow("creating EdgeCrafter segmentation engine",
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

	ortlog.Infow("EdgeCrafter segmentation engine created successfully",
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
		ortlog.Infow("destroying EdgeCrafter segmentation engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict runs segmentation on a single image.
func (e *SegEngine) Predict(img image.Image) ([]SegResult, error) {
	inputTensor, params, err := preprocess(img, e.config.InputSize, e.session)
	if err != nil {
		return nil, fmt.Errorf("preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()

	if len(e.session.InputNames) == 0 {
		return nil, fmt.Errorf("model has no input")
	}
	inputName := e.session.InputNames[0]

	inputValues := map[string]*ort.Value{
		inputName: inputTensor,
	}
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	boxesOut, logitsOut, masksOut, err := resolveSegOutputs(outputValues)
	if err != nil {
		return nil, err
	}

	return e.postprocess(boxesOut, logitsOut, masksOut, params)
}

func (e *SegEngine) postprocess(boxesOut, logitsOut, masksOut *ort.Value, params imageParams) ([]SegResult, error) {
	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}
	masksShape, err := masksOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get masks shape: %w", err)
	}

	if len(boxesShape) != 3 || len(logitsShape) != 3 || len(masksShape) != 4 {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v logits=%v masks=%v", boxesShape, logitsShape, masksShape)
	}

	numDetections := int(boxesShape[1])
	numClasses := int(logitsShape[2])
	maskH := int(masksShape[2])
	maskW := int(masksShape[3])

	boxesRaw, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	logitsRaw, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}
	masksRaw, err := ort.GetTensorData[float32](masksOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get masks data: %w", err)
	}

	boxesData := make([]float32, len(boxesRaw))
	copy(boxesData, boxesRaw)
	logitsData := make([]float32, len(logitsRaw))
	copy(logitsData, logitsRaw)
	masksData := make([]float32, len(masksRaw))
	copy(masksData, masksRaw)

	boxesOut.Destroy()
	logitsOut.Destroy()
	masksOut.Destroy()

	type segCandidate struct {
		classID int
		score   float32
		cx, cy  float32
		w, h    float32
		index   int
	}

	candidates := make([]segCandidate, 0, numDetections)
	for i := 0; i < numDetections; i++ {
		boxOffset := i * 4
		logitOffset := i * numClasses

		maxScore := float32(0.0)
		classID := -1
		for c := 0; c < numClasses; c++ {
			s := sigmoid(logitsData[logitOffset+c])
			if s > maxScore {
				maxScore = s
				classID = c
			}
		}

		if maxScore < e.config.ConfThreshold {
			continue
		}

		candidates = append(candidates, segCandidate{
			classID: classID,
			score:   maxScore,
			cx:      boxesData[boxOffset+0],
			cy:      boxesData[boxOffset+1],
			w:       boxesData[boxOffset+2],
			h:       boxesData[boxOffset+3],
			index:   i,
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > e.config.MaxDetections {
		candidates = candidates[:e.config.MaxDetections]
	}

	maskPlaneSize := maskH * maskW

	results := make([]SegResult, 0, len(candidates))
	for _, cand := range candidates {
		box := boxCxcywhToXyxy(cand.cx, cand.cy, cand.w, cand.h, params.origW, params.origH)

		maskOffset := cand.index * maskPlaneSize
		maskSlice := masksData[maskOffset : maskOffset+maskPlaneSize]
		mask := resizeMask(maskSlice, maskH, maskW, box, params.origW, params.origH, e.config.MaskThreshold)

		results = append(results, SegResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     box,
			Mask:    mask,
		})
	}

	return results, nil
}

// PredictBatch runs segmentation on a batch of images.
func (e *SegEngine) PredictBatch(imgs []image.Image) ([][]SegResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	if !e.config.DynamicBatch {
		results := make([][]SegResult, len(imgs))
		for i, img := range imgs {
			res, err := e.Predict(img)
			if err != nil {
				return nil, fmt.Errorf("image %d prediction failed: %w", i, err)
			}
			results[i] = res
		}
		return results, nil
	}

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

	boxesOut, logitsOut, masksOut, err := resolveSegOutputs(outputValues)
	if err != nil {
		return nil, err
	}

	results, err := e.postprocessBatch(boxesOut, logitsOut, masksOut, paramsList)
	if err != nil {
		return nil, err
	}

	ortlog.Debugw("edgecrafter seg batch inference completed",
		"batchSize", len(imgs),
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
	)

	return results, nil
}

func (e *SegEngine) postprocessBatch(boxesOut, logitsOut, masksOut *ort.Value, paramsList []imageParams) ([][]SegResult, error) {
	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}
	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}
	masksShape, err := masksOut.GetShape()
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get masks shape: %w", err)
	}

	if len(boxesShape) != 3 || len(logitsShape) != 3 || len(masksShape) != 4 {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("unexpected output shapes: boxes=%v logits=%v masks=%v", boxesShape, logitsShape, masksShape)
	}

	batchSize := int(boxesShape[0])
	numDetections := int(boxesShape[1])
	numClasses := int(logitsShape[2])
	maskH := int(masksShape[2])
	maskW := int(masksShape[3])

	boxesRaw, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	logitsRaw, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}
	masksRaw, err := ort.GetTensorData[float32](masksOut)
	if err != nil {
		boxesOut.Destroy()
		logitsOut.Destroy()
		masksOut.Destroy()
		return nil, fmt.Errorf("failed to get masks data: %w", err)
	}

	boxesData := make([]float32, len(boxesRaw))
	copy(boxesData, boxesRaw)
	logitsData := make([]float32, len(logitsRaw))
	copy(logitsData, logitsRaw)
	masksData := make([]float32, len(masksRaw))
	copy(masksData, masksRaw)

	boxesOut.Destroy()
	logitsOut.Destroy()
	masksOut.Destroy()

	if batchSize != len(paramsList) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(paramsList))
	}

	boxStride := numDetections * 4
	logitStride := numDetections * numClasses
	maskPlaneSize := maskH * maskW
	maskStride := numDetections * maskPlaneSize

	allResults := make([][]SegResult, batchSize)
	for b := 0; b < batchSize; b++ {
		params := paramsList[b]

		type segCandidate struct {
			classID int
			score   float32
			cx, cy  float32
			w, h    float32
			index   int
		}

		candidates := make([]segCandidate, 0, numDetections)
		for i := 0; i < numDetections; i++ {
			boxOffset := b*boxStride + i*4
			logitOffset := b*logitStride + i*numClasses

			maxScore := float32(0.0)
			classID := -1
			for c := 0; c < numClasses; c++ {
				s := sigmoid(logitsData[logitOffset+c])
				if s > maxScore {
					maxScore = s
					classID = c
				}
			}

			if maxScore < e.config.ConfThreshold {
				continue
			}

			candidates = append(candidates, segCandidate{
				classID: classID,
				score:   maxScore,
				cx:      boxesData[boxOffset+0],
				cy:      boxesData[boxOffset+1],
				w:       boxesData[boxOffset+2],
				h:       boxesData[boxOffset+3],
				index:   i,
			})
		}

		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].score > candidates[j].score
		})

		if len(candidates) > e.config.MaxDetections {
			candidates = candidates[:e.config.MaxDetections]
		}

		results := make([]SegResult, 0, len(candidates))
		for _, cand := range candidates {
			box := boxCxcywhToXyxy(cand.cx, cand.cy, cand.w, cand.h, params.origW, params.origH)

			maskOffset := b*maskStride + cand.index*maskPlaneSize
			maskSlice := masksData[maskOffset : maskOffset+maskPlaneSize]
			mask := resizeMask(maskSlice, maskH, maskW, box, params.origW, params.origH, e.config.MaskThreshold)

			results = append(results, SegResult{
				ClassID: cand.classID,
				Score:   cand.score,
				Box:     box,
				Mask:    mask,
			})
		}

		allResults[b] = results
	}

	return allResults, nil
}
