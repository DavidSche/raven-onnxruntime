package yolov11

import (
	"fmt"
	"image"
	"log"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// DetEngine YOLOv11-det Engine
type DetEngine struct {
	session *ort.Session
	config  Config
}

// NewDetEngine initializes the detection engine
func NewDetEngine(cfg Config) (*DetEngine, error) {
	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}
	// initialize ONNX
	if err := oc.New(); err != nil {
		return nil, err
	}

	// create session
	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	oc.Destroy()
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &DetEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *DetEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict executes detection inference
func (e *DetEngine) Predict(img image.Image) ([]DetResult, error) {
	// preprocess
	inputTensor, params, err := preprocess(img, e.config.InputSize, e.session)
	if err != nil {
		return nil, fmt.Errorf("preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()

	// get actual input name (compatible with different models)
	if len(e.session.InputNames) == 0 {
		return nil, fmt.Errorf("model has no input")
	}
	inputName := e.session.InputNames[0]

	// inference
	inputValues := map[string]*ort.Value{
		inputName: inputTensor,
	}
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// get first output (compatible with different output names)
	if len(e.session.OutputNames) == 0 {
		return nil, fmt.Errorf("model has no output")
	}
	outputName := e.session.OutputNames[0]
	outputValue, ok := outputValues[outputName]
	if !ok || outputValue == nil {
		return nil, fmt.Errorf("output %q does not exist", outputName)
	}
	defer outputValue.Destroy()

	// Output Shape: [1, 84, 8400]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}
	shape, err := outputValue.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get output shape: %w", err)
	}

	// postprocess
	return e.postprocess(data, shape, params)
}

// postprocess performs post-processing
func (e *DetEngine) postprocess(data []float32, shape []int64, params imageParams) ([]DetResult, error) {
	numChannels := int(shape[1]) // 4 (box) + 80 (cls) = 84
	numAnchors := int(shape[2])  // 8400

	// parse candidates
	candidates := e.parseCandidates(data, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]DetResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]
		results = append(results, DetResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     cand.origBox,
		})
	}

	return results, nil
}

// parseCandidates parses candidate boxes
func (e *DetEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	var cands []candidate

	// check channel count
	expectedChannels := 4 + e.config.NumClasses
	if channels != expectedChannels {
		log.Printf("warning: channel count mismatch: got %d, expected %d", channels, expectedChannels)
		return cands
	}

	for i := 0; i < anchors; i++ {
		// find max class score
		maxScore := float32(0.0)
		classID := -1
		for c := 0; c < e.config.NumClasses; c++ {
			score := data[(4+c)*anchors+i]
			if score > maxScore {
				maxScore = score
				classID = c
			}
		}
		if maxScore < e.config.ConfThreshold {
			continue
		}

		// extract coordinates
		cx := data[0*anchors+i]
		cy := data[1*anchors+i]
		w := data[2*anchors+i]
		h := data[3*anchors+i]

		// convert back to original image rectangle coordinates
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2
		origX1 := max(0, int(x1/params.scale))
		origY1 := max(0, int(y1/params.scale))
		origX2 := min(params.origW, int(x2/params.scale))
		origY2 := min(params.origH, int(y2/params.scale))

		cands = append(cands, candidate{
			box:     [4]float32{x1, y1, x2, y2},
			origBox: image.Rect(origX1, origY1, origX2, origY2),
			score:   maxScore,
			classID: classID,
		})
	}
	return cands
}
