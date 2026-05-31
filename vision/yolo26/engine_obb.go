package yolo26

import (
	"fmt"
	"image"
	"math"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// OBBEngine YOLO26-OBB Engine
type OBBEngine struct {
	session *ort.Session
	config  Config
}

// NewOBBEngine initializes the OBB engine
func NewOBBEngine(cfg Config) (*OBBEngine, error) {
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
		return nil, fmt.Errorf("failed to load obb model %s: %w", cfg.ModelPath, err)
	}

	return &OBBEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *OBBEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict executes rotated object detection
func (e *OBBEngine) Predict(img image.Image) ([]OBBResult, error) {
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

	// parse output [1, 300, 7]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}

	shape, err := outputValue.GetShape()
	if err != nil || len(shape) < 3 {
		return nil, fmt.Errorf("invalid output shape")
	}

	numBoxes := int(shape[1]) // 300
	numAttrs := int(shape[2]) // 7

	results := make([]OBBResult, 0)

	for i := 0; i < numBoxes; i++ {
		offset := i * numAttrs

		// [cx, cy, w, h, score, class_id, angle]
		cx := data[offset+0]
		cy := data[offset+1]
		w := data[offset+2]
		h := data[offset+3]
		score := data[offset+4]
		classID := int(data[offset+5])
		angle := data[offset+6]

		if score < e.config.ConfThreshold {
			continue
		}

		// get 4 corner points of the rotated rectangle
		corners := getRotatedCorners(cx, cy, w, h, angle)

		// map back to original image coordinates
		var origCorners [4]image.Point
		for j, pt := range corners {
			// boundary check
			ox := min(max(0, int(math.Round(float64(pt[0]/params.scale)))), params.origW)
			oy := min(max(0, int(math.Round(float64(pt[1]/params.scale)))), params.origH)
			origCorners[j] = image.Point{X: ox, Y: oy}
		}

		results = append(results, OBBResult{
			ClassID: classID,
			Score:   score,
			Corners: origCorners,
			Center: image.Point{
				X: (origCorners[0].X + origCorners[2].X) / 2,
				Y: (origCorners[0].Y + origCorners[2].Y) / 2,
			},
			Angle: angle,
		})
	}

	return results, nil
}
