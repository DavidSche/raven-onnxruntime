package yolov11

import (
	"fmt"
	"image"
	"math"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// OBBEngine YOLOv11-OBB Engine
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
	inputTensor, params, err := preprocess(img, e.config.InputSize, e.session, e.config.PreprocessConfig)
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
	defer ort.DestroyValues(outputValues)

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

	// Output Shaper: [1, 20, 21504]
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
func (e *OBBEngine) postprocess(data []float32, shape []int64, params imageParams) ([]OBBResult, error) {
	numChannels := int(shape[1])
	numAnchors := int(shape[2])

	// parse candidates
	candidates := e.parseCandidates(data, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]OBBResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]

		// recalculate the 4 rotated corner points
		corners := getRotatedCorners(cand.box[0], cand.box[1], cand.box[2], cand.box[3], cand.angle)

		// map back to original image coordinates
		origCorners := [4]image.Point{}
		for i, pt := range corners {
			ox := min(max(0, int(pt[0]/params.scale)), params.origW)
			oy := min(max(0, int(pt[1]/params.scale)), params.origH)

			origCorners[i] = image.Point{X: ox, Y: oy}
		}

		results = append(results, OBBResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Corners: origCorners,
			Center:  image.Point{X: (origCorners[0].X + origCorners[2].X) / 2, Y: (origCorners[0].Y + origCorners[2].Y) / 2},
			Angle:   cand.angle,
		})
	}

	return results, nil
}

// parseCandidates parses candidate boxes
func (e *OBBEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	// data
	// T [cx, cy, w, h, c1...c15, angle]

	var cands []candidate
	angleIdx := channels - 1

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
		angle := data[angleIdx*anchors+i]

		// get 4 corner points of the rotated rectangle
		corners := getRotatedCorners(cx, cy, w, h, angle)

		// find bounding rect min/max
		minX, minY := float32(math.MaxFloat32), float32(math.MaxFloat32)
		maxX, maxY := float32(-math.MaxFloat32), float32(-math.MaxFloat32)
		for _, pt := range corners {
			minX = min(pt[0], minX)
			maxX = max(pt[0], maxX)
			minY = min(pt[1], minY)
			maxY = max(pt[1], maxY)
		}
		origX1 := int(minX / params.scale)
		origY1 := int(minY / params.scale)
		origX2 := int(maxX / params.scale)
		origY2 := int(maxY / params.scale)

		cands = append(cands, candidate{
			box:     [4]float32{cx, cy, w, h},
			angle:   angle,
			origBox: image.Rect(origX1, origY1, origX2, origY2),
			score:   maxScore,
			classID: classID,
		})
	}
	return cands
}
