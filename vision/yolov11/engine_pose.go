package yolov11

import (
	"fmt"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"log"
)

// PoseEngine YOLOv11-pose Engine
type PoseEngine struct {
	session *ort.Session
	config  Config
}

// NewPoseEngine initializes the pose estimation engine
func NewPoseEngine(cfg Config) (*PoseEngine, error) {
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

	return &PoseEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *PoseEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict executes pose estimation
func (e *PoseEngine) Predict(img image.Image) ([]PoseResult, error) {
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

	// Output Shaper: [1, 56, 8400]
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
func (e *PoseEngine) postprocess(data []float32, shape []int64, params imageParams) ([]PoseResult, error) {
	numChannels := int(shape[1])
	numAnchors := int(shape[2])

	// parse candidates
	candidates := e.parseCandidates(data, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]PoseResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]
		// decode keypoints
		kpts := e.decodeKeyPoints(cand.rawKeyPoints, params)

		results = append(results, PoseResult{
			ClassID:   cand.classID,
			Score:     cand.score,
			Box:       cand.origBox,
			KeyPoints: kpts,
		})
	}

	return results, nil
}

// parseCandidates parses candidate boxes
func (e *PoseEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	// data
	// T [cx, cy, w, h, c1, x1,y1,conf1...x17,y17,conf17]

	var cands []candidate

	// check channel count
	expectedChannels := 4 + e.config.NumClasses + e.config.NumKeyPoints*3
	if channels != expectedChannels {
		log.Printf("warning: channel count mismatch: got %d, expected %d", channels, expectedChannels)
		return cands
	}

	kptStartIdx := 4 + e.config.NumClasses
	// each keypoint contains 3 floats: x, y, conf
	numKptValues := e.config.NumKeyPoints * 3

	for i := 0; i < anchors; i++ {
		// find max classification score
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
		origX1 := min(max(0, int(x1/params.scale)), params.origW)
		origY1 := min(max(0, int(y1/params.scale)), params.origH)
		origX2 := min(max(0, int(x2/params.scale)), params.origW)
		origY2 := min(max(0, int(y2/params.scale)), params.origH)

		// store raw keypoints data
		rawKpts := make([]float32, numKptValues)
		for k := 0; k < numKptValues; k++ {
			rawKpts[k] = data[(kptStartIdx+k)*anchors+i]
		}

		cands = append(cands, candidate{
			box:          [4]float32{x1, y1, x2, y2},
			origBox:      image.Rect(origX1, origY1, origX2, origY2),
			score:        maxScore,
			classID:      classID,
			rawKeyPoints: rawKpts,
		})
	}
	return cands
}

// decodeKeyPoints decodes keypoint coordinates
func (e *PoseEngine) decodeKeyPoints(raw []float32, params imageParams) []KeyPoint {
	kpts := make([]KeyPoint, e.config.NumKeyPoints)

	for i := 0; i < e.config.NumKeyPoints; i++ {
		idx := i * 3
		x := raw[idx]
		y := raw[idx+1]
		conf := raw[idx+2]

		// map coordinates back to original image
		origX := min(max(0, int(x/params.scale)), params.origW)
		origY := min(max(0, int(y/params.scale)), params.origH)

		kpts[i] = KeyPoint{
			X:     origX,
			Y:     origY,
			Score: conf,
		}
	}
	return kpts
}
