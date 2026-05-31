package yolov11

import (
	"fmt"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"sort"
)

// ClsEngine YOLOv11-cls Engine
type ClsEngine struct {
	session *ort.Session
	config  Config
}

// NewClsEngine initializes the classification engine
func NewClsEngine(cfg Config) (*ClsEngine, error) {
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

	return &ClsEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *ClsEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict executes classification inference
//
// # Params:
//
//	img: image to classify
//	topK: number of top-K classes with highest probability to return
func (e *ClsEngine) Predict(img image.Image, topK int) ([]ClassResult, error) {
	// preprocess
	inputTensor, _, err := preprocess(img, e.config.InputSize, e.session)
	if err != nil {
		return nil, fmt.Errorf("preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()

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

	if len(e.session.OutputNames) == 0 {
		return nil, fmt.Errorf("model has no output")
	}
	outputName := e.session.OutputNames[0]
	outputValue, ok := outputValues[outputName]
	if !ok || outputValue == nil {
		return nil, fmt.Errorf("output %q does not exist", outputName)
	}
	defer outputValue.Destroy()

	// Output Shape: [1, 1000]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}

	// postprocess
	return e.postprocess(data, topK), nil
}

// postprocess performs post-processing
func (e *ClsEngine) postprocess(logits []float32, topK int) []ClassResult {
	// convert to classification results
	results := make([]ClassResult, len(logits))
	for i, score := range logits {
		results[i] = ClassResult{
			ClassID: i,
			Score:   score,
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results[:min(topK, len(results))]
}
