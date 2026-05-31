package yolo26

import (
	"fmt"
	"image"
	"image/color"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// SegEngine YOLO26-seg Engine
type SegEngine struct {
	session *ort.Session
	config  Config
}

// NewSegEngine initializes the segmentation engine
func NewSegEngine(cfg Config) (*SegEngine, error) {
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

	return &SegEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *SegEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict executes segmentation inference
func (e *SegEngine) Predict(img image.Image) ([]SegResult, error) {
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
	defer func() {
		for _, v := range outputValues {
			v.Destroy()
		}
	}()

	// get outputs (compatible with different output names)
	if len(e.session.OutputNames) < 2 {
		return nil, fmt.Errorf("segmentation model requires at least 2 outputs, got %d", len(e.session.OutputNames))
	}
	out0, ok0 := outputValues[e.session.OutputNames[0]]
	out1, ok1 := outputValues[e.session.OutputNames[1]]
	if !ok0 || !ok1 || out0 == nil || out1 == nil {
		return nil, fmt.Errorf("segmentation model output does not exist")
	}

	// output0: Detections [1,300,38]
	// output1: Mask Protos [1, 32, 160, 160]

	// postprocess
	return e.postprocess(out0, out1, params)
}

// postprocess performs post-processing
func (e *SegEngine) postprocess(out0, out1 *ort.Value, params imageParams) ([]SegResult, error) {
	data0, err := ort.GetTensorData[float32](out0)
	if err != nil {
		return nil, fmt.Errorf("failed to get data: %w", err)
	}
	data1, err := ort.GetTensorData[float32](out1)
	if err != nil {
		return nil, fmt.Errorf("failed to get mask protos: %w", err)
	}
	shape1, _ := out1.GetShape()
	protoC, protoH, protoW := int(shape1[1]), int(shape1[2]), int(shape1[3])

	const (
		numDetections = 300
		dimTotal      = 38
		indexMask     = 6
	)

	var results []SegResult

	// [x1, y1, x2, y2, score, class, mask_coeffs...]
	for i := 0; i < numDetections; i++ {
		offset := i * dimTotal

		score := data0[offset+4]
		if score < e.config.ConfThreshold {
			continue
		}

		x1 := data0[offset+0]
		y1 := data0[offset+1]
		x2 := data0[offset+2]
		y2 := data0[offset+3]
		classID := int(data0[offset+5])

		// map back to original image
		origX1 := max(0, int(x1/params.scale))
		origY1 := max(0, int(y1/params.scale))
		origX2 := min(params.origW, int(x2/params.scale))
		origY2 := min(params.origH, int(y2/params.scale))
		origBox := image.Rect(origX1, origY1, origX2, origY2)

		// extract 32 mask coefficients
		coeffs := data0[offset+indexMask : offset+indexMask+32]

		// decode mask
		mask := e.decodeMask(origBox, coeffs, data1, protoC, protoH, protoW, params)

		results = append(results, SegResult{
			ClassID: classID,
			Score:   score,
			Box:     origBox,
			Mask:    mask,
		})
	}

	return results, nil
}

// decodeMask decodes the mask
//
// # Params:
//
//	origBox: detection box on the original image
//	maskCoeffs: 32 mask coefficients for the current detection box
//	protos: prototype data from model Output1
//	c, h, w: prototype dimensions (32, 160, 160)
//	params: image size scaling parameters
func (e *SegEngine) decodeMask(origBox image.Rectangle, maskCoeffs []float32, protos []float32, c, h, w int, params imageParams) *image.Gray {
	finalMask := image.NewGray(image.Rect(0, 0, params.origW, params.origH))

	// mask prototype scaling ratio relative to InputSize(640)
	maskStride := float32(e.config.InputSize) / float32(w)

	for y := origBox.Min.Y; y < origBox.Max.Y; y++ {
		for x := origBox.Min.X; x < origBox.Max.X; x++ {
			// map back to 640 scale
			inputX := float32(x) * params.scale
			inputY := float32(y) * params.scale

			// map back to 160 mask scale
			mx := int(inputX / maskStride)
			my := int(inputY / maskStride)

			if mx >= 0 && mx < w && my >= 0 && my < h {
				// compute mask value for this pixel (dot product)
				sum := float32(0.0)
				for k := 0; k < c; k++ {
					sum += maskCoeffs[k] * protos[k*h*w+my*w+mx]
				}

				if sigmoid(sum) > e.config.MaskThreshold {
					finalMask.SetGray(x, y, color.Gray{Y: 255})
				}
			}
		}
	}
	return finalMask
}
