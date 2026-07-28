package yolov11

import (
	"fmt"
	"image"
	"image/color"
	"log"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// SegEngine YOLOv11-seg Engine
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

	// output0: Detections [1, 116, 8400]
	// output1: Mask Protos [1, 32, 160, 160]

	// postprocess
	return e.postprocess(out0, out1, params)
}

// postprocess performs post-processing
func (e *SegEngine) postprocess(out0, out1 *ort.Value, params imageParams) ([]SegResult, error) {
	data0, err := ort.GetTensorData[float32](out0)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}
	shape0, err := out0.GetShape() // [1, 116, 8400]
	if err != nil {
		return nil, fmt.Errorf("failed to get output shape: %w", err)
	}
	numChannels := int(shape0[1]) // 4 (box) + 80 (cls) + 32 (mask) = 116
	numAnchors := int(shape0[2])  // 8400

	data1, err := ort.GetTensorData[float32](out1)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}
	shape1, err := out1.GetShape() // [1, 32, 160, 160]
	if err != nil {
		return nil, fmt.Errorf("failed to get output shape: %w", err)
	}
	protoC, protoH, protoW := int(shape1[1]), int(shape1[2]), int(shape1[3])

	// parse candidates
	candidates := e.parseCandidates(data0, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]SegResult, 0, len(keptIndices))

	// generate masks
	for _, idx := range keptIndices {
		cand := candidates[idx]

		// generate binary mask
		mask := e.decodeMask(cand, data1, protoC, protoH, protoW, params)
		results = append(results, SegResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     cand.origBox, // image.Rectangle
			Mask:    mask,
		})
	}

	return results, nil
}

// parseCandidates parses candidate boxes
//
// # Params:
//
//	data: model output array
//		[x1, x2 ..., x8400]
//		[y1, y2 ..., y8400]
//		[w1, w2 ..., w8400]
//		[h1, h2 ..., h8400]
//		[c1_1, c1_2 ..., c1_8400]
//		[c2_1, c2_2 ..., c2_8400]
//		...
//		[m1, m2 ..., m8400]
//	channels: number of output channels
//	anchors: number of output anchors
//	params: image size info
func (e *SegEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	var cands []candidate

	// check channel count
	expectedChannels := 4 + e.config.NumClasses + e.config.NumMaskCoeffs
	if channels != expectedChannels {
		log.Printf("warning: channel count mismatch: got %d, expected %d", channels, expectedChannels)
		return cands
	}

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

		// extract mask coefficients
		coeffs := make([]float32, e.config.NumMaskCoeffs)
		for j := 0; j < e.config.NumMaskCoeffs; j++ {
			coeffs[j] = data[(4+e.config.NumClasses+j)*anchors+i]
		}

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
			box:        [4]float32{x1, y1, x2, y2},
			origBox:    image.Rect(origX1, origY1, origX2, origY2),
			score:      maxScore,
			classID:    classID,
			maskCoeffs: coeffs,
		})
	}
	return cands
}

// decodeMask decodes the mask
//
// # Params:
//
//	cand: candidate result
//	protos: mask prototype data from model output
//	c: number of prototype mask channels
//	h: height of a single prototype mask
//	w: width of a single prototype mask
//	params: image size info
func (e *SegEngine) decodeMask(cand candidate, protos []float32, c, h, w int, params imageParams) *image.Gray {
	finalMask := image.NewGray(image.Rect(0, 0, params.origW, params.origH))

	// mask prototype scaling ratio relative to InputSize(640)
	maskStride := float32(e.config.InputSize) / float32(w)

	// iterate over the box region on the original image
	origBox := cand.origBox
	coeffs := cand.maskCoeffs
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
					sum += coeffs[k] * protos[k*h*w+my*w+mx]
				}

				if sigmoid(sum) > e.config.MaskThreshold {
					finalMask.SetGray(x, y, color.Gray{Y: 255})
				}
			}
		}
	}
	return finalMask
}
