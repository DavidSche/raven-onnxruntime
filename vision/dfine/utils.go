package dfine

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"strings"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

// colorGray255 is a reusable white pixel for mask binarization.
var colorGray255 = color.Gray{Y: 255}

// Known D-FINE model resolutions keyed by substrings in the model path.
var dfineModelResolutions = map[string]int{
	"dfine_n":   640,
	"dfine_s":   640,
	"dfine_m":   640,
	"dfine_l":   640,
	"dfine_x":   640,
	"dfine_seg": 640,
	"dfine":     640,
}

// detectInputSize detects the model input size from the model path or ONNX metadata.
func detectInputSize(modelPath string) int {
	size, _ := detectInputSizeAndDynamicBatch(modelPath)
	if size <= 0 {
		size = 640
	}
	return size
}

// detectInputSizeAndDynamicBatch detects the input resolution and whether
// the model supports a dynamic batch dimension.
func detectInputSizeAndDynamicBatch(modelPath string) (int, bool) {
	lower := strings.ToLower(modelPath)
	for name, size := range dfineModelResolutions {
		if strings.Contains(lower, name) {
			return size, false
		}
	}
	if size, dynamic, err := parseOnnxInputSizeAndDynamicBatch(modelPath); err == nil && size > 0 {
		return size, dynamic
	}
	return 640, false
}

// parseOnnxInputSizeAndDynamicBatch reads the ONNX protobuf to extract the
// input tensor shape from the ModelProto → GraphProto → ValueInfoProto chain.
func parseOnnxInputSizeAndDynamicBatch(modelPath string) (int, bool, error) {
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return 0, false, err
	}

	inputInfo := findOnnxInputInfo(data)
	if len(inputInfo.dims) >= 4 {
		h := inputInfo.dims[2]
		w := inputInfo.dims[3]
		if h > 0 && h == w {
			return int(h), inputInfo.dynamicBatch, nil
		}
		if h > 0 {
			return int(h), inputInfo.dynamicBatch, nil
		}
	}
	if len(inputInfo.dims) >= 3 {
		h := inputInfo.dims[2]
		if h > 0 {
			return int(h), inputInfo.dynamicBatch, nil
		}
	}
	return 0, inputInfo.dynamicBatch, nil
}

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf scanner (field numbers copied from rfdetr/ltdetr).
// ---------------------------------------------------------------------------

type onnxInputInfo struct {
	name         string
	dims         []int64
	dtype        int32
	dynamicBatch bool
}

func findOnnxInputInfo(data []byte) onnxInputInfo {
	var info onnxInputInfo
	findModelGraph(data, &info)
	return info
}

func findModelGraph(data []byte, info *onnxInputInfo) {
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 7 && s.wireType == 2 {
			findGraphInputs(s.bytes(), info)
			return
		}
	}
}

func findGraphInputs(data []byte, info *onnxInputInfo) {
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 11 && s.wireType == 2 {
			if parseValueInfo(s.bytes(), info) {
				return
			}
		}
	}
}

func parseValueInfo(data []byte, info *onnxInputInfo) bool {
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 2 && s.wireType == 2 {
			return parseTypeProto(s.bytes(), info)
		}
	}
	return false
}

func parseTypeProto(data []byte, info *onnxInputInfo) bool {
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 1 && s.wireType == 2 {
			return parseTensorType(s.bytes(), info)
		}
	}
	return false
}

func parseTensorType(data []byte, info *onnxInputInfo) bool {
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 2 && s.wireType == 2 {
			parseTensorShape(s.bytes(), info)
			return len(info.dims) > 0
		}
	}
	return false
}

func parseTensorShape(data []byte, info *onnxInputInfo) {
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 1 && s.wireType == 2 {
			parseTensorShapeDim(s.bytes(), info)
		}
	}
}

func parseTensorShapeDim(data []byte, info *onnxInputInfo) {
	dimIdx := len(info.dims)
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 1 && s.wireType == 0 {
			info.dims = append(info.dims, int64(s.varint()))
		} else if s.fieldNum == 2 && s.wireType == 2 {
			info.dims = append(info.dims, -1)
			if dimIdx == 0 {
				info.dynamicBatch = true
			}
		}
	}
}

type protobufScanner struct {
	data     []byte
	pos      int
	fieldNum uint64
	wireType uint64
	value    uint64
	rawBytes []byte
}

func newProtobufScanner(data []byte) *protobufScanner {
	return &protobufScanner{data: data}
}

func (s *protobufScanner) next() bool {
	if s.pos >= len(s.data) {
		return false
	}
	tag, n := decodeVarint(s.data, s.pos)
	if n == 0 {
		return false
	}
	s.pos += n
	s.fieldNum = tag >> 3
	s.wireType = tag & 0x7

	switch s.wireType {
	case 0:
		val, n2 := decodeVarint(s.data, s.pos)
		if n2 == 0 {
			return false
		}
		s.value = val
		s.pos += n2
	case 2:
		msgLen, n2 := decodeVarint(s.data, s.pos)
		if n2 == 0 {
			return false
		}
		s.pos += n2
		end := s.pos + int(msgLen)
		if end > len(s.data) {
			return false
		}
		s.rawBytes = s.data[s.pos:end]
		s.pos = end
	case 1:
		s.pos += 8
	case 5:
		s.pos += 4
	default:
		return false
	}
	return true
}

func (s *protobufScanner) varint() uint64 { return s.value }
func (s *protobufScanner) bytes() []byte  { return s.rawBytes }

func decodeVarint(data []byte, pos int) (uint64, int) {
	var result uint64
	var shift uint
	for i := 0; pos+i < len(data) && i < 10; i++ {
		b := data[pos+i]
		result |= uint64(b&0x7F) << shift
		shift += 7
		if b&0x80 == 0 {
			return result, i + 1
		}
	}
	return 0, 0
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

// preprocess resizes and normalizes a single image for D-FINE models.
// Uses ImageNet normalization and simple resize (no letterbox).
func preprocess(img image.Image, inputSize int, session *ort.Session, ppCfg vision.PreprocessConfig) (*ort.Value, imageParams, error) {
	if img == nil {
		return nil, imageParams{}, fmt.Errorf("input image is nil")
	}
	bounds := img.Bounds()
	params := imageParams{
		origW: bounds.Dx(),
		origH: bounds.Dy(),
	}

	means, stds := vision.GetNormalizeParams(ppCfg)
	resized := vision.Resize(img, inputSize, inputSize, ppCfg.Interpolation)

	data := make([]float32, 3*inputSize*inputSize)
	planeSize := inputSize * inputSize
	if err := vision.FillCHWFromImage(data, resized, planeSize, inputSize, inputSize, inputSize, means, stds); err != nil {
		return nil, imageParams{}, fmt.Errorf("failed to fill CHW data: %w", err)
	}

	tensor, err := session.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, params, err
}

// preprocessBatch resizes and normalizes a batch of images.
func preprocessBatch(imgs []image.Image, inputSize int, session *ort.Session, ppCfg vision.PreprocessConfig) (*ort.Value, []imageParams, error) {
	if len(imgs) == 0 {
		return nil, nil, nil
	}

	batchSize := len(imgs)
	planeSize := inputSize * inputSize
	sampleSize := 3 * planeSize
	data := make([]float32, batchSize*sampleSize)
	paramsList := make([]imageParams, batchSize)

	for i, img := range imgs {
		if img == nil {
			return nil, nil, fmt.Errorf("input image at index %d is nil", i)
		}
		bounds := img.Bounds()
		paramsList[i] = imageParams{
			origW: bounds.Dx(),
			origH: bounds.Dy(),
		}

		means, stds := vision.GetNormalizeParams(ppCfg)
		resized := vision.Resize(img, inputSize, inputSize, ppCfg.Interpolation)

		base := i * sampleSize
		if err := vision.FillCHWFromImage(data[base:base+sampleSize], resized, planeSize, inputSize, inputSize, inputSize, means, stds); err != nil {
			return nil, nil, fmt.Errorf("failed to fill CHW data for image %d: %w", i, err)
		}
	}

	tensor, err := session.NewTensor([]int64{int64(batchSize), 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, paramsList, err
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// ---------------------------------------------------------------------------
// Box coordinate conversion
// ---------------------------------------------------------------------------

// boxXyxyToOrigScale converts xyxy pixel coordinates from the model input
// resolution to the original image dimensions. D-FINE and D-FINE-seg ONNX
// models output boxes in absolute pixel coordinates on the model input grid.
func boxXyxyToOrigScale(x1, y1, x2, y2 float32, inputSize int, origW, origH int) image.Rectangle {
	scaleX := float32(origW) / float32(inputSize)
	scaleY := float32(origH) / float32(inputSize)

	origX1 := max(0, int(x1*scaleX))
	origY1 := max(0, int(y1*scaleY))
	origX2 := min(origW, int(x2*scaleX))
	origY2 := min(origH, int(y2*scaleY))

	return image.Rect(origX1, origY1, origX2, origY2)
}

// ---------------------------------------------------------------------------
// Mask resizing
// ---------------------------------------------------------------------------

// resizeMask takes a per-query mask plane at low resolution (e.g. 160×160),
// extracts the region corresponding to the detection box, and produces a
// binarized full-resolution (origW×origH) mask.
func resizeMask(maskData []float32, maskH, maskW int, origBox image.Rectangle, origW, origH int, maskThreshold float32) *image.Gray {
	finalMask := image.NewGray(image.Rect(0, 0, origW, origH))

	scaleX := float32(maskW) / float32(origW)
	scaleY := float32(maskH) / float32(origH)

	for y := origBox.Min.Y; y < origBox.Max.Y; y++ {
		for x := origBox.Min.X; x < origBox.Max.X; x++ {
			mx := int(float32(x) * scaleX)
			my := int(float32(y) * scaleY)

			if mx >= 0 && mx < maskW && my >= 0 && my < maskH {
				val := maskData[my*maskW+mx]
				// The mask tensor from the model is already sigmoid-applied
				// (output range [0, 1]), no need for further sigmoid.
				if val > maskThreshold {
					finalMask.SetGray(x, y, colorGray255)
				}
			}
		}
	}

	return finalMask
}
