package rfdetr

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"strconv"
	"strings"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

var colorGray255 = color.Gray{Y: 255}

var imagenetMeans = [3]float32{0.485, 0.456, 0.406}
var imagenetStds = [3]float32{0.229, 0.224, 0.225}

var rfdetrModelResolutions = map[string]int{
	"rf-detr-nano":        384,
	"rf-detr-small":       512,
	"rf-detr-base":        640,
	"rf-detr-base-coco":   640,
	"rf-detr-base-o365":   640,
	"rf-detr-medium":      576,
	"rf-detr-large":       704,
	"rf-detr-large-2026":  704,
	"rf-detr-xlarge":      700,
	"rf-detr-xxlarge":     880,
	"rf-detr-seg-nano":    312,
	"rf-detr-seg-small":   384,
	"rf-detr-seg-medium":  432,
	"rf-detr-seg-large":   504,
	"rf-detr-seg-xlarge":  624,
	"rf-detr-seg-xxlarge": 768,
	"rf-detr-keypoint":    576,
	"rfdetr-keypoint":     576,
}

func detectInputSize(modelPath string) int {
	size, _ := detectInputSizeAndDynamicBatch(modelPath)
	if size <= 0 {
		size = 640
	}
	return size
}

func detectInputSizeAndDynamicBatch(modelPath string) (int, bool) {
	for name, size := range rfdetrModelResolutions {
		if strings.Contains(strings.ToLower(modelPath), name) {
			return size, false
		}
	}
	if size, dynamic, err := parseOnnxInputSizeAndDynamicBatch(modelPath); err == nil {
		if size > 0 {
			return size, dynamic
		}
		// Dynamic-shape exports (symbolic H/W dims, e.g. the raven-rfdetr
		// exporter's dynamic_axes={0:batch,2:height,3:width}) cannot report a
		// concrete input size from the graph input dims alone. Fall back to the
		// `resolution` metadata prop written by the raven-rfdetr exporter; the
		// parsed dynamicBatch flag is kept (batch may still be dynamic).
		if metaSize, ok := parseOnnxMetadataResolution(modelPath); ok && metaSize > 0 {
			return metaSize, dynamic
		}
	}
	return 640, false
}

// parseOnnxMetadataResolution reads the top-level ModelProto ``metadata_props``
// (field 14, repeated StringStringEntryProto: key=1, value=2) and returns the
// integer ``resolution`` prop when present. Used as a fallback for models whose
// graph input dims are symbolic (dynamic H/W) so the engine still feeds the
// resolution the graph was exported at.
func parseOnnxMetadataResolution(modelPath string) (int, bool) {
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return 0, false
	}
	s := newProtobufScanner(data)
	for s.next() {
		if s.fieldNum == 14 && s.wireType == 2 {
			key, val, ok := parseStringStringEntry(s.bytes())
			if ok && key == "resolution" {
				n, err := strconv.Atoi(strings.TrimSpace(val))
				if err == nil && n > 0 {
					return n, true
				}
				return 0, false
			}
		}
	}
	return 0, false
}

func parseStringStringEntry(data []byte) (string, string, bool) {
	s := newProtobufScanner(data)
	var key, val string
	var hasKey, hasVal bool
	for s.next() {
		if s.fieldNum == 1 && s.wireType == 2 {
			key = string(s.bytes())
			hasKey = true
		} else if s.fieldNum == 2 && s.wireType == 2 {
			val = string(s.bytes())
			hasVal = true
		}
	}
	return key, val, hasKey && hasVal
}

func parseOnnxInputSize(modelPath string) (int, error) {
	size, _, err := parseOnnxInputSizeAndDynamicBatch(modelPath)
	return size, err
}

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
	return 0, inputInfo.dynamicBatch, nil
}

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

func (s *protobufScanner) varint() uint64 {
	return s.value
}

func (s *protobufScanner) bytes() []byte {
	return s.rawBytes
}

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

type imageParams struct {
	origW, origH int
	tpadX, padY  int
}

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
	// rf-detr predict() resizes with torchvision F.resize(antialias=False), i.e.
	// bilinear align_corners=False — use the matching kernel so normalized
	// pixels (and borderline confidence scores) stay consistent with PyTorch.
	resized := vision.ResizeTorchBilinear(img, inputSize, inputSize)

	data := make([]float32, 3*inputSize*inputSize)
	if err := vision.FillCHWFromImage(data, resized, inputSize*inputSize, inputSize, inputSize, inputSize, means, stds); err != nil {
		return nil, imageParams{}, err
	}

	tensor, err := session.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, params, err
}

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
		// See preprocess(): match torchvision F.resize(antialias=False).
		resized := vision.ResizeTorchBilinear(img, inputSize, inputSize)
		base := i * sampleSize
		if err := vision.FillCHWFromImage(data[base:base+sampleSize], resized, planeSize, inputSize, inputSize, inputSize, means, stds); err != nil {
			return nil, nil, err
		}
	}

	tensor, err := session.NewTensor([]int64{int64(batchSize), 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, paramsList, err
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func boxCxcywhToXyxy(cx, cy, w, h float32, origW, origH int) image.Rectangle {
	x1 := cx - w/2
	y1 := cy - h/2
	x2 := cx + w/2
	y2 := cy + h/2

	origX1 := max(0, int(x1*float32(origW)))
	origY1 := max(0, int(y1*float32(origH)))
	origX2 := min(origW, int(x2*float32(origW)))
	origY2 := min(origH, int(y2*float32(origH)))

	return image.Rect(origX1, origY1, origX2, origY2)
}

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
				if sigmoid(val) > maskThreshold {
					finalMask.SetGray(x, y, colorGray255)
				}
			}
		}
	}

	return finalMask
}

type rawRGBImage interface {
	RawRGB() ([]byte, int)
}

func fillCHWFromImage(data []float32, img image.Image, inputSize int) error {
	return vision.FillCHWFromImage(data, img, inputSize*inputSize, inputSize, inputSize, inputSize, &imagenetMeans, &imagenetStds)
}

func fillCHWFromRGBA(data []float32, pix []byte, stride, inputSize int) error {
	return vision.FillCHWFromRGBA(data, pix, stride, inputSize*inputSize, inputSize, inputSize, inputSize, &imagenetMeans, &imagenetStds)
}
