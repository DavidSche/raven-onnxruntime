package ltdetr

import (
	"fmt"
	"image"
	"image/draw"
	"math"
	"os"
	"strings"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

var imagenetMeans = [3]float32{0.485, 0.456, 0.406}
var imagenetStds = [3]float32{0.229, 0.224, 0.225}

var ltdetrModelResolutions = map[string]int{
	"vitt16-ltdetr":     640,
	"vitt16plus-ltdetr": 640,
	"vits16-ltdetr":     640,
	"vitb16-ltdetr":     640,
	"vitl16-ltdetr":     640,
	"tiny-ltdetr":       640,
	"ltdetr-coco":       640,
	"ltdetr":            640,
}

func detectInputSize(modelPath string) int {
	size, _ := detectInputSizeAndDynamicBatch(modelPath)
	if size <= 0 {
		size = 640
	}
	return size
}

func detectInputSizeAndDynamicBatch(modelPath string) (int, bool) {
	lower := strings.ToLower(modelPath)
	for name, size := range ltdetrModelResolutions {
		if strings.Contains(lower, name) {
			return size, false
		}
	}
	if size, dynamic, err := parseOnnxInputSizeAndDynamicBatch(modelPath); err == nil && size > 0 {
		return size, dynamic
	}
	return 640, false
}

func parseOnnxInputSizeAndDynamicBatch(modelPath string) (int, bool, error) {
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return 0, false, err
	}

	inputInfo := findOnnxInputInfo(data)
	if len(inputInfo.dims) >= 3 {
		h := inputInfo.dims[2]
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
	padX, padY   int
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
	resized := vision.Resize(img, inputSize, inputSize, ppCfg.Interpolation)

	data := make([]float32, 3*inputSize*inputSize)
	planeSize := inputSize * inputSize
	if err := vision.FillCHWFromImage(data, resized, planeSize, inputSize, inputSize, inputSize, means, stds); err != nil {
		return nil, imageParams{}, fmt.Errorf("failed to fill CHW data: %w", err)
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

		meansb, stdsb := vision.GetNormalizeParams(ppCfg)
		resized := vision.Resize(img, inputSize, inputSize, ppCfg.Interpolation)

		base := i * sampleSize
		if err := vision.FillCHWFromImage(data[base:base+sampleSize], resized, planeSize, inputSize, inputSize, inputSize, meansb, stdsb); err != nil {
			return nil, nil, fmt.Errorf("failed to fill CHW data for image %d: %w", i, err)
		}
	}

	tensor, err := session.NewTensor([]int64{int64(batchSize), 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, paramsList, err
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// boxXyxyToOrigScale converts xyxy pixel coordinates from the model input size
// to the original image scale. LTDETR ONNX outputs boxes in xyxy absolute pixel
// coordinates relative to the input resolution (e.g. 640x640).
func boxXyxyToOrigScale(x1, y1, x2, y2 float32, inputSize int, origW, origH int) image.Rectangle {
	scaleX := float32(origW) / float32(inputSize)
	scaleY := float32(origH) / float32(inputSize)

	origX1 := max(0, int(x1*scaleX))
	origY1 := max(0, int(y1*scaleY))
	origX2 := min(origW, int(x2*scaleX))
	origY2 := min(origH, int(y2*scaleY))

	return image.Rect(origX1, origY1, origX2, origY2)
}

type rawRGBImage interface {
	RawRGB() ([]byte, int)
}

func fillCHWFromImage(data []float32, img image.Image, inputSize int) {
	planeSize := inputSize * inputSize

	switch src := img.(type) {
	case rawRGBImage:
		pix, stride := src.RawRGB()
		for y := 0; y < inputSize; y++ {
			rowBase := y * stride
			dstBase := y * inputSize
			for x := 0; x < inputSize; x++ {
				srcIdx := rowBase + x*3
				dstIdx := dstBase + x
				data[dstIdx] = (float32(pix[srcIdx])/255.0 - imagenetMeans[0]) / imagenetStds[0]
				data[planeSize+dstIdx] = (float32(pix[srcIdx+1])/255.0 - imagenetMeans[1]) / imagenetStds[1]
				data[2*planeSize+dstIdx] = (float32(pix[srcIdx+2])/255.0 - imagenetMeans[2]) / imagenetStds[2]
			}
		}
	case *image.RGBA:
		fillCHWFromRGBA(data, src.Pix, src.Stride, inputSize)
	case *image.NRGBA:
		rgba := image.NewRGBA(src.Bounds())
		draw.Draw(rgba, rgba.Bounds(), src, src.Bounds().Min, draw.Src)
		fillCHWFromRGBA(data, rgba.Pix, rgba.Stride, inputSize)
	default:
		rgba := image.NewRGBA(img.Bounds())
		draw.Draw(rgba, rgba.Bounds(), img, img.Bounds().Min, draw.Src)
		fillCHWFromRGBA(data, rgba.Pix, rgba.Stride, inputSize)
	}
}

func fillCHWFromRGBA(data []float32, pix []byte, stride, inputSize int) {
	planeSize := inputSize * inputSize
	for y := 0; y < inputSize; y++ {
		rowBase := y * stride
		dstBase := y * inputSize
		for x := 0; x < inputSize; x++ {
			srcIdx := rowBase + x*4
			dstIdx := dstBase + x
			data[dstIdx] = (float32(pix[srcIdx])/255.0 - imagenetMeans[0]) / imagenetStds[0]
			data[planeSize+dstIdx] = (float32(pix[srcIdx+1])/255.0 - imagenetMeans[1]) / imagenetStds[1]
			data[2*planeSize+dstIdx] = (float32(pix[srcIdx+2])/255.0 - imagenetMeans[2]) / imagenetStds[2]
		}
	}
}
