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

	// 自动适配模型真实输入尺寸：静态输入模型（如 yolo26x-obb 固定 1024×1024 输入）
	// 必须按模型输入形状预处理，否则会话运行会因张量形状不匹配而失败；动态输入模型
	// （yolo26n/s/m/l 官方导出 imgsz=1280）接受任意尺寸，沿用配置的 InputSize。
	if size := staticSquareInputSize(session); size > 0 && size != cfg.InputSize {
		cfg.InputSize = size
	}

	return &OBBEngine{
		session: session,
		config:  cfg,
	}, nil
}

// staticSquareInputSize 返回模型输入的静态方形尺寸（H==W 且为具体数值）。
// 动态输入（符号维度，ORT 返回 0）或形状不满足 4 维 / 非方形时返回 0，表示无需适配。
func staticSquareInputSize(session *ort.Session) int {
	if session == nil || len(session.InputNames) == 0 {
		return 0
	}
	dims, err := session.GetInputShape(0)
	if err != nil || len(dims) < 4 {
		return 0
	}
	h, w := dims[2], dims[3]
	if h > 0 && w > 0 && h == w {
		return int(h)
	}
	return 0
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

	// parse output [1, 300, 7]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}

	shape, err := outputValue.GetShape()
	if err != nil || len(shape) < 3 {
		return nil, fmt.Errorf("invalid output shape")
	}

	// decode [1, 300, 7] into OBB results
	return e.decodeOutput(data, shape, params), nil
}

// decodeOutput parses the model output [1, numBoxes, 7] into OBB results.
// Per-row layout: [cx, cy, w, h, score, class_id, angle].
// 独立方法以便单元测试直接驱动（无需 ONNX 会话/DLL）。
func (e *OBBEngine) decodeOutput(data []float32, shape []int64, params imageParams) []OBBResult {
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

		// map back to original image coordinates (letterbox pad removed, same as yolov11)
		origCorners := mapCornersToOrig(corners, params)

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

	return results
}

// mapCornersToOrig maps rotated-box corners from model coordinates back to the
// original image coordinates: subtract the letterbox padding (padX/padY) first,
// then divide by the scale, and clamp to the image bounds.
// Regression test target: before the fix the padding was not subtracted, so on
// non-square inputs corners shifted down by ~padY/scale pixels.
func mapCornersToOrig(corners [4][2]float32, params imageParams) [4]image.Point {
	var orig [4]image.Point
	for j, pt := range corners {
		// boundary check
		ox := min(max(0, int(math.Round(float64((pt[0]-float32(params.padX))/params.scale)))), params.origW)
		oy := min(max(0, int(math.Round(float64((pt[1]-float32(params.padY))/params.scale)))), params.origH)
		orig[j] = image.Point{X: ox, Y: oy}
	}
	return orig
}
