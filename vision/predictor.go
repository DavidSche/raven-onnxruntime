package vision

import (
	"errors"
	"image"
)

// ─────────────────────────────────────────────────────
// 统一结果类型
// ─────────────────────────────────────────────────────

// KeyPoint 关键点（姿态估计）
type KeyPoint struct {
	X, Y  int
	Score float32
}

// Corner OBB 旋转框顶点
type Corner struct {
	X, Y int
}

// Detection 是 vision 层的标准化检测/分割/姿态估计结果。
//
// 设计原则：
//   - 使用 nil 区分"不适用"与"空列表"（如 Mask==nil 表示非分割任务）
//   - ClassName 由 vision 层填充（对开放词汇模型）或由 raven-go 填充（对 COCO 类标签）
//   - Predictor 只负责 Detection 家族（detect/seg/pose/obb/open-vocab）
//   - OCR/VLM/Depth 使用独立接口，不纳入此结构
type Detection struct {
	ClassID   int
	ClassName string // 开放词汇模型直接填充；COCO 标签由 raven-go 根据 ID 查表
	Score     float32
	Box       image.Rectangle
	Mask      *image.Gray // 实例分割 mask，nil if N/A
	KeyPoints []KeyPoint  // 姿态关键点，nil if N/A
	Corners   []Corner    // OBB 旋转框顶点，nil if N/A
}

// ─────────────────────────────────────────────────────
// 运行时提示（Prompt）机制
// ─────────────────────────────────────────────────────

// Prompt 封装推理时的可变参数，解决不同模型需要不同运行时输入的问题。
//
// 使用示例：
//
//	// 标准图像检测（无提示）
//	prompt := vision.NoPrompt()
//
//	// 开放词汇检测（文本提示）
//	prompt := vision.TextPrompt("person", "car", "bicycle")
type Prompt struct {
	Texts  []string          // 文本提示（GroundingDINO / Grounded-SAM2）
	Points []image.Point     // 交互点提示（SAM 系列，未来扩展）
	Boxes  []image.Rectangle // 框提示（SAM Box Mode，未来扩展）
}

// NoPrompt 返回空提示（标准模型使用）。
func NoPrompt() Prompt { return Prompt{} }

// TextPrompt 返回文本提示（开放词汇模型使用）。
func TextPrompt(texts ...string) Prompt { return Prompt{Texts: texts} }

// ─────────────────────────────────────────────────────
// 统一推理接口
// ─────────────────────────────────────────────────────

// Predictor 是 vision 层的统一推理接口，覆盖 Detection 家族：
//   - 目标检测（YOLO、RF-DETR、LTDETR、EdgeCrafter）
//   - 实例分割（YOLO-seg、RF-DETR-seg、EdgeCrafter-seg）
//   - 姿态估计（YOLO-pose、RF-DETR-kp、EdgeCrafter-pose）
//   - 旋转目标检测 OBB（YOLO-obb）
//   - 开放词汇检测（GroundingDINO）via TextPrompt
//   - 开放词汇分割（Grounded-SAM2）via TextPrompt
//
// 不在范围内（独立接口）：
//   - 深度估计（DepthPredictor）
//   - 交互式分割（SAM3，两阶段 EncodeImage+Decode）
//   - OCR / VLM（未来独立接口）
type Predictor interface {
	// Predict 对单帧执行推理。
	// prompt 为空（NoPrompt()）时等价于原始 Predict(img)。
	Predict(img image.Image, prompt Prompt) ([]Detection, error)

	// Destroy 释放所有资源（ONNX Session、内存等）。
	Destroy()
}

// BatchPredictor 是可选的批量推理接口。
// 支持动态批量的模型（如 RF-DETR DynamicBatch）实现此接口以启用批量推理。
// 不实现的模型由调用方自动降级为逐帧 Predict。
//
// 类型断言示例：
//
//	if bp, ok := p.(BatchPredictor); ok {
//	    results, err := bp.PredictBatch(imgs, prompt)
//	}
type BatchPredictor interface {
	// PredictBatch 批量推理。
	PredictBatch(imgs []image.Image, prompt Prompt) ([][]Detection, error)

	// MaxBatch 返回模型支持的最大批量大小。固定 batch=1 的模型返回 1。
	MaxBatch() int
}

// ErrBatchNotSupported 表示模型不支持批量推理，调用方应降级为逐帧。
var ErrBatchNotSupported = errors.New("batch inference not supported by this model")

// ─────────────────────────────────────────────────────
// 深度估计（独立接口，不纳入 Predictor）
// ─────────────────────────────────────────────────────

// DepthMap 深度估计结果。
type DepthMap struct {
	Depth      []float32 // H×W depth values
	Confidence []float32 // H×W confidence (optional)
	Width      int
	Height     int
	MinDepth   float32
	MaxDepth   float32
}

// DepthPredictor 深度估计推理接口（独立于 Predictor）。
type DepthPredictor interface {
	PredictDepth(img image.Image) (*DepthMap, error)
	Destroy()
}
