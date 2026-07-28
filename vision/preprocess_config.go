package vision

import (
	"fmt"
	"image"

	"golang.org/x/image/draw"
)

// ─────────────────────────────────────────────────────
// 预处理参数类型定义
// ─────────────────────────────────────────────────────

// InterpolationType 定义图像缩放插值算法。
type InterpolationType int

const (
	InterpolationBilinear        InterpolationType = iota // 双线性插值（默认）
	InterpolationNearestNeighbor                          // 最近邻插值
	InterpolationArea                                     // INTER_AREA（降采样时平均像素，使用 BiLinear 近似）
	InterpolationBicubic                                  // 双三次插值（Catmull-Rom）
	InterpolationLanczos4                                 // Lanczos4（使用 Catmull-Rom 近似）
)

// NormalizeMethod 定义图像归一化方法。
type NormalizeMethod int

const (
	NormalizeDivideBy255 NormalizeMethod = iota // [0,1] 缩放，pixel/255.0（默认，YOLO 系列使用）
	NormalizeImageNet                           // Z-score 归一化：(pixel/255 - mean) / std，使用 ImageNet 统计
	NormalizeCustom                             // 自定义 Z-score 归一化，使用 PreprocessConfig 中的 Mean/Std
)

// LetterBoxAlign 定义 LetterBox 填充对齐方式。
type LetterBoxAlign int

const (
	LetterBoxCenter  LetterBoxAlign = iota // 居中填充（默认），缩放后图像居中，周围填充 LetterBoxColor
	LetterBoxTopLeft                       // 左上角对齐，缩放后图像位于左上角，右侧和底部填充
)

// PreprocessConfig 封装图像预处理的所有可配置参数。
// 各字段均有合理的默认值，零值即可使用。
type PreprocessConfig struct {
	Interpolation   InterpolationType // 缩放插值算法，默认 InterpolationBilinear
	LetterBoxColor  [3]uint8          // LetterBox 填充颜色 [R, G, B]，默认 {114, 114, 114}
	LetterBoxAlign  LetterBoxAlign    // 填充对齐方式，默认 LetterBoxCenter
	NormalizeMethod NormalizeMethod   // 归一化方法，默认 NormalizeDivideBy255
	NormalizeMean   [3]float32        // 自定义归一化均值（仅 NormalizeCustom 时生效）
	NormalizeStd    [3]float32        // 自定义归一化标准差（仅 NormalizeCustom 时生效）
}

// DefaultPreprocessConfig 返回默认预处理配置。
//   - 插值: Bilinear
//   - LetterBox 颜色: [114, 114, 114]（YOLO 标准灰色）
//   - LetterBox 对齐: 居中
//   - 归一化: [0,1] 缩放
func DefaultPreprocessConfig() PreprocessConfig {
	return PreprocessConfig{
		Interpolation:   InterpolationBilinear,
		LetterBoxColor:  [3]uint8{114, 114, 114},
		LetterBoxAlign:  LetterBoxCenter,
		NormalizeMethod: NormalizeDivideBy255,
	}
}

// DefaultImageNetPreprocessConfig 返回 ImageNet 归一化的默认预处理配置。
// EdgeCrafter / RF-DETR / LTDETR / GroundingDINO 等 Transformer 模型使用此配置。
func DefaultImageNetPreprocessConfig() PreprocessConfig {
	return PreprocessConfig{
		Interpolation:   InterpolationBilinear,
		LetterBoxColor:  [3]uint8{114, 114, 114},
		LetterBoxAlign:  LetterBoxCenter,
		NormalizeMethod: NormalizeImageNet,
	}
}

// ─────────────────────────────────────────────────────
// 可配置插值的图像缩放
// ─────────────────────────────────────────────────────

// selectInterpolator 将 InterpolationType 映射为 golang.org/x/image/draw 的 Scaler。
func selectInterpolator(t InterpolationType) draw.Scaler {
	switch t {
	case InterpolationNearestNeighbor:
		return draw.NearestNeighbor
	case InterpolationBicubic, InterpolationLanczos4:
		// Catmull-Rom 是标准双三次插值的良好近似；
		// Lanczos4 没有标准 Go 实现，Catmull-Rom 作为高质量替代
		return draw.CatmullRom
	case InterpolationArea:
		// INTER_AREA 降采样时平均像素；draw.BiLinear 是 Go 中可用的最接近近似
		return draw.BiLinear
	default:
		return draw.BiLinear
	}
}

// Resize 将图像缩放到指定尺寸，使用可配置的插值算法。
// 返回 *image.RGBA 以兼容 FillCHWFromImage。
func Resize(img image.Image, width, height int, interp InterpolationType) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	scaler := selectInterpolator(interp)
	scaler.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Src, nil)
	return dst
}

// ─────────────────────────────────────────────────────
// LetterBox 辅助函数
// ─────────────────────────────────────────────────────

// LetterBoxParams 计算缩放和填充参数。
type LetterBoxParams struct {
	Scale float32 // 缩放因子
	NewW  int     // 缩放后的宽度
	NewH  int     // 缩放后的高度
	PadX  int     // 水平填充量（像素）
	PadY  int     // 垂直填充量（像素）
	OrigW int     // 原始宽度
	OrigH int     // 原始高度
}

// CalcLetterBox 根据输入尺寸和目标尺寸计算 LetterBox 参数。
//   - stride: 对齐步长（默认为 0 或 32）
//   - align: 填充对齐方式（居中对齐/左上角对齐）
func CalcLetterBox(origW, origH, targetSize int, stride int, align LetterBoxAlign) (LetterBoxParams, error) {
	if origW <= 0 || origH <= 0 || targetSize <= 0 {
		return LetterBoxParams{}, fmt.Errorf("invalid dimensions: origW=%d origH=%d targetSize=%d", origW, origH, targetSize)
	}

	if stride <= 0 {
		stride = 32
	}

	scale := float32(targetSize) / float32(max(origW, origH))
	newW := int(float32(origW) * scale)
	newH := int(float32(origH) * scale)

	// stride 对齐
	if newW%stride != 0 {
		newW = (newW/stride + 1) * stride
	}
	if newH%stride != 0 {
		newH = (newH/stride + 1) * stride
	}
	// stride 向上取整后可能使缩放尺寸超过 targetSize，导致 padX/padY 为负数，这里做钳制
	if newW > targetSize {
		newW = targetSize
	}
	if newH > targetSize {
		newH = targetSize
	}

	var padX, padY int
	if align == LetterBoxCenter {
		padX = (targetSize - newW) / 2
		padY = (targetSize - newH) / 2
	} else {
		padX = 0
		padY = 0
	}

	return LetterBoxParams{
		Scale: scale,
		NewW:  newW,
		NewH:  newH,
		PadX:  padX,
		PadY:  padY,
		OrigW: origW,
		OrigH: origH,
	}, nil
}

// defaultStride 返回默认 stride=32（YOLO 模型标准）。
const DefaultStride = 32
