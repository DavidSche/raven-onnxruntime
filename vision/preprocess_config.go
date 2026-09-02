package vision

import (
	"fmt"
	"image"
	"math"

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

// pyRound 模拟 Python 内置 round()（银行家舍入：恰好 .5 时舍入到偶数），
// 用于与 ultralytics letterbox 的内容尺寸 round(w*r) 保持逐位一致。
// 仅处理非负输入（图像尺寸恒为正）。
func pyRound(x float64) int {
	f := math.Floor(x)
	frac := x - f
	if frac < 0.5 {
		return int(f)
	}
	if frac > 0.5 {
		return int(f) + 1
	}
	// 恰为 .5：舍入到偶数
	if int(f)%2 == 0 {
		return int(f)
	}
	return int(f) + 1
}

// CalcLetterBox 根据输入尺寸和目标尺寸计算 LetterBox 参数。
//   - stride: 对齐步长（默认为 0 或 32）
//   - align: 填充对齐方式（居中对齐/左上角对齐）
func CalcLetterBox(origW, origH, targetSize int, stride int, align LetterBoxAlign) (LetterBoxParams, error) {
	if origW <= 0 || origH <= 0 || targetSize <= 0 {
		return LetterBoxParams{}, fmt.Errorf("invalid dimensions: origW=%d origH=%d targetSize=%d", origW, origH, targetSize)
	}

	// stride 参数仅保留 API 兼容（调用方传入 DefaultStride）；内容尺寸已不做 stride 对齐
	//（见下方注释），该参数当前不使用。
	if stride <= 0 {
		stride = 32
	}

	// 内容尺寸比例在 float64 下计算（与 ultralytics letterbox 的
	// r = min(t/h, t/w) 逐位一致），避免 float32 量化误差把乘积翻过 .5 边界。
	// 内容尺寸使用 Python round()（银行家舍入）语义，与 ultralytics 的 round(w*r)
	// 一致：int() 截断会在 float 乘积略低于整数时少 1px（如 963×(1280/963)
	// =1279.999… 截断为 1279 而 round 为 1280）。回归测试：preprocess_test.go 的
	// TestCalcLetterBox_RoundSemantics。
	scale64 := float64(targetSize) / float64(max(origW, origH))
	newW := pyRound(float64(origW) * scale64)
	newH := pyRound(float64(origH) * scale64)
	scale := float32(scale64)

	// 注意：不再对内容尺寸做 stride(32) 对齐。早期实现会把缩放后的内容向上取整到
	// 32 的倍数（如 360→384），导致喂给模型的内容长宽比被拉伸（Y 向最多 +6.7%），
	// 与 ultralytics 标准 letterbox（内容=精确缩放尺寸，画布 padding 到 targetSize）
	// 不一致，输出坐标系统性偏移（非 32 对齐内容上可达数十像素）。
	// 回归护栏：raven-go/internal/inference/ 下的 yolo11/yolo26 OBB、det/seg/pose
	// 多宽高比门控测试（obb_multimage_gated_test.go、yolo11_multimage_gated_test.go）。
	// 钳制：缩放后尺寸理论上不会超过 targetSize（scale 取 min 边），保留钳制以防浮点边界。
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
