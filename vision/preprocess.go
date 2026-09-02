package vision

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
)

// RawRGBImage exposes raw interleaved RGB24 pixels for a fast preprocessing path.
type RawRGBImage interface {
	RawRGB() ([]byte, int)
}

// FillCHWFromImage fills data with CHW pixel values from an image.
// It returns an error if the provided buffers are too small for the given dimensions.
func FillCHWFromImage(data []float32, img image.Image, planeSize, targetW, width, height int, means, stds *[3]float32) error {
	switch src := img.(type) {
	case RawRGBImage:
		pix, stride := src.RawRGB()
		return FillCHWFromRGB24(data, pix, stride, planeSize, targetW, width, height, means, stds)
	case *image.RGBA:
		return FillCHWFromRGBA(data, src.Pix, src.Stride, planeSize, targetW, width, height, means, stds)
	case *image.NRGBA:
		return FillCHWFromRGBA(data, src.Pix, src.Stride, planeSize, targetW, width, height, means, stds)
	default:
		rgba := image.NewRGBA(image.Rect(0, 0, width, height))
		draw.Draw(rgba, rgba.Bounds(), img, img.Bounds().Min, draw.Src)
		return FillCHWFromRGBA(data, rgba.Pix, rgba.Stride, planeSize, targetW, width, height, means, stds)
	}
}

// FillCHWFromRGB24 fills data with RGB24 pixels in CHW order.
// It returns an error if the provided buffers are too small for the given dimensions.
func FillCHWFromRGB24(data []float32, pix []byte, stride, planeSize, targetW, width, height int, means, stds *[3]float32) error {
	return fillCHWFromPixels(data, pix, stride, 3, planeSize, targetW, width, height, means, stds)
}

// FillCHWFromRGBA fills data with RGBA pixels in CHW order.
// It returns an error if the provided buffers are too small for the given dimensions.
func FillCHWFromRGBA(data []float32, pix []byte, stride, planeSize, targetW, width, height int, means, stds *[3]float32) error {
	return fillCHWFromPixels(data, pix, stride, 4, planeSize, targetW, width, height, means, stds)
}

func fillCHWFromPixels(data []float32, pix []byte, stride, channels, planeSize, targetW, width, height int, means, stds *[3]float32) error {
	if planeSize < 0 || targetW < 0 || width < 0 || height < 0 || stride < 0 || channels < 3 {
		return fmt.Errorf("invalid dimensions: planeSize=%d targetW=%d width=%d height=%d stride=%d channels=%d", planeSize, targetW, width, height, stride, channels)
	}
	if len(data) < 3*planeSize {
		return fmt.Errorf("data buffer too small: need %d, got %d", 3*planeSize, len(data))
	}
	if height > 0 && width > 0 {
		// last byte accessed is pix[(height-1)*stride + (width-1)*channels + 2]
		needPix := (height-1)*stride + (width-1)*channels + 3
		if len(pix) < needPix {
			return fmt.Errorf("pix buffer too small: need %d, got %d", needPix, len(pix))
		}
	}

	if means != nil && stds != nil {
		m0, m1, m2 := means[0], means[1], means[2]
		s0, s1, s2 := stds[0], stds[1], stds[2]
		for y := 0; y < height; y++ {
			rowBase := y * stride
			dstBase := y * targetW
			for x := 0; x < width; x++ {
				srcIdx := rowBase + x*channels
				dstIdx := dstBase + x
				data[dstIdx] = (float32(pix[srcIdx])/255.0 - m0) / s0
				data[planeSize+dstIdx] = (float32(pix[srcIdx+1])/255.0 - m1) / s1
				data[2*planeSize+dstIdx] = (float32(pix[srcIdx+2])/255.0 - m2) / s2
			}
		}
		return nil
	}

	for y := 0; y < height; y++ {
		rowBase := y * stride
		dstBase := y * targetW
		for x := 0; x < width; x++ {
			srcIdx := rowBase + x*channels
			dstIdx := dstBase + x
			data[dstIdx] = float32(pix[srcIdx]) / 255.0
			data[planeSize+dstIdx] = float32(pix[srcIdx+1]) / 255.0
			data[2*planeSize+dstIdx] = float32(pix[srcIdx+2]) / 255.0
		}
	}
	return nil
}

// ResizeTorchBilinear resizes img to width×height using PyTorch's bilinear
// sampling with align_corners=False — the exact algorithm behind
// ``torchvision.transforms.functional.resize(antialias=False)`` that
// rf-detr's predict() pipeline uses. Plain bilinear kernels (PIL,
// x/image/draw.BiLinear) map output pixels to source coordinates with a
// different convention, which after a large downscale can shift normalized
// pixels by ~0.1+ (normalized space) and flip borderline detection
// confidence across a threshold — Transformer engines (rfdetr / ltdetr /
// edgecrafter) should therefore use this instead of Resize().
//
// Sampling formula (ATen upsample_bilinear2d, align_corners=False):
//
//	scale = in / out
//	src   = clamp((dst + 0.5) * scale - 0.5, min=0)
//	i0    = floor(src); frac = src - i0; i1 = min(i0+1, in-1)
//	val   = (1-frac_h)*((1-frac_w)*p[i0][j0] + frac_w*p[i0][j1])
//	      + frac_h   *((1-frac_w)*p[i1][j0] + frac_w*p[i1][j1])
//
// Interpolation is computed in float32 (ATen's accscalar_t<float>), and the
// result is rounded back to uint8 so callers keep the RGBA pipeline.
func ResizeTorchBilinear(img image.Image, width, height int) *image.RGBA {
	srcW, srcH := img.Bounds().Dx(), img.Bounds().Dy()

	// Normalize the source into an *image.RGBA with a (0,0) origin.
	src := image.NewRGBA(image.Rect(0, 0, srcW, srcH))
	draw.Draw(src, src.Bounds(), img, img.Bounds().Min, draw.Src)

	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	if srcW == 0 || srcH == 0 || width == 0 || height == 0 {
		return dst
	}

	rh := float32(srcH) / float32(height)
	rw := float32(srcW) / float32(width)

	type coord struct {
		i0, i1 int32
		frac   float32
	}
	rows := make([]coord, height)
	for oh := 0; oh < height; oh++ {
		r := (float32(oh)+0.5)*rh - 0.5
		if r < 0 {
			r = 0
		}
		i0 := int32(r)
		i1 := i0 + 1
		if i1 > int32(srcH)-1 {
			i1 = int32(srcH) - 1
		}
		rows[oh] = coord{i0: i0, i1: i1, frac: r - float32(i0)}
	}
	cols := make([]coord, width)
	for ow := 0; ow < width; ow++ {
		c := (float32(ow)+0.5)*rw - 0.5
		if c < 0 {
			c = 0
		}
		i0 := int32(c)
		i1 := i0 + 1
		if i1 > int32(srcW)-1 {
			i1 = int32(srcW) - 1
		}
		cols[ow] = coord{i0: i0, i1: i1, frac: c - float32(i0)}
	}

	for oh := 0; oh < height; oh++ {
		r := rows[oh]
		h0l := 1 - r.frac
		rowA := src.Pix[int(r.i0)*src.Stride:]
		rowB := src.Pix[int(r.i1)*src.Stride:]
		dstRow := dst.Pix[oh*dst.Stride:]
		for ow := 0; ow < width; ow++ {
			c := cols[ow]
			w0l := 1 - c.frac
			offA := int(c.i0) * 4
			offB := int(c.i1) * 4
			for ch := 0; ch < 3; ch++ {
				// Interpolate in [0,1] float32 exactly like ATen (input pixels are
				// divided by 255 before upsampling), then scale back for uint8.
				v11 := float32(rowA[offA+ch]) / 255.0
				v12 := float32(rowA[offB+ch]) / 255.0
				v21 := float32(rowB[offA+ch]) / 255.0
				v22 := float32(rowB[offB+ch]) / 255.0
				top := w0l*v11 + c.frac*v12
				bot := w0l*v21 + c.frac*v22
				val := h0l*top + r.frac*bot
				dstRow[ow*4+ch] = uint8(val*255 + 0.5)
			}
			dstRow[ow*4+3] = 0xff
		}
	}
	return dst
}

// ─────────────────────────────────────────────────────
// Letterbox 预处理
// ─────────────────────────────────────────────────────

// GetNormalizeParams 将 PreprocessConfig 的归一化方法转换为 means/stds 指针。
func GetNormalizeParams(cfg PreprocessConfig) (*[3]float32, *[3]float32) {
	switch cfg.NormalizeMethod {
	case NormalizeImageNet:
		return &[3]float32{0.485, 0.456, 0.406}, &[3]float32{0.229, 0.224, 0.225}
	case NormalizeCustom:
		return &cfg.NormalizeMean, &cfg.NormalizeStd
	default:
		return nil, nil
	}
}

// LetterboxImage 将图像缩放并填充到目标尺寸，返回 *image.RGBA。
// emptyBG 参数控制是否先用填充颜色初始化背景（默认为 true）。
func LetterboxImage(img image.Image, targetSize int, cfg PreprocessConfig) (*image.RGBA, error) {
	params, err := CalcLetterBox(img.Bounds().Dx(), img.Bounds().Dy(), targetSize, DefaultStride, cfg.LetterBoxAlign)
	if err != nil {
		return nil, err
	}

	// 1. 创建目标大小的背景图像，填充指定的 LetterBox 颜色
	bg := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))
	fillColor := color.RGBA{
		R: cfg.LetterBoxColor[0],
		G: cfg.LetterBoxColor[1],
		B: cfg.LetterBoxColor[2],
		A: 255,
	}
	draw.Draw(bg, bg.Bounds(), &image.Uniform{fillColor}, image.Point{}, draw.Src)

	// 2. 使用可配置插值缩放原始图像
	resized := Resize(img, params.NewW, params.NewH, cfg.Interpolation)

	// 3. 将缩放后的图像绘制到背景的居中/左上位置
	dstRect := image.Rect(params.PadX, params.PadY, params.PadX+params.NewW, params.PadY+params.NewH)
	draw.Draw(bg, dstRect, resized, resized.Bounds().Min, draw.Src)

	return bg, nil
}

// FillCHWWithLetterbox 执行完整的 LetterBox 预处理流程：
//  1. 计算缩放和填充参数
//  2. 使用可配置插值算法缩放图像
//  3. 按照填充颜色和居中方式填充到目标尺寸
//  4. 按照归一化方法转换为 CHW float32 格式
func FillCHWWithLetterbox(data []float32, img image.Image, cfg PreprocessConfig, inputSize int) (LetterBoxParams, error) {
	params, err := CalcLetterBox(img.Bounds().Dx(), img.Bounds().Dy(), inputSize, DefaultStride, cfg.LetterBoxAlign)
	if err != nil {
		return LetterBoxParams{}, err
	}
	means, stds := GetNormalizeParams(cfg)

	// 创建 LetterBox 图像并填充
	letterboxed, err := LetterboxImage(img, inputSize, cfg)
	if err != nil {
		return LetterBoxParams{}, err
	}

	// 转换为 CHW 格式（现在图像已经是 inputSize × inputSize）
	if err := FillCHWFromRGBA(data, letterboxed.Pix, letterboxed.Stride,
		inputSize*inputSize, inputSize, inputSize, inputSize, means, stds); err != nil {
		return LetterBoxParams{}, err
	}

	return params, nil
}
