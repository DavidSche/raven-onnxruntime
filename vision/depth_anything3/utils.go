package depth_anything3

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

// DepthToColormap converts the depth map to a pseudo-color image (turbo-like colormap)
func DepthToColormap(result *DepthResult) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, result.Width, result.Height))

	minVal, maxVal := float32(1e10), float32(-1e10)
	for _, v := range result.Depth {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	rangeVal := maxVal - minVal
	if rangeVal < 1e-6 {
		rangeVal = 1.0
	}

	for i, v := range result.Depth {
		normalized := (v - minVal) / rangeVal
		r, g, b := turboColormap(normalized)
		img.Pix[i*4] = r
		img.Pix[i*4+1] = g
		img.Pix[i*4+2] = b
		img.Pix[i*4+3] = 255
	}

	return img
}

// DepthToHeatmap converts the depth map to a heatmap image
func DepthToHeatmap(result *DepthResult) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, result.Width, result.Height))

	minVal, maxVal := float32(1e10), float32(-1e10)
	for _, v := range result.Depth {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	rangeVal := maxVal - minVal
	if rangeVal < 1e-6 {
		rangeVal = 1.0
	}

	for i, v := range result.Depth {
		normalized := (v - minVal) / rangeVal
		var r, g, b uint8
		if normalized < 0.25 {
			t := normalized / 0.25
			b = 255
			g = uint8(t * 255)
		} else if normalized < 0.5 {
			t := (normalized - 0.25) / 0.25
			g = 255
			b = uint8((1 - t) * 255)
		} else if normalized < 0.75 {
			t := (normalized - 0.5) / 0.25
			r = uint8(t * 255)
			g = 255
		} else {
			t := (normalized - 0.75) / 0.25
			r = 255
			g = uint8((1 - t) * 255)
		}
		img.Pix[i*4] = r
		img.Pix[i*4+1] = g
		img.Pix[i*4+2] = b
		img.Pix[i*4+3] = 255
	}

	return img
}

// turboColormap approximates the Turbo colormap
func turboColormap(t float32) (uint8, uint8, uint8) {
	t = clamp01(t)
	r := float32(0.5) + 0.5*float32(math.Sin(2.0*math.Pi*float64(t)+0.0))
	g := float32(0.5) + 0.5*float32(math.Sin(2.0*math.Pi*float64(t)-2.094))
	b := float32(0.5) + 0.5*float32(math.Sin(2.0*math.Pi*float64(t)-4.189))
	return uint8(r * 255), uint8(g * 255), uint8(b * 255)
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// clampColor8 clamps a float32 value to uint8 range
func clampColor8(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// DrawDepthOverlay blends the depth pseudo-color image onto the original image
func DrawDepthOverlay(img image.Image, result *DepthResult, alpha float32) image.Image {
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, img, bounds.Min, draw.Src)

	depthColor := DepthToColormap(result)

	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			idx := y*bounds.Dx() + x
			dr := depthColor.Pix[idx*4]
			dg := depthColor.Pix[idx*4+1]
			db := depthColor.Pix[idx*4+2]

			origR, origG, origB, _ := dst.RGBAAt(x, y).RGBA()
			oR := uint8(origR >> 8)
			oG := uint8(origG >> 8)
			oB := uint8(origB >> 8)

			blendR := uint8(float32(oR)*(1-alpha) + float32(dr)*alpha)
			blendG := uint8(float32(oG)*(1-alpha) + float32(dg)*alpha)
			blendB := uint8(float32(oB)*(1-alpha) + float32(db)*alpha)

			dst.SetRGBA(x, y, color.RGBA{R: blendR, G: blendG, B: blendB, A: 255})
		}
	}

	return dst
}

// DrawDepthResult overlays the depth map on the original image (red=near, blue=far)
func DrawDepthResult(img image.Image, result *DepthResult) image.Image {
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, img, bounds.Min, draw.Src)

	depthImg := result.ToGrayImage()
	if depthImg.Bounds().Dx() != bounds.Dx() || depthImg.Bounds().Dy() != bounds.Dy() {
		return dst
	}

	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			depthVal := depthImg.Pix[y*bounds.Dx()+x]
			if depthVal > 0 {
				r, g, b, _ := dst.At(x, y).RGBA()
				a := float32(depthVal) / 255.0 * 0.4
				dr := float32(uint8(r>>8)) * (1 - a)
				dg := float32(uint8(g>>8)) * (1 - a)
				db := float32(uint8(b>>8)) * (1 - a)

				if depthVal > 128 {
					dr += a * 255
				} else {
					db += a * 255
				}

				dst.SetRGBA(x, y, color.RGBA{R: clampColor8(dr), G: clampColor8(dg), B: clampColor8(db), A: 255})
			}
		}
	}

	return dst
}

// DrawDepthWithConfidence draws a depth map filtered by confidence threshold
func DrawDepthWithConfidence(result *DepthResult, confThreshold float32) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, result.Width, result.Height))

	minVal, maxVal := float32(1e10), float32(-1e10)
	for i, v := range result.Depth {
		if result.Confidence[i] >= confThreshold {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
	}

	rangeVal := maxVal - minVal
	if rangeVal < 1e-6 {
		rangeVal = 1.0
	}

	for i, v := range result.Depth {
		if result.Confidence[i] < confThreshold {
			img.Pix[i*4] = 0
			img.Pix[i*4+1] = 0
			img.Pix[i*4+2] = 0
			img.Pix[i*4+3] = 255
			continue
		}
		normalized := (v - minVal) / rangeVal
		r, g, b := turboColormap(normalized)
		img.Pix[i*4] = r
		img.Pix[i*4+1] = g
		img.Pix[i*4+2] = b
		img.Pix[i*4+3] = 255
	}

	return img
}

// DrawDepthLegend draws a depth color legend
func DrawDepthLegend(result *DepthResult, width, height int) image.Image {
	legend := image.NewRGBA(image.Rect(0, 0, width, height))

	minVal, maxVal := float32(1e10), float32(-1e10)
	for _, v := range result.Depth {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Draw color band
	for y := 0; y < height-20; y++ {
		t := 1.0 - float32(y)/float32(height-20)
		r, g, b := turboColormap(t)
		for x := 10; x < width-10; x++ {
			legend.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	_ = minVal
	_ = maxVal

	return legend
}

// ResizeWithBilinear resizes a depth map using bilinear interpolation
func ResizeWithBilinear(data []float32, srcW, srcH, dstW, dstH int) []float32 {
	output := make([]float32, dstW*dstH)
	xRatio := float64(srcW-1) / float64(dstW-1)
	yRatio := float64(srcH-1) / float64(dstH-1)

	for y := 0; y < dstH; y++ {
		srcY := float64(y) * yRatio
		y0 := int(math.Floor(srcY))
		y1 := min(y0+1, srcH-1)
		fy := float32(srcY - float64(y0))

		for x := 0; x < dstW; x++ {
			srcX := float64(x) * xRatio
			x0 := int(math.Floor(srcX))
			x1 := min(x0+1, srcW-1)
			fx := float32(srcX - float64(x0))

			v00 := data[y0*srcW+x0]
			v10 := data[y0*srcW+x1]
			v01 := data[y1*srcW+x0]
			v11 := data[y1*srcW+x1]

			output[y*dstW+x] = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy
		}
	}

	return output
}
