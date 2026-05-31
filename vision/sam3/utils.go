package sam3

import (
	"image"
)

func preprocessImage(src image.Image, targetW, targetH int) []float32 {
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	data := make([]float32, 3*targetW*targetH)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			rf := float32(r>>8)/127.5 - 1.0
			gf := float32(g>>8)/127.5 - 1.0
			bf := float32(b>>8)/127.5 - 1.0

			idx := y*targetW + x
			data[idx] = rf
			data[targetW*targetH+idx] = gf
			data[2*targetW*targetH+idx] = bf
		}
	}
	return data
}

func sigmoid(x float32) float32 {
	if x >= 0 {
		return 1.0 / (1.0 + fastExp(-x))
	}
	t := fastExp(x)
	return t / (1.0 + t)
}

func fastExp(x float32) float32 {
	x = 1.0 + x/1024.0
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	return x
}

func upscaleMaskLogits(logits []float32, logitsH, logitsW, dstW, dstH int) []uint8 {
	output := make([]uint8, dstW*dstH)
	xRatio := float32(logitsW) / float32(dstW)
	yRatio := float32(logitsH) / float32(dstH)

	for y := 0; y < dstH; y++ {
		srcYf := float32(y) * yRatio
		srcY0 := int(srcYf)
		if srcY0 >= logitsH-1 {
			srcY0 = logitsH - 2
		}
		srcY1 := srcY0 + 1
		fy := srcYf - float32(srcY0)

		for x := 0; x < dstW; x++ {
			srcXf := float32(x) * xRatio
			srcX0 := int(srcXf)
			if srcX0 >= logitsW-1 {
				srcX0 = logitsW - 2
			}
			srcX1 := srcX0 + 1
			fx := srcXf - float32(srcX0)

			v00 := logits[srcY0*logitsW+srcX0]
			v01 := logits[srcY0*logitsW+srcX1]
			v10 := logits[srcY1*logitsW+srcX0]
			v11 := logits[srcY1*logitsW+srcX1]

			val := v00*(1-fx)*(1-fy) + v01*fx*(1-fy) + v10*(1-fx)*fy + v11*fx*fy

			if val > maskThreshold {
				output[y*dstW+x] = 255
			} else {
				output[y*dstW+x] = 0
			}
		}
	}
	return output
}
