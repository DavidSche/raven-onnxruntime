package yolov11

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"sort"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/imageutil"
)

func preprocess(img image.Image, inputSize int, session *ort.Session, ppCfg vision.PreprocessConfig) (*ort.Value, imageParams, error) {
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	data := make([]float32, 3*inputSize*inputSize)
	lbParams, err := vision.FillCHWWithLetterbox(data, img, ppCfg, inputSize)
	if err != nil {
		return nil, imageParams{}, err
	}

	params := imageParams{
		origW: origW,
		origH: origH,
		scale: lbParams.Scale,
		padX:  lbParams.PadX,
		padY:  lbParams.PadY,
	}

	tensor, err := session.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, params, err
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func nms(cands []candidate, iouThresh float32) []int {
	sort.Slice(cands, func(i, j int) bool {
		return cands[i].score > cands[j].score
	})
	keep := make([]int, 0)
	suppressed := make([]bool, len(cands))
	for i := 0; i < len(cands); i++ {
		if suppressed[i] {
			continue
		}
		keep = append(keep, i)
		for j := i + 1; j < len(cands); j++ {
			if suppressed[j] {
				continue
			}
			if computeIOU(cands[i].origBox, cands[j].origBox) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func computeIOU(r1, r2 image.Rectangle) float32 {
	intersect := r1.Intersect(r2)
	if intersect.Empty() {
		return 0.0
	}
	interArea := intersect.Dx() * intersect.Dy()
	area1 := r1.Dx() * r1.Dy()
	area2 := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(area1+area2-interArea)
}

var skeleton = [][2]int{
	{15, 13}, {13, 11}, {16, 14}, {14, 12},
	{11, 12}, {5, 11}, {6, 12},
	{5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10},
	{1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4},
}

func DrawPoseResult(img image.Image, results []PoseResult) image.Image {
	dst := image.NewRGBA(img.Bounds())
	draw.Draw(dst, img.Bounds(), img, img.Bounds().Min, draw.Src)
	lineColor := color.RGBA{G: 255, A: 255}
	pointColor := color.RGBA{R: 255, A: 255}
	for _, res := range results {
		kpts := res.KeyPoints
		for _, pair := range skeleton {
			idxA, idxB := pair[0], pair[1]
			kpA := kpts[idxA]
			kpB := kpts[idxB]
			if kpA.Score > 0.5 && kpB.Score > 0.5 {
				imageutil.DrawThickLine(dst, image.Point{X: kpA.X, Y: kpA.Y}, image.Point{X: kpB.X, Y: kpB.Y}, 5, lineColor)
			}
		}
		for _, kp := range kpts {
			if kp.Score > 0.5 {
				imageutil.DrawFilledCircle(dst, image.Point{X: kp.X, Y: kp.Y}, 10, pointColor)
			}
		}
	}
	return dst
}

func getRotatedCorners(cx, cy, w, h, angle float32) [4][2]float32 {
	cosA := float32(math.Cos(float64(angle)))
	sinA := float32(math.Sin(float64(angle)))
	dx := []float32{-w / 2, w / 2, w / 2, -w / 2}
	dy := []float32{-h / 2, -h / 2, h / 2, h / 2}
	var corners [4][2]float32
	for i := 0; i < 4; i++ {
		rx := dx[i]*cosA - dy[i]*sinA
		ry := dx[i]*sinA + dy[i]*cosA
		corners[i][0] = cx + rx
		corners[i][1] = cy + ry
	}
	return corners
}
