package yolo26

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

// preprocess performs image preprocessing
func preprocess(img image.Image, inputSize int, session *ort.Session, ppCfg vision.PreprocessConfig) (*ort.Value, imageParams, error) {
	data, params, err := preprocessToCHW(img, inputSize, ppCfg)
	if err != nil {
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
	sampleSize := 3 * inputSize * inputSize
	data := make([]float32, batchSize*sampleSize)
	paramsList := make([]imageParams, batchSize)

	for i, img := range imgs {
		params, err := fillCHW(data[i*sampleSize:(i+1)*sampleSize], img, inputSize, ppCfg)
		if err != nil {
			return nil, nil, err
		}
		paramsList[i] = params
	}

	tensor, err := session.NewTensor([]int64{int64(batchSize), 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, paramsList, err
}

func preprocessToCHW(img image.Image, inputSize int, ppCfg vision.PreprocessConfig) ([]float32, imageParams, error) {
	data := make([]float32, 3*inputSize*inputSize)
	params, err := fillCHW(data, img, inputSize, ppCfg)
	return data, params, err
}

func fillCHW(data []float32, img image.Image, inputSize int, ppCfg vision.PreprocessConfig) (imageParams, error) {
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	// Fast path: when the image already matches the target input size, read
	// pixels directly instead of going through the LetterBox pipeline. This
	// avoids an unnecessary resize, which matters because interpolators sample
	// at sub-pixel positions and can blend adjacent pixels, corrupting exact
	// pixel values (e.g. the unit-test 2x2 image, or any 1:1 case where a
	// 255 channel must normalize to exactly 1.0).
	if origW == inputSize && origH == inputSize {
		means, stds := vision.GetNormalizeParams(ppCfg)
		planeSize := inputSize * inputSize
		if err := vision.FillCHWFromImage(data, img, planeSize, inputSize, inputSize, inputSize, means, stds); err != nil {
			return imageParams{}, err
		}
		return imageParams{
			origW: origW,
			origH: origH,
			scale: 1.0,
			padX:  0,
			padY:  0,
		}, nil
	}

	// 使用 LetterBox 预处理（支持可配置插值、填充颜色、居中/靠边、归一化方法）
	lbParams, err := vision.FillCHWWithLetterbox(data, img, ppCfg, inputSize)
	if err != nil {
		return imageParams{}, err
	}

	params := imageParams{
		origW: origW,
		origH: origH,
		scale: lbParams.Scale,
		padX:  lbParams.PadX,
		padY:  lbParams.PadY,
	}

	return params, nil
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

type nmsCandidate interface {
	GetBox() [4]float32
	GetScore() float32
}

func nms[T nmsCandidate](cands []T, iouThresh float32) []int {
	sort.Slice(cands, func(i, j int) bool {
		return cands[i].GetScore() > cands[j].GetScore()
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
			boxI := cands[i].GetBox()
			boxJ := cands[j].GetBox()
			r1 := image.Rect(int(boxI[0]), int(boxI[1]), int(boxI[2]), int(boxI[3]))
			r2 := image.Rect(int(boxJ[0]), int(boxJ[1]), int(boxJ[2]), int(boxJ[3]))
			if computeIOU(r1, r2) > iouThresh {
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

// skeleton connection pairs
var skeleton = [][2]int{
	{15, 13}, {13, 11}, {16, 14}, {14, 12}, // legs
	{11, 12}, {5, 11}, {6, 12}, // torso
	{5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, // arms/shoulders
	{1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, // face
}

// DrawPoseResult draws the skeleton on the image
func DrawPoseResult(img image.Image, results []PoseResult) image.Image {
	dst := image.NewRGBA(img.Bounds())
	draw.Draw(dst, img.Bounds(), img, img.Bounds().Min, draw.Src)

	lineColor := color.RGBA{G: 255, A: 255}  // green skeleton
	pointColor := color.RGBA{R: 255, A: 255} // red keypoints

	for _, res := range results {
		kpts := res.KeyPoints

		// draw connection lines
		for _, pair := range skeleton {
			idxA, idxB := pair[0], pair[1]
			kpA := kpts[idxA]
			kpB := kpts[idxB]
			if kpA.Score > 0.5 && kpB.Score > 0.5 {
				imageutil.DrawThickLine(dst, image.Point{X: kpA.X, Y: kpA.Y}, image.Point{X: kpB.X, Y: kpB.Y}, 5, lineColor)
			}
		}

		// draw keypoints
		for _, kp := range kpts {
			if kp.Score > 0.5 {
				imageutil.DrawFilledCircle(dst, image.Point{X: kp.X, Y: kp.Y}, 10, pointColor)
			}
		}
	}
	return dst
}

// getRotatedCorners calculates the 4 corner points of a rotated rectangle
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
