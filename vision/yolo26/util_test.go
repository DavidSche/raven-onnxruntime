package yolo26

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/vision"
)

type testCandidate struct {
	box   [4]float32
	score float32
}

func (c testCandidate) GetBox() [4]float32 { return c.box }
func (c testCandidate) GetScore() float32  { return c.score }

func TestComputeIOU(t *testing.T) {
	r1 := image.Rect(0, 0, 10, 10)
	r2 := image.Rect(5, 5, 15, 15)

	got := computeIOU(r1, r2)
	want := float32(25.0 / 175.0)
	if diff := math.Abs(float64(got - want)); diff > 1e-6 {
		t.Fatalf("iou mismatch: got %f want %f", got, want)
	}
}

func TestNMS(t *testing.T) {
	cands := []testCandidate{
		{score: 0.90, box: [4]float32{0, 0, 10, 10}},
		{score: 0.80, box: [4]float32{1, 1, 11, 11}},
		{score: 0.70, box: [4]float32{20, 20, 30, 30}},
	}

	keep := nms(cands, 0.5)
	if len(keep) != 2 {
		t.Fatalf("keep length mismatch: got %d want 2", len(keep))
	}
	if cands[keep[0]].score < cands[keep[1]].score {
		t.Fatalf("kept results not sorted by score: %+v", keep)
	}
}

func TestFillCHW(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})
	img.SetRGBA(1, 0, color.RGBA{R: 0, G: 255, B: 0, A: 255})
	img.SetRGBA(0, 1, color.RGBA{R: 0, G: 0, B: 255, A: 255})
	img.SetRGBA(1, 1, color.RGBA{R: 255, G: 255, B: 255, A: 255})

	data := make([]float32, 12)
	params, err := fillCHW(data, img, 2, vision.DefaultPreprocessConfig())
	if err != nil {
		t.Fatalf("fillCHW failed: %v", err)
	}
	if params.origW != 2 || params.origH != 2 || math.Abs(float64(params.scale-1.0)) > 1e-6 {
		t.Fatalf("unexpected params: %+v", params)
	}

	want := []float32{
		1, 0, 0, 1,
		0, 1, 0, 1,
		0, 0, 1, 1,
	}
	for i, got := range data {
		if diff := math.Abs(float64(got - want[i])); diff > 1e-6 {
			t.Fatalf("data[%d] got %f want %f", i, got, want[i])
		}
	}
}
