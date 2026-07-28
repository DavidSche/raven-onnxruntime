package vision

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestFillCHWFromImageRGBA(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 0, A: 255})

	data := make([]float32, 3)
	if err := FillCHWFromImage(data, img, 1, 1, 1, 1, nil, nil); err != nil {
		t.Fatalf("FillCHWFromImage failed: %v", err)
	}

	want := []float32{1, 128.0 / 255.0, 0}
	for i, got := range data {
		if diff := math.Abs(float64(got - want[i])); diff > 1e-6 {
			t.Fatalf("channel %d: got %f want %f", i, got, want[i])
		}
	}
}

func TestFillCHWFromImageNormalized(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})

	data := make([]float32, 3)
	means := [3]float32{0.5, 0.0, 0.0}
	stds := [3]float32{0.5, 1.0, 1.0}
	if err := FillCHWFromImage(data, img, 1, 1, 1, 1, &means, &stds); err != nil {
		t.Fatalf("FillCHWFromImage failed: %v", err)
	}

	want := []float32{1, 0, 0}
	for i, got := range data {
		if diff := math.Abs(float64(got - want[i])); diff > 1e-6 {
			t.Fatalf("channel %d: got %f want %f", i, got, want[i])
		}
	}
}
