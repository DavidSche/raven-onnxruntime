package anomaly

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func preprocessingConfig(normalize string) Config {
	return Config{
		InputSize: image.Point{X: 2, Y: 2},
		PreprocessContract: PreprocessContractConfig{
			Normalize:           normalize,
			ResizeKind:          "stretch",
			TensorLayout:        "NCHW",
			ColorSpace:          "RGB",
			ScaleToUnitInterval: true,
		},
	}
}

func TestBuildPreprocessTensorDataDivide255(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.SetRGBA(0, 0, color.RGBA{R: 0, G: 64, B: 255, A: 255})
	img.SetRGBA(1, 0, color.RGBA{R: 32, G: 96, B: 192, A: 255})
	img.SetRGBA(0, 1, color.RGBA{R: 64, G: 128, B: 128, A: 255})
	img.SetRGBA(1, 1, color.RGBA{R: 96, G: 160, B: 64, A: 255})

	data, params, err := buildPreprocessTensorData(img, preprocessingConfig("divide_255"))
	if err != nil {
		t.Fatalf("buildPreprocessTensorData() error = %v", err)
	}
	if params.originalWidth != 2 || params.originalHeight != 2 {
		t.Fatalf("original size = %v, want 2x2", params)
	}
	expected := []float32{
		0, 32.0 / 255, 64.0 / 255, 96.0 / 255,
		64.0 / 255, 96.0 / 255, 128.0 / 255, 160.0 / 255,
		255.0 / 255, 192.0 / 255, 128.0 / 255, 64.0 / 255,
	}
	for i, value := range expected {
		if data[i] != value {
			t.Fatalf("data[%d] = %v, want %v", i, data[i], value)
		}
	}
}

func TestBuildPreprocessTensorDataImageNet(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}
	data, _, err := buildPreprocessTensorData(img, preprocessingConfig("imagenet"))
	if err != nil {
		t.Fatalf("buildPreprocessTensorData() error = %v", err)
	}
	expectedRed := (128.0/255 - 0.485) / 0.229
	expectedGreen := (128.0/255 - 0.456) / 0.224
	expectedBlue := (128.0/255 - 0.406) / 0.225
	for _, value := range data[0:4] {
		if math.Abs(float64(value)-expectedRed) > 1e-6 {
			t.Fatalf("red value = %v, want %v", value, expectedRed)
		}
	}
	for _, value := range data[4:8] {
		if math.Abs(float64(value)-expectedGreen) > 1e-6 {
			t.Fatalf("green value = %v, want %v", value, expectedGreen)
		}
	}
	for _, value := range data[8:12] {
		if math.Abs(float64(value)-expectedBlue) > 1e-6 {
			t.Fatalf("blue value = %v, want %v", value, expectedBlue)
		}
	}
}

func TestBuildPreprocessTensorDataRejectsInvalidImageAndMode(t *testing.T) {
	if _, _, err := buildPreprocessTensorData(nil, preprocessingConfig("divide_255")); err == nil {
		t.Fatal("nil image accepted")
	}
	empty := image.NewRGBA(image.Rect(0, 0, 0, 0))
	if _, _, err := buildPreprocessTensorData(empty, preprocessingConfig("divide_255")); err == nil {
		t.Fatal("empty image accepted")
	}
	if _, _, err := buildPreprocessTensorData(image.NewRGBA(image.Rect(0, 0, 1, 1)), preprocessingConfig("invalid")); err == nil {
		t.Fatal("invalid normalization accepted")
	}
}

func TestStretchResizeResultKeepsOriginalDimensions(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 4; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x * 60), G: 0, B: uint8(y * 200), A: 255})
		}
	}
	_, params, err := buildPreprocessTensorData(img, preprocessingConfig("divide_255"))
	if err != nil {
		t.Fatal(err)
	}
	if params.originalWidth != 4 || params.originalHeight != 2 {
		t.Fatalf("original size = %v, want 4x2", params)
	}
}
