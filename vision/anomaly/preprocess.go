package anomaly

import (
	"fmt"
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

type preprocessResult struct {
	originalWidth  int
	originalHeight int
}

// preprocessImage performs the frozen anomaly-v1 out-of-graph preprocessing:
// torchvision-style stretch resize to the authoritative ONNX H/W, RGB byte to
// NCHW float32 conversion, and the model-specific normalization.
func preprocessImage(session *ort.Session, img image.Image, cfg Config) (*ort.Value, preprocessResult, error) {
	if img == nil {
		return nil, preprocessResult{}, fmt.Errorf("%w: image is nil", ErrInvalidConfig)
	}
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, preprocessResult{}, fmt.Errorf("%w: image has non-positive dimensions %dx%d", ErrInvalidConfig, width, height)
	}
	targetWidth, targetHeight := cfg.InputSize.X, cfg.InputSize.Y
	if targetWidth <= 0 || targetHeight <= 0 {
		return nil, preprocessResult{}, fmt.Errorf("%w: ONNX input size is not initialized", ErrInvalidConfig)
	}

	tensorData, result, err := buildPreprocessTensorData(img, cfg)
	if err != nil {
		return nil, preprocessResult{}, err
	}
	tensor, err := session.NewTensor([]int64{1, 3, int64(targetHeight), int64(targetWidth)}, tensorData)
	if err != nil {
		return nil, preprocessResult{}, fmt.Errorf("failed to create input tensor: %w", err)
	}
	return tensor, result, nil
}

func buildPreprocessTensorData(img image.Image, cfg Config) ([]float32, preprocessResult, error) {
	if img == nil {
		return nil, preprocessResult{}, fmt.Errorf("%w: image is nil", ErrInvalidConfig)
	}
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, preprocessResult{}, fmt.Errorf("%w: image has non-positive dimensions %dx%d", ErrInvalidConfig, width, height)
	}
	targetWidth, targetHeight := cfg.InputSize.X, cfg.InputSize.Y
	if targetWidth <= 0 || targetHeight <= 0 {
		return nil, preprocessResult{}, fmt.Errorf("%w: ONNX input size is not initialized", ErrInvalidConfig)
	}

	resized := vision.ResizeTorchBilinear(img, targetWidth, targetHeight)
	planeSize := targetWidth * targetHeight
	tensorData := make([]float32, 3*planeSize)
	var means, stds *[3]float32
	switch cfg.PreprocessContract.Normalize {
	case normalizeDivide255:
	case normalizeImageNet:
		means = &[3]float32{0.485, 0.456, 0.406}
		stds = &[3]float32{0.229, 0.224, 0.225}
	default:
		return nil, preprocessResult{}, fmt.Errorf("%w: unsupported normalize mode %q", ErrInvalidConfig, cfg.PreprocessContract.Normalize)
	}
	if err := vision.FillCHWFromRGBA(tensorData, resized.Pix, resized.Stride, planeSize, targetWidth, targetHeight, targetHeight, means, stds); err != nil {
		return nil, preprocessResult{}, fmt.Errorf("failed to fill NCHW tensor: %w", err)
	}
	return tensorData, preprocessResult{originalWidth: width, originalHeight: height}, nil
}
