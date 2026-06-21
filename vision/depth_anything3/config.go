package depth_anything3

import (
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
)

// ImageNet normalization constants
const (
	MeanR = 0.485
	MeanG = 0.456
	MeanB = 0.406

	StdR = 0.229
	StdG = 0.224
	StdB = 0.225

	// PatchSize DA3 ViT patch size
	PatchSize = 14
)

// ModelVariant DA3 model variant
type ModelVariant string

const (
	VariantSmall ModelVariant = "da3-small"
	VariantBase  ModelVariant = "da3-base"
	VariantLarge ModelVariant = "da3-large"
	VariantGiant ModelVariant = "da3-giant"
)

// Config holds configuration options for the Depth-Anything-3 engine.
type Config struct {
	ModelPath          string // ONNX model path
	OnnxRuntimeLibPath string // ONNX Runtime shared library path

	// Model parameters
	InputHeight int // Input height (must be a multiple of 14, default 518)
	InputWidth  int // Input width (must be a multiple of 14, default 518)

	// Optional parameters
	UseCuda           bool           // (optional) enable CUDA
	NumThreads        int            // (optional) ONNX thread count, default determined by CPU cores
	EnableCpuMemArena bool           // (optional) enable ONNX memory pool
	ApiVersion        ort.ApiVersion // (optional) ONNX Runtime C API version, default ort.DefaultApiVersion
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		InputHeight:        518,
		InputWidth:         518,
	}
}

// DepthResult holds the depth estimation result
type DepthResult struct {
	// Depth depth map (H, W), larger values indicate closer distance
	Depth []float32
	// Confidence depth confidence map (H, W)
	Confidence []float32
	// Width depth map width
	Width int
	// Height depth map height
	Height int
}

// ToGrayImage converts the depth map to a grayscale image (normalized to 0-255)
func (r *DepthResult) ToGrayImage() *image.Gray {
	img := image.NewGray(image.Rect(0, 0, r.Width, r.Height))

	minVal, maxVal := float32(1e10), float32(-1e10)
	for _, v := range r.Depth {
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

	for i, v := range r.Depth {
		normalized := (v - minVal) / rangeVal
		img.Pix[i] = uint8(normalized * 255.0)
	}

	return img
}

// ToConfGrayImage converts the confidence map to a grayscale image (normalized to 0-255)
func (r *DepthResult) ToConfGrayImage() *image.Gray {
	img := image.NewGray(image.Rect(0, 0, r.Width, r.Height))

	minVal, maxVal := float32(1e10), float32(-1e10)
	for _, v := range r.Confidence {
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

	for i, v := range r.Confidence {
		normalized := (v - minVal) / rangeVal
		img.Pix[i] = uint8(normalized * 255.0)
	}

	return img
}

// imageParams holds image dimension information for preprocessing
type imageParams struct {
	origW, origH int
	newW, newH   int
}
