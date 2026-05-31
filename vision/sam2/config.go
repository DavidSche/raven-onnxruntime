package sam2

import (
	ort "github.com/DavidSche/raven-onnxruntime/ort"
)

type Label int

const (
	LabelBackground  Label = 0 // background/exclude
	LabelForeground  Label = 1 // foreground/click
	LabelBoxTopLeft  Label = 2 // box top-left
	LabelBoxBotRight Label = 3 // box bottom-right
)

// mean and std constants
const (
	MeanG = 0.456
	MeanB = 0.406
	MeanR = 0.485

	StdG = 0.224
	StdB = 0.225
	StdR = 0.229
)

const (
	// inputSize is the long edge size of the input image
	inputSize = 1024
	// maskThreshold threshold
	maskThreshold = 0.0
)

type Point struct {
	X, Y  float32
	Label Label
}

// Config holds configuration options
type Config struct {
	// required parameters
	OnnxRuntimeLibPath string // path to onnxruntime.dll (or .so, .dylib)
	EncodeModelPath    string // image feature extraction model
	DecodeModelPath    string // Mask decoding model

	// optional parameters
	UseCuda           bool           // (optional) enable CUDA
	NumThreads        int            // (optional) ONNX thread count, default determined by CPU cores
	EnableCpuMemArena bool           // (optional) enable ONNX memory pool
	ApiVersion        ort.ApiVersion // (optional) ONNX Runtime C API version, default ort.DefaultApiVersion
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		EncodeModelPath:    "./sam2_weights/vision_encoder.onnx",
		DecodeModelPath:    "./sam2_weights/prompt_encoder_mask_decoder.onnx",
	}
}
