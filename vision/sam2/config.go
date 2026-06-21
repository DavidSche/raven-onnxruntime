package sam2

import (
	"path/filepath"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision/manifest"
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
	// ModelPath 模型包目录路径 (包含 manifest.json 和 ONNX 子模型)
	ModelPath          string
	OnnxRuntimeLibPath string

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
		ModelPath:          modelpath.ModelPath("sam2"),
	}
}

// resolveSubModelPaths 从 manifest.json 解析子模型路径
func (c *Config) resolveSubModelPaths() (encoderPath, decoderPath string, err error) {
	mf, loadErr := manifest.Load(c.ModelPath)
	if loadErr == nil {
		// 从 manifest.json 解析
		encoderPath = mf.SubModelPath(c.ModelPath, "image_encoder")
		decoderPath = mf.SubModelPath(c.ModelPath, "mask_decoder")
	} else {
		// 回退：使用目录内默认文件名
		encoderPath = filepath.Join(c.ModelPath, "sam2_image_encoder.onnx")
		decoderPath = filepath.Join(c.ModelPath, "sam2_mask_decoder.onnx")
	}
	return
}
