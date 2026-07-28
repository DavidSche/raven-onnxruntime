package ltdetr

import (
	"image"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

type Config struct {
	ModelPath          string
	OnnxRuntimeLibPath string

	ConfThreshold float32
	InputSize     int
	NumClasses    int
	MaxDetections int

	DynamicBatch      bool
	UseCuda           bool
	NumThreads        int
	EnableCpuMemArena bool
	ApiVersion        ort.ApiVersion
	PreprocessConfig  vision.PreprocessConfig
}

func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		PreprocessConfig:   vision.DefaultImageNetPreprocessConfig(),
		ConfThreshold:      0.5,
		InputSize:          0,
		NumClasses:         80,
		MaxDetections:      300,
	}
}

func DefaultDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("ltdetr", modelpath.LTDETRDetFile)
	return cfg
}

type DetResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
}
