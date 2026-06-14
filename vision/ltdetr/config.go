package ltdetr

import (
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
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
}

func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		ConfThreshold:      0.5,
		InputSize:          0,
		NumClasses:         80,
		MaxDetections:      300,
	}
}

func DefaultDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = "./models/ltdetr/dinov3_vits16_ltdetr_coco.onnx"
	return cfg
}

type DetResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
}
