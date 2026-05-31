package rfdetr

import (
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
)

type Config struct {
	ModelPath          string
	OnnxRuntimeLibPath string

	ConfThreshold float32
	IOUThreshold  float32
	MaskThreshold float32

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
		IOUThreshold:       0.45,
		MaskThreshold:      0.5,
		InputSize:          0,
		NumClasses:         80,
		MaxDetections:      300,
	}
}

func DefaultDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = "./models/rf-detr/rf-detr-base-coco.onnx"
	return cfg
}

func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = "./models/rf-detr/rf-detr-seg-nano.onnx"
	return cfg
}

type DetResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
}

type SegResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
	Mask    *image.Gray
}
