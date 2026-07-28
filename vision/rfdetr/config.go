package rfdetr

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
	PreprocessConfig  vision.PreprocessConfig
}

func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		PreprocessConfig:   vision.DefaultImageNetPreprocessConfig(),
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
	cfg.ModelPath = modelpath.ModelPath("rf-detr", modelpath.RFDETRDetFile)
	return cfg
}

func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("rf-detr", modelpath.RFDETRSegFile)
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

type KeypointResult struct {
	X        int     // pixel x coordinate
	Y        int     // pixel y coordinate
	Score    float32 // visibility confidence [0,1]
	Visible  bool    // visible flag (sigmoid > 0.5)
	Findable bool    // findable flag (sigmoid > 0.5)
}

type KpResult struct {
	ClassID   int
	Score     float32
	Box       image.Rectangle
	KeyPoints []KeypointResult
}

func DefaultKpConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("rf-detr", modelpath.RFDETRKeypointFile)
	cfg.NumClasses = 90
	cfg.InputSize = 576
	return cfg
}
