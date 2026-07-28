package dfine

import (
	"image"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

// Config holds engine initialization parameters for D-FINE and D-FINE-seg.
type Config struct {
	ModelPath          string // ONNX model path
	OnnxRuntimeLibPath string // ONNX Runtime dynamic library path

	// Inference parameters
	ConfThreshold float32 // confidence threshold (default 0.5)
	MaskThreshold float32 // mask binarization threshold (default 0.5)

	// Model parameters
	InputSize     int // model input resolution (default 640)
	NumClasses    int // number of COCO classes (default 80)
	MaxDetections int // maximum number of detections after score filtering (default 300)
	NumMasks      int // number of mask queries per image (default 300)
	MaskHeight    int // mask output height (default 160)
	MaskWidth     int // mask output width (default 160)

	// Runtime options
	DynamicBatch      bool
	UseCuda           bool
	NumThreads        int
	EnableCpuMemArena bool
	ApiVersion        ort.ApiVersion

	// Preprocessing
	PreprocessConfig vision.PreprocessConfig
}

// DefaultConfig returns default configuration.
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		PreprocessConfig:   vision.DefaultImageNetPreprocessConfig(),
		ConfThreshold:      0.5,
		MaskThreshold:      0.5,
		InputSize:          640,
		NumClasses:         80,
		MaxDetections:      300,
		NumMasks:           300,
		MaskHeight:         160,
		MaskWidth:          160,
	}
}

// DefaultDetConfig returns default detection configuration for D-FINE models.
func DefaultDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("dfine", modelpath.DFINEDetFile)
	return cfg
}

// DefaultSegDetConfig returns default detection configuration for D-FINE-seg models.
func DefaultSegDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("dfine_seg", modelpath.DFINESegDetFile)
	return cfg
}

// DefaultSegConfig returns default segmentation configuration for D-FINE-seg models.
func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("dfine_seg", modelpath.DFINESegSegFile)
	return cfg
}

// imageParams holds image dimension info for post-processing.
type imageParams struct {
	origW, origH int
}

// DetResult holds a single detection result.
type DetResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
}

// SegResult holds a single segmentation result.
type SegResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
	Mask    *image.Gray
}
