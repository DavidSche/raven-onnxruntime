package edgecrafter

import (
	"image"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
)

// Config holds engine initialization parameters for EdgeCrafter models.
type Config struct {
	ModelPath          string
	OnnxRuntimeLibPath string

	ConfThreshold float32
	IOUThreshold  float32
	MaskThreshold float32

	InputSize     int
	NumClasses    int
	MaxDetections int

	// Pose-specific
	NumBodyPoints int
	NumSelect     int

	DynamicBatch      bool
	UseCuda           bool
	NumThreads        int
	EnableCpuMemArena bool
	ApiVersion        ort.ApiVersion
}

// DefaultConfig returns default configuration.
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		ConfThreshold:      0.5,
		IOUThreshold:       0.45,
		MaskThreshold:      0.5,
		InputSize:          0,
		NumClasses:         80,
		MaxDetections:      300,
		NumBodyPoints:      17,
		NumSelect:          60,
	}
}

// DefaultDetConfig returns default detection configuration.
func DefaultDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("ecdet", "ecdet_s.onnx")
	return cfg
}

// DefaultSegConfig returns default segmentation configuration.
func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("ecdet", "ecseg_s.onnx")
	return cfg
}

// DefaultPoseConfig returns default pose configuration.
func DefaultPoseConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("ecdet", "ecpose_s.onnx")
	cfg.NumClasses = 2
	return cfg
}

// DetResult holds object detection result.
type DetResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
}

// SegResult holds instance segmentation result.
type SegResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
	Mask    *image.Gray
}

// KeyPoint holds a single keypoint.
type KeyPoint struct {
	X, Y  int
	Score float32
}

// PoseResult holds pose estimation result.
type PoseResult struct {
	ClassID   int
	Score     float32
	Box       image.Rectangle
	KeyPoints []KeyPoint
}
