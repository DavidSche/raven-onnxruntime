package yolo26

import (
	"image"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

// Config holds engine initialization parameters
type Config struct {
	ModelPath          string // ONNX model path
	OnnxRuntimeLibPath string // ONNX Runtime dynamic library path

	// inference parameters
	ConfThreshold float32 // confidence threshold (default 0.45)
	IOUThreshold  float32 // NMS IOU threshold (default 0.45)
	MaskThreshold float32 // Mask binarization threshold (default 0.5)

	// model parameters
	InputSize     int // default 640
	NumClasses    int // default 80
	NumMaskCoeffs int // default 32
	NumKeyPoints  int // default 17

	// optional parameters
	UseCuda           bool
	NumThreads        int
	EnableCpuMemArena bool
	ApiVersion        ort.ApiVersion

	// preprocessing configuration
	PreprocessConfig vision.PreprocessConfig
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		PreprocessConfig:   vision.DefaultPreprocessConfig(),
		ConfThreshold:      0.45,
		IOUThreshold:       0.45,
		MaskThreshold:      0.50,
		InputSize:          640,
		NumClasses:         80,
		NumMaskCoeffs:      32,
		NumKeyPoints:       17,
	}
}

// DefaultDetConfig returns default detection configuration
func DefaultDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("yolo26", modelpath.YOLO26DetFile)
	return cfg
}

// DefaultSegConfig returns default segmentation configuration
func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("yolo26", modelpath.YOLO26SegFile)
	return cfg
}

// DefaultClsConfig returns default classification configuration
func DefaultClsConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 224
	cfg.ModelPath = modelpath.ModelPath("yolo26", modelpath.YOLO26ClsFile)
	return cfg
}

// DefaultPoseConfig returns default pose estimation configuration
func DefaultPoseConfig() Config {
	cfg := DefaultConfig()
	cfg.NumClasses = 1
	cfg.ModelPath = modelpath.ModelPath("yolo26", modelpath.YOLO26PoseFile)
	return cfg
}

// DefaultOBBConfig returns default OBB configuration
func DefaultOBBConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 1024
	cfg.NumClasses = 15
	cfg.ModelPath = modelpath.ModelPath("yolo26", modelpath.YOLO26OBBFile)
	return cfg
}

// imageParams holds image dimension info
type imageParams struct {
	origW, origH int
	scale        float32
	padX, padY   int // letterbox padding in model coordinate space
}

// candidate holds a detection candidate
type candidate struct {
	box          [4]float32
	origBox      image.Rectangle
	score        float32
	classID      int
	maskCoeffs   []float32
	rawKeyPoints []float32
	angle        float32
}

// DetResult holds object detection result
type DetResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
}

// SegResult holds segmentation result
type SegResult struct {
	ClassID int
	Score   float32
	Box     image.Rectangle
	Mask    *image.Gray
}

// ClassResult holds classification result
type ClassResult struct {
	ClassID int
	Score   float32
}

// KeyPoint holds a single keypoint
type KeyPoint struct {
	X, Y  int
	Score float32
}

// PoseResult holds pose estimation result
type PoseResult struct {
	ClassID   int
	Score     float32
	Box       image.Rectangle
	KeyPoints []KeyPoint
}

// OBBResult holds rotated object detection result
type OBBResult struct {
	ClassID int
	Score   float32
	// rotated box vertex coordinates: TopLeft, TopRight, BottomRight, BottomLeft
	Corners [4]image.Point
	Center  image.Point
	Angle   float32
}
