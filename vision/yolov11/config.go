package yolov11

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
	IOUThreshold  float32 // NMS IOU threshold (default 0.5)
	MaskThreshold float32 // Mask binarization threshold (default 0.5)

	// model parameters
	InputSize     int // default 640
	NumClasses    int // default 80
	NumMaskCoeffs int // default 32
	NumKeyPoints  int // default 17

	// optional parameters
	UseCuda           bool           // (optional) enable CUDA
	NumThreads        int            // (optional) ONNX thread count, default determined by CPU cores
	EnableCpuMemArena bool           // (optional) enable ONNX memory pool
	ApiVersion        ort.ApiVersion // (optional) ONNX Runtime C API version, default ort.DefaultApiVersion
	PreprocessConfig  vision.PreprocessConfig
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		PreprocessConfig:   vision.DefaultPreprocessConfig(),
		ConfThreshold:      0.45,
		IOUThreshold:       0.50,
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
	cfg.ModelPath = modelpath.ModelPath("yolo11", modelpath.YOLO11DetFile)
	return cfg
}

// DefaultSegConfig returns default segmentation configuration
func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = modelpath.ModelPath("yolo11", modelpath.YOLO11SegFile)
	return cfg
}

// DefaultClsConfig returns default classification configuration
func DefaultClsConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 224
	cfg.ModelPath = modelpath.ModelPath("yolo11", modelpath.YOLO11ClsFile)
	return cfg
}

// DefaultPoseConfig returns default pose estimation configuration
func DefaultPoseConfig() Config {
	cfg := DefaultConfig()
	cfg.NumClasses = 1
	cfg.ModelPath = modelpath.ModelPath("yolo11", modelpath.YOLO11PoseFile)
	return cfg
}

// DefaultOBBConfig returns default OBB configuration
func DefaultOBBConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 1024
	cfg.NumClasses = 15
	cfg.ModelPath = modelpath.ModelPath("yolo11", modelpath.YOLO11OBBFile)
	return cfg
}

// imageParams holds image dimension info
type imageParams struct {
	origW, origH int
	scale        float32
	padX, padY   int
}

// candidate holds a detection candidate
type candidate struct {
	box          [4]float32      // model output cx, cy, w, h
	origBox      image.Rectangle // detection box on original image
	score        float32
	classID      int
	maskCoeffs   []float32 // Mask coefficients
	rawKeyPoints []float32
	angle        float32 // rotation angle
}

// DetResult holds object detection result
type DetResult struct {
	// Class ID, e.g.:
	//	0: person
	//  1: bicycle
	//  2: car
	// - For full mapping, see:
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
	ClassID int
	Score   float32
	Box     image.Rectangle // detection box
}

// SegResult holds segmentation result
type SegResult struct {
	// Class ID, e.g.:
	//	0: person
	//  1: bicycle
	//  2: car
	// For full mapping, see:
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
	ClassID int
	Score   float32
	Box     image.Rectangle // segmented rectangular region
	Mask    *image.Gray     // decoded Mask
}

// ClassResult holds classification result
type ClassResult struct {
	// Class ID, e.g.:
	//	436: station wagon
	//	656: minivan
	// For full mapping, see:
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml
	ClassID int
	Score   float32
}

// KeyPoint holds a single keypoint
type KeyPoint struct {
	X, Y  int     // original image coordinates
	Score float32 // visibility/confidence
}

// PoseResult holds pose estimation result
type PoseResult struct {
	ClassID   int
	Score     float32
	Box       image.Rectangle
	KeyPoints []KeyPoint // keypoint list
}

// OBBResult holds rotated object detection result
type OBBResult struct {
	ClassID int
	Score   float32
	// rotated box vertex coordinates: TopLeft, TopRight, BottomRight, BottomLeft
	Corners [4]image.Point

	Center image.Point
	Angle  float32 // radians
}
