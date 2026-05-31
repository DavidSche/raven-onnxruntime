package yolo26

import (
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
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
	UseCuda           bool           // (optional) enable CUDA
	NumThreads        int            // (optional) ONNX thread count, default determined by CPU cores
	EnableCpuMemArena bool           // (optional) enable ONNX memory pool
	ApiVersion        ort.ApiVersion // (optional) ONNX Runtime C API version, default ort.DefaultApiVersion
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

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
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
	cfg.ModelPath = "./yolo26_weights/yolo26m.onnx"
	return cfg
}

// imageParams holds image dimension info
type imageParams struct {
	origW, origH int
	scale        float32
}

// DefaultSegConfig returns default segmentation configuration
func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = "./yolo26_weights/yolo26m-seg.onnx"
	return cfg
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

// DefaultClsConfig returns default classification configuration
func DefaultClsConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 224
	cfg.ModelPath = "./yolo26_weights/yolo26-cls.onnx"
	return cfg
}

// KeyPoint holds a single keypoint
type KeyPoint struct {
	X, Y  int     // original image coordinates
	Score float32 // visibility/confidence
}

// PoseResult holds pose estimation result
type PoseResult struct {
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml
	ClassID   int
	Score     float32
	Box       image.Rectangle
	KeyPoints []KeyPoint // keypoint list
}

// DefaultPoseConfig returns default pose estimation configuration
func DefaultPoseConfig() Config {
	cfg := DefaultConfig()
	cfg.NumClasses = 1
	cfg.ModelPath = "./yolo26_weights/yolo26m-pose.onnx"
	return cfg
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

// DefaultOBBConfig returns default OBB configuration
func DefaultOBBConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 1024
	cfg.NumClasses = 15
	cfg.ModelPath = "./yolo26_weights/yolo26m-obb.onnx"
	return cfg
}
