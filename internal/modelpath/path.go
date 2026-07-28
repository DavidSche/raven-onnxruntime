package modelpath

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

const (
	YOLO11DetFile  = "yolo11m.onnx"
	YOLO11SegFile  = "yolo11m-seg.onnx"
	YOLO11ClsFile  = "yolo11m-cls.onnx"
	YOLO11PoseFile = "yolo11m-pose.onnx"
	YOLO11OBBFile  = "yolo11m-obb.onnx"

	YOLO26DetFile  = "yolo26m.onnx"
	YOLO26SegFile  = "yolo26m-seg.onnx"
	YOLO26ClsFile  = "yolo26-cls.onnx"
	YOLO26PoseFile = "yolo26m-pose.onnx"
	YOLO26OBBFile  = "yolo26m-obb.onnx"

	RFDETRDetFile      = "rf-detr-base-coco.onnx"
	RFDETRSegFile      = "rf-detr-seg-nano.onnx"
	RFDETRKeypointFile = "rf-detr-keypoint-preview.onnx"

	LTDETRDetFile = "dinov3_vits16-ltdetr-coco.onnx"

	EdgeCrafterDetFile  = "ecdet_s.onnx"
	EdgeCrafterSegFile  = "ecseg_s.onnx"
	EdgeCrafterPoseFile = "ecpose_s.onnx"

	DFINEDetFile    = "dfine_n_coco.onnx"
	DFINESegDetFile = "dfine_s_1x3x640x640.onnx"
	DFINESegSegFile = "dfine_seg_s_1x3x640x640.onnx"
)

var (
	repoRootOnce sync.Once
	repoRoot     string
)

// RepoRoot returns the repository root directory.
func RepoRoot() string {
	repoRootOnce.Do(func() {
		_, file, _, ok := runtime.Caller(0)
		if !ok {
			repoRoot = "."
			return
		}
		repoRoot = filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	})
	return repoRoot
}

// ModelsRoot returns the canonical model directory.
// RAVEN_MODELS_DIR can override the location.
func ModelsRoot() string {
	if env := os.Getenv("RAVEN_MODELS_DIR"); env != "" {
		return env
	}
	return filepath.Join(RepoRoot(), "models")
}

// ModelPath joins the canonical model root with the provided relative parts.
func ModelPath(parts ...string) string {
	all := append([]string{ModelsRoot()}, parts...)
	return filepath.Clean(filepath.Join(all...))
}
