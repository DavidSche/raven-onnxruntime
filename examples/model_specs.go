package examples

// DetectionModelSpecs returns the standard detection benchmark suite.
func DetectionModelSpecs(useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	return []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: ExampleModelPath("yolo26", "yolo26s.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "rfdetr", Task: "det", ModelPath: ExampleModelPath("rf-detr", "rf-detr-small.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "ltdetr", Task: "det", ModelPath: ExampleModelPath("ltdetr", "dinov3_vits16-ltdetr-coco.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "dfine", Task: "det", ModelPath: ExampleModelPath("dfine", "dfine_n_coco.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "edgecrafter", Task: "det", ModelPath: ExampleModelPath("ecdet", "ecdet_s.onnx"), LibPath: libPath, UseCuda: useCuda},
	}
}

// SegmentationModelSpecs returns the standard segmentation benchmark suite.
func SegmentationModelSpecs(useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	return []EngineSpec{
		{Name: "yolo26", Task: "seg", ModelPath: ExampleModelPath("yolo26", "yolo26s-seg.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "rfdetr", Task: "seg", ModelPath: ExampleModelPath("rf-detr", "rf-detr-seg-small.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "dfine", Task: "seg", ModelPath: ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "edgecrafter", Task: "seg", ModelPath: ExampleModelPath("ecdet", "ecseg_s.onnx"), LibPath: libPath, UseCuda: useCuda},
	}
}

// PoseModelSpecs returns the standard pose benchmark suite.
func PoseModelSpecs(useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	return []EngineSpec{
		{Name: "yolo26", Task: "pose", ModelPath: ExampleModelPath("yolo26", "yolo26s-pose.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "edgecrafter", Task: "pose", ModelPath: ExampleModelPath("ecdet", "ecpose_s.onnx"), LibPath: libPath, UseCuda: useCuda},
	}
}

// BatchDetectionModelSpecs returns the standard batch benchmark suite.
func BatchDetectionModelSpecs(useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	return []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: ExampleModelPath("yolo26", "yolo26s.onnx"), LibPath: libPath, UseCuda: useCuda, DynBatch: true},
		{Name: "rfdetr", Task: "det", ModelPath: ExampleModelPath("rf-detr", "rf-detr-small.onnx"), LibPath: libPath, UseCuda: useCuda, DynBatch: true},
		{Name: "ltdetr", Task: "det", ModelPath: ExampleModelPath("ltdetr", "dinov3_vits16-ltdetr-coco.onnx"), LibPath: libPath, UseCuda: useCuda, DynBatch: true},
		{Name: "dfine", Task: "det", ModelPath: ExampleModelPath("dfine", "dfine_n_coco.onnx"), LibPath: libPath, UseCuda: useCuda, DynBatch: true},
		{Name: "edgecrafter", Task: "det", ModelPath: ExampleModelPath("ecdet", "ecdet_s.onnx"), LibPath: libPath, UseCuda: useCuda, DynBatch: true},
	}
}

// DetectionPairSpecs returns a small pairwise detection comparison suite.
func DetectionPairSpecs(left, right string, useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	return []EngineSpec{
		{Name: left, Task: "det", ModelPath: ExampleModelPath(modelFolder(left), modelFile(left)), LibPath: libPath, UseCuda: useCuda},
		{Name: right, Task: "det", ModelPath: ExampleModelPath(modelFolder(right), modelFile(right)), LibPath: libPath, UseCuda: useCuda},
	}
}

// DepthModelSpec describes a depth model benchmark target.
type DepthModelSpec struct {
	Name      string
	ModelPath string
	LibPath   string
	UseCuda   bool
}

// DepthAnything3Specs returns the standard depth benchmark suite.
func DepthAnything3Specs() []DepthModelSpec {
	return []DepthModelSpec{
		{
			Name:      "da3-small",
			ModelPath: ExampleModelPath("da3-small", "da3-small_518x518.onnx"),
			LibPath:   ExampleORTLibraryPath(),
		},
		{
			Name:      "da3-small-sim",
			ModelPath: ExampleModelPath("da3-small", "da3-small_518x518_sim.onnx"),
			LibPath:   ExampleORTLibraryPath(),
		},
	}
}

// ============================================================
// Same-Model Scale Comparison Specs
// ============================================================

// YOLO26ScaleSpecs returns specs for comparing YOLO26 at different scales (n/s/m/l).
// The task parameter selects the model variant: "det", "seg", "pose", "obb", "cls".
func YOLO26ScaleSpecs(task string, useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	var specs []EngineSpec

	scales := []struct {
		suffix string
		label  string
	}{
		{"n", "YOLO26n"},
		{"s", "YOLO26s"},
		{"m", "YOLO26m"},
		{"l", "YOLO26l"},
	}

	for _, sc := range scales {
		filename := "yolo26" + sc.suffix
		if task != "det" {
			filename += "-" + task
		}
		filename += ".onnx"
		specs = append(specs, EngineSpec{
			Name:      "yolo26",
			Task:      task,
			ModelPath: ExampleModelPath("yolo26", filename),
			LibPath:   libPath,
			UseCuda:   useCuda,
		})
		// Override the display name for the adapter
		_ = sc.label // label is used by the test for identification
	}
	return specs
}

// RFDETRScaleSpecs returns specs for comparing RF-DETR at different scales (small/medium/large).
func RFDETRScaleSpecs(useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	return []EngineSpec{
		{Name: "rfdetr", Task: "det", ModelPath: ExampleModelPath("rf-detr", "rf-detr-small.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "rfdetr", Task: "det", ModelPath: ExampleModelPath("rf-detr", "rf-detr-medium.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "rfdetr", Task: "det", ModelPath: ExampleModelPath("rf-detr", "rf-detr-large.onnx"), LibPath: libPath, UseCuda: useCuda},
	}
}

// ============================================================
// YOLOv11 Model Specs
// ============================================================

// YOLOv11ModelSpecs returns specs for YOLOv11 models.
// The task parameter selects the model variant: "det", "seg", "pose", "obb", "cls".
func YOLOv11ModelSpecs(task string, useCuda bool) []EngineSpec {
	libPath := ExampleORTLibraryPath()
	filename := "yolo11s"
	if task != "det" {
		filename += "-" + task
	}
	filename += ".onnx"
	return []EngineSpec{
		{Name: "yolov11", Task: task, ModelPath: ExampleModelPath("yolo11", filename), LibPath: libPath, UseCuda: useCuda},
	}
}

func modelFolder(name string) string {
	switch name {
	case "yolo26":
		return "yolo26"
	case "rfdetr":
		return "rf-detr"
	case "ltdetr":
		return "ltdetr"
	case "dfine":
		return "dfine"
	case "edgecrafter":
		return "ecdet"
	default:
		return name
	}
}

func modelFile(name string) string {
	switch name {
	case "yolo26":
		return "yolo26s.onnx"
	case "rfdetr":
		return "rf-detr-small.onnx"
	case "ltdetr":
		return "dinov3_vits16-ltdetr-coco.onnx"
	case "dfine":
		return "dfine_n_coco.onnx"
	case "edgecrafter":
		return "ecdet_s.onnx"
	default:
		return name + ".onnx"
	}
}
