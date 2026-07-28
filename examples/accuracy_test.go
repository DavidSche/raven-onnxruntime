package examples

import (
	"fmt"
	"os"
	"testing"
)

// ============================================================
// COCO Dataset Availability Check
// ============================================================

// skipIfNoCOCO skips the test if COCO dataset is not available.
func skipIfNoCOCO(t *testing.T) *COCODataset {
	t.Helper()

	annPath := COCOAnnotationPath()
	imgDir := COCOImageDir()

	if _, err := os.Stat(annPath); os.IsNotExist(err) {
		t.Skipf("COCO annotations not found at %s. Set RAVEN_COCO_ANN or run data preparation script.", annPath)
	}

	if _, err := os.Stat(imgDir); os.IsNotExist(err) {
		t.Skipf("COCO images not found at %s. Set RAVEN_COCO_DIR or run data preparation script.", imgDir)
	}

	ds, err := LoadCOCODataset(annPath)
	if err != nil {
		t.Fatalf("Failed to load COCO annotations: %v", err)
	}

	fmt.Printf("COCO dataset loaded: %d images, %d annotations, %d categories\n",
		len(ds.Images), len(ds.Annotations), len(ds.Categories))

	return ds
}

// maxImagesFromEnv returns the maximum number of images to evaluate.
// Controlled by RAVEN_ACC_MAX_IMAGES env var. Default: 100. Set to 0 for all.
func maxImagesFromEnv() int {
	v := os.Getenv("RAVEN_ACC_MAX_IMAGES")
	if v == "" {
		return 100
	}
	var n int
	fmt.Sscanf(v, "%d", &n)
	return n
}

// createAccEnginesFromSpecs creates accuracy engines from specs.
func createAccEnginesFromSpecs(t *testing.T, specs []AccuracySpec) []AccuracyEngine {
	t.Helper()
	var engines []AccuracyEngine
	for _, spec := range specs {
		eng, err := CreateAccuracyEngine(spec)
		if err != nil {
			t.Logf("Warning: failed to create engine %s: %v", spec.Name, err)
			continue
		}
		engines = append(engines, eng)
	}
	return engines
}

// destroyAccEngines destroys all accuracy engines.
func destroyAccEngines(engines []AccuracyEngine) {
	for _, eng := range engines {
		eng.Destroy()
	}
}

// ============================================================
// Detection Accuracy Tests
// ============================================================

// TestAccDetModels evaluates detection accuracy for all models on COCO.
func TestAccDetModels(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	specs := DetAccuracySpecs(useCuda)
	engines := createAccEnginesFromSpecs(t, specs)
	defer destroyAccEngines(engines)

	if len(engines) == 0 {
		t.Fatal("No engines created")
	}

	CompareAccuracy(engines, cocoDS, imgDir, maxImages)
}

// TestAccDetModelsCuda evaluates detection accuracy with CUDA.
func TestAccDetModelsCuda(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()

	specs := DetAccuracySpecs(true)
	engines := createAccEnginesFromSpecs(t, specs)
	defer destroyAccEngines(engines)

	if len(engines) == 0 {
		t.Fatal("No engines created")
	}

	CompareAccuracy(engines, cocoDS, imgDir, maxImages)
}

// ============================================================
// Segmentation Accuracy Tests
// ============================================================

// TestAccSegModels evaluates segmentation accuracy for all models on COCO.
func TestAccSegModels(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	specs := SegAccuracySpecs(useCuda)
	engines := createAccEnginesFromSpecs(t, specs)
	defer destroyAccEngines(engines)

	if len(engines) == 0 {
		t.Fatal("No engines created")
	}

	CompareAccuracy(engines, cocoDS, imgDir, maxImages)
}

// ============================================================
// YOLO26 Scale Accuracy Tests
// ============================================================

// TestAccYOLO26Scales evaluates YOLO26 detection accuracy at different scales.
func TestAccYOLO26Scales(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	specs := YOLO26ScaleAccuracySpecs("det", useCuda)
	engines := createAccEnginesFromSpecs(t, specs)
	defer destroyAccEngines(engines)

	if len(engines) == 0 {
		t.Fatal("No engines created")
	}

	CompareAccuracy(engines, cocoDS, imgDir, maxImages)
}

// TestAccYOLO26ScalesCuda evaluates YOLO26 detection accuracy at different scales with CUDA.
func TestAccYOLO26ScalesCuda(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()

	specs := YOLO26ScaleAccuracySpecs("det", true)
	engines := createAccEnginesFromSpecs(t, specs)
	defer destroyAccEngines(engines)

	if len(engines) == 0 {
		t.Fatal("No engines created")
	}

	CompareAccuracy(engines, cocoDS, imgDir, maxImages)
}

// ============================================================
// Single Model Accuracy Tests
// ============================================================

// TestAccYOLO26Det evaluates YOLO26 detection accuracy.
func TestAccYOLO26Det(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	eng, err := NewYOLO26DetAccEngine(
		ExampleModelPath("yolo26", "yolo26m.onnx"),
		ExampleORTLibraryPath(), useCuda,
	)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer eng.Destroy()

	result, err := EvaluateAccuracy(eng, cocoDS, imgDir, maxImages)
	if err != nil {
		t.Fatalf("Evaluation failed: %v", err)
	}

	PrintMAPReport("YOLO26-det Accuracy", result)

	if result.MAP50 < 0.1 {
		t.Errorf("mAP@0.50 = %.4f is suspiciously low, expected > 0.1", result.MAP50)
	}
}

// TestAccRFDETRDet evaluates RF-DETR detection accuracy.
func TestAccRFDETRDet(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	eng, err := NewRFDETRDetAccEngine(
		ExampleModelPath("rf-detr", "rf-detr-medium.onnx"),
		ExampleORTLibraryPath(), useCuda,
	)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer eng.Destroy()

	result, err := EvaluateAccuracy(eng, cocoDS, imgDir, maxImages)
	if err != nil {
		t.Fatalf("Evaluation failed: %v", err)
	}

	PrintMAPReport("RF-DETR-det Accuracy", result)

	if result.MAP50 < 0.1 {
		t.Errorf("mAP@0.50 = %.4f is suspiciously low, expected > 0.1", result.MAP50)
	}
}

// TestAccYOLOv11Det evaluates YOLOv11 detection accuracy.
func TestAccYOLOv11Det(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	eng, err := NewYOLOv11DetAccEngine(
		ExampleModelPath("yolov11", "yolov11m.onnx"),
		ExampleORTLibraryPath(), useCuda,
	)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer eng.Destroy()

	result, err := EvaluateAccuracy(eng, cocoDS, imgDir, maxImages)
	if err != nil {
		t.Fatalf("Evaluation failed: %v", err)
	}

	PrintMAPReport("YOLOv11-det Accuracy", result)

	if result.MAP50 < 0.1 {
		t.Errorf("mAP@0.50 = %.4f is suspiciously low, expected > 0.1", result.MAP50)
	}
}

// ============================================================
// D-FINE Accuracy Tests
// ============================================================

// TestAccDFINEDet evaluates D-FINE detection accuracy on COCO.
func TestAccDFINEDet(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	eng, err := NewDFINEDetAccEngine(
		ExampleModelPath("dfine", "dfine_n_coco.onnx"),
		ExampleORTLibraryPath(), useCuda,
	)
	if err != nil {
		t.Fatalf("Failed to create D-FINE DetEngine: %v", err)
	}
	defer eng.Destroy()

	result, err := EvaluateAccuracy(eng, cocoDS, imgDir, maxImages)
	if err != nil {
		t.Fatalf("Evaluation failed: %v", err)
	}

	PrintMAPReport("D-FINE-det Accuracy", result)

	if result.MAP50 < 0.1 {
		t.Errorf("mAP@0.50 = %.4f is suspiciously low, expected > 0.1", result.MAP50)
	}
}

// TestAccDFINESeg evaluates D-FINE-seg segmentation accuracy on COCO.
func TestAccDFINESeg(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	maxImages := maxImagesFromEnv()
	useCuda := useCudaFromEnv(false)

	eng, err := NewDFINESegAccEngine(
		ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx"),
		ExampleORTLibraryPath(), useCuda,
	)
	if err != nil {
		t.Fatalf("Failed to create D-FINE SegEngine: %v", err)
	}
	defer eng.Destroy()

	result, err := EvaluateAccuracy(eng, cocoDS, imgDir, maxImages)
	if err != nil {
		t.Fatalf("Evaluation failed: %v", err)
	}

	PrintMAPReport("D-FINE-seg Accuracy", result)

	if result.MAP50 < 0.1 {
		t.Errorf("mAP@0.50 = %.4f is suspiciously low, expected > 0.1", result.MAP50)
	}
}

// ============================================================
// Full COCO Evaluation (all 5000 images)
// ============================================================

// TestAccDetFull runs full COCO val2017 evaluation (5000 images).
// Set RAVEN_ACC_MAX_IMAGES=0 to evaluate all images.
func TestAccDetFull(t *testing.T) {
	cocoDS := skipIfNoCOCO(t)
	imgDir := COCOImageDir()
	useCuda := useCudaFromEnv(false)

	maxImages := 0
	if v := os.Getenv("RAVEN_ACC_MAX_IMAGES"); v != "" {
		fmt.Sscanf(v, "%d", &maxImages)
	}

	specs := DetAccuracySpecs(useCuda)
	engines := createAccEnginesFromSpecs(t, specs)
	defer destroyAccEngines(engines)

	if len(engines) == 0 {
		t.Fatal("No engines created")
	}

	CompareAccuracy(engines, cocoDS, imgDir, maxImages)
}
