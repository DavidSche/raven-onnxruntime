package examples

import (
	"fmt"
	"image"
	"os"
	"testing"

	"github.com/up-zero/gotool/imageutil"
)

// createEnginesFromSpecs creates multiple BenchEngines from specs, destroying all on failure.
func createEnginesFromSpecs(specs []EngineSpec) ([]BenchEngine, error) {
	var engines []BenchEngine
	for _, spec := range specs {
		e, err := CreateEngine(spec)
		if err != nil {
			// Cleanup already created engines
			for _, created := range engines {
				created.Destroy()
			}
			return nil, fmt.Errorf("failed to create engine %s/%s: %w", spec.Name, spec.Task, err)
		}
		engines = append(engines, e)
	}
	return engines, nil
}

func destroyEngines(engines []BenchEngine) {
	for _, e := range engines {
		e.Destroy()
	}
}

// ============================================================
// Detection Model Comparison Benchmarks
// ============================================================

// TestBenchDetModels compares all detection models on a single image.
func TestBenchDetModels(t *testing.T) {
	img, err := imageutil.Open("./test.png")
	if err != nil {
		t.Skip("test.png not found, skipping detection benchmark")
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: "../models/yolo26s.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "rfdetr", Task: "det", ModelPath: "../models/rf-detr/rf-detr-small.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "ltdetr", Task: "det", ModelPath: "../models/ltdetr/dinov3_vits16_ltdetr_coco.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "edgecrafter", Task: "det", ModelPath: "../models/edgecrafter/ecdet-s.onnx", LibPath: "../lib/onnxruntime.dll"},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("Detection Models Benchmark", summaries)

	if err := WriteBenchCSV("bench_det_results.csv", summaries); err != nil {
		t.Logf("failed to write CSV: %v", err)
	} else {
		t.Log("Results saved to bench_det_results.csv")
	}
}

// TestBenchDetModelsCuda runs detection benchmark with CUDA.
func TestBenchDetModelsCuda(t *testing.T) {
	img, err := imageutil.Open("./test.png")
	if err != nil {
		t.Skip("test.png not found, skipping CUDA detection benchmark")
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: "../models/yolo26s.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "rfdetr", Task: "det", ModelPath: "../models/rf-detr/rf-detr-small.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "ltdetr", Task: "det", ModelPath: "../models/ltdetr/dinov3_vits16_ltdetr_coco.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "edgecrafter", Task: "det", ModelPath: "../models/edgecrafter/ecdet-s.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = true
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("Detection Models Benchmark (CUDA)", summaries)
}

// ============================================================
// Segmentation Model Comparison Benchmarks
// ============================================================

// TestBenchSegModels compares all segmentation models on a single image.
func TestBenchSegModels(t *testing.T) {
	img, err := imageutil.Open("./test.png")
	if err != nil {
		t.Skip("test.png not found, skipping segmentation benchmark")
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "seg", ModelPath: "../models/yolo26s-seg.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "rfdetr", Task: "seg", ModelPath: "../models/rf-detr/rf-detr-seg-small.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "edgecrafter", Task: "seg", ModelPath: "../models/edgecrafter/ecseg-s.onnx", LibPath: "../lib/onnxruntime.dll"},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("Segmentation Models Benchmark", summaries)

	if err := WriteBenchCSV("bench_seg_results.csv", summaries); err != nil {
		t.Logf("failed to write CSV: %v", err)
	} else {
		t.Log("Results saved to bench_seg_results.csv")
	}
}

// ============================================================
// Pose Estimation Model Comparison Benchmarks
// ============================================================

// TestBenchPoseModels compares all pose estimation models.
func TestBenchPoseModels(t *testing.T) {
	img, err := imageutil.Open("./person.jpg")
	if err != nil {
		t.Skip("person.jpg not found, skipping pose benchmark")
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "pose", ModelPath: "../models/yolo26s-pose.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "edgecrafter", Task: "pose", ModelPath: "../models/edgecrafter/ecpose-s.onnx", LibPath: "../lib/onnxruntime.dll"},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("Pose Estimation Models Benchmark", summaries)
}

// ============================================================
// Batch Inference Benchmarks
// ============================================================

// TestBenchDetBatch compares batch inference across detection models.
func TestBenchDetBatch(t *testing.T) {
	img1, err1 := imageutil.Open("./test.png")
	img2, err2 := imageutil.Open("./ship.jpg")
	if err1 != nil || err2 != nil {
		t.Skip("test.png or ship.jpg not found, skipping batch benchmark")
	}

	imgs := []image.Image{img1, img2}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: "../models/yolo26s.onnx", LibPath: "../lib/onnxruntime.dll", DynBatch: true},
		{Name: "rfdetr", Task: "det", ModelPath: "../models/rf-detr/rf-detr-small.onnx", LibPath: "../lib/onnxruntime.dll", DynBatch: true},
		{Name: "ltdetr", Task: "det", ModelPath: "../models/ltdetr/dinov3_vits16_ltdetr_coco.onnx", LibPath: "../lib/onnxruntime.dll", DynBatch: true},
		{Name: "edgecrafter", Task: "det", ModelPath: "../models/edgecrafter/ecdet-s.onnx", LibPath: "../lib/onnxruntime.dll", DynBatch: true},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	var summaries []BenchSummary
	for _, e := range engines {
		s := RunBatchBench(e, imgs, cfg)
		summaries = append(summaries, s)
	}

	PrintBatchBenchTable("Detection Batch Inference Benchmark", summaries, len(imgs))
}

// ============================================================
// Image Directory Benchmarks
// ============================================================

// TestBenchDetImageDir benchmarks detection models against all images in a directory.
func TestBenchDetImageDir(t *testing.T) {
	dir := "."
	imagePaths, err := loadImagesFromDir(dir)
	if err != nil || len(imagePaths) == 0 {
		t.Skip("no image files found in current directory")
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: "../models/yolo26s.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "rfdetr", Task: "det", ModelPath: "../models/rf-detr/rf-detr-small.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "ltdetr", Task: "det", ModelPath: "../models/ltdetr/dinov3_vits16_ltdetr_coco.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "edgecrafter", Task: "det", ModelPath: "../models/edgecrafter/ecdet-s.onnx", LibPath: "../lib/onnxruntime.dll"},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	summaries, err := RunImageDirBench(engines, dir, cfg)
	if err != nil {
		t.Fatalf("image dir benchmark failed: %v", err)
	}

	PrintBenchSummaryTable("Detection Models - Image Directory Benchmark", summaries)
	PrintBenchDetectionsTable("Detection Counts", summaries)
}

// ============================================================
// Video Benchmarks
// ============================================================

// TestBenchDetVideo benchmarks detection models against video frames.
func TestBenchDetVideo(t *testing.T) {
	checkFFmpeg(t)

	videoPath := ""
	if v := os.Getenv("VIDEO_PATH"); v != "" {
		videoPath = v
	} else {
		found, err := findVideoFile(".")
		if err != nil {
			t.Skip("no video file found. Set VIDEO_PATH env var or place a video file in the examples directory")
		}
		videoPath = found
	}

	t.Logf("Using video: %s", videoPath)

	specs := []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: "../models/yolo26s.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "rfdetr", Task: "det", ModelPath: "../models/rf-detr/rf-detr-small.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "ltdetr", Task: "det", ModelPath: "../models/ltdetr/dinov3_vits16_ltdetr_coco.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "edgecrafter", Task: "det", ModelPath: "../models/edgecrafter/ecdet-s.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	summaries := RunVideoBench(t, engines, videoPath, 5.0)
	PrintBenchSummaryTable("Detection Models - Video Benchmark (5fps)", summaries)
	PrintBenchDetectionsTable("Detection Counts per Frame", summaries)

	if err := WriteBenchCSV("bench_det_video_results.csv", summaries); err != nil {
		t.Logf("failed to write CSV: %v", err)
	}
}

// TestBenchDetVideoStream benchmarks detection models via video stream (pipe).
func TestBenchDetVideoStream(t *testing.T) {
	checkFFmpeg(t)

	videoPath := ""
	if v := os.Getenv("VIDEO_PATH"); v != "" {
		videoPath = v
	} else {
		found, err := findVideoFile(".")
		if err != nil {
			t.Skip("no video file found. Set VIDEO_PATH env var or place a video file in the examples directory")
		}
		videoPath = found
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: "../models/yolo26s.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "rfdetr", Task: "det", ModelPath: "../models/rf-detr/rf-detr-small.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "ltdetr", Task: "det", ModelPath: "../models/ltdetr/dinov3_vits16_ltdetr_coco.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
		{Name: "edgecrafter", Task: "det", ModelPath: "../models/edgecrafter/ecdet-s.onnx", LibPath: "../lib/onnxruntime.dll", UseCuda: true},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	summaries := RunVideoStreamBench(t, engines, videoPath, 5.0)
	PrintBenchSummaryTable("Detection Models - Video Stream Benchmark (5fps)", summaries)
}

// ============================================================
// Custom Comparison (pairwise)
// ============================================================

// TestBenchYOLO26VsLTDETR compares YOLO26 vs LTDETR detection.
func TestBenchYOLO26VsLTDETR(t *testing.T) {
	img, err := imageutil.Open("./test.png")
	if err != nil {
		t.Skip("test.png not found")
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "det", ModelPath: "../models/yolo26s.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "ltdetr", Task: "det", ModelPath: "../models/ltdetr/dinov3_vits16_ltdetr_coco.onnx", LibPath: "../lib/onnxruntime.dll"},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("YOLO26 vs LTDETR", summaries)
}

// TestBenchRFDETRVsEdgeCrafter compares RF-DETR vs EdgeCrafter detection.
func TestBenchRFDETRVsEdgeCrafter(t *testing.T) {
	img, err := imageutil.Open("./test.png")
	if err != nil {
		t.Skip("test.png not found")
	}

	specs := []EngineSpec{
		{Name: "rfdetr", Task: "det", ModelPath: "../models/rf-detr/rf-detr-small.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "edgecrafter", Task: "det", ModelPath: "../models/edgecrafter/ecdet-s.onnx", LibPath: "../lib/onnxruntime.dll"},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("RF-DETR vs EdgeCrafter", summaries)
}

// TestBenchYOLO26VsEdgeCrafterPose compares YOLO26 vs EdgeCrafter pose estimation.
func TestBenchYOLO26VsEdgeCrafterPose(t *testing.T) {
	img, err := imageutil.Open("./person.jpg")
	if err != nil {
		t.Skip("person.jpg not found")
	}

	specs := []EngineSpec{
		{Name: "yolo26", Task: "pose", ModelPath: "../models/yolo26s-pose.onnx", LibPath: "../lib/onnxruntime.dll"},
		{Name: "edgecrafter", Task: "pose", ModelPath: "../models/edgecrafter/ecpose-s.onnx", LibPath: "../lib/onnxruntime.dll"},
	}

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("YOLO26 vs EdgeCrafter Pose Estimation", summaries)
}
