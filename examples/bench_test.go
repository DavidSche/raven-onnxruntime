package examples

import (
	"fmt"
	"image"
	"os"
	"testing"
	"time"

	"github.com/DavidSche/raven-onnxruntime/vision/depth_anything3"
)

// createEnginesFromSpecs creates multiple BenchEngines from specs, destroying all on failure.
func createEnginesFromSpecs(specs []EngineSpec) ([]BenchEngine, error) {
	var engines []BenchEngine
	for _, spec := range specs {
		e, err := CreateEngine(spec)
		if err != nil {
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

// useCudaFromEnv returns whether CUDA should be enabled, checking the
// RAVEN_USE_CUDA environment variable. Returns the explicit value passed
// as default if the env var is not set.
func useCudaFromEnv(defaultVal bool) bool {
	if v := os.Getenv("RAVEN_USE_CUDA"); v != "" {
		return v == "1" || v == "true" || v == "TRUE"
	}
	return defaultVal
}

// ============================================================
// Detection Model Comparison Benchmarks
// ============================================================

func TestBenchDetModels(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := DetectionModelSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "Detection Models Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)

	csvPath := exampleArtifactPath("bench_det_results.csv")
	if err := WriteBenchCSV(csvPath, summaries); err != nil {
		t.Logf("failed to write CSV: %v", err)
	} else {
		t.Logf("Results saved to %s", csvPath)
	}
}

func TestBenchDetModelsCuda(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	specs := DetectionModelSpecs(true)

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

func TestBenchSegModels(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := SegmentationModelSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "Segmentation Models Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)

	csvPath := exampleArtifactPath("bench_seg_results.csv")
	if err := WriteBenchCSV(csvPath, summaries); err != nil {
		t.Logf("failed to write CSV: %v", err)
	} else {
		t.Logf("Results saved to %s", csvPath)
	}
}

func TestBenchSegModelsCuda(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	specs := SegmentationModelSpecs(true)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = true
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("Segmentation Models Benchmark (CUDA)", summaries)
}

// ============================================================
// Pose Estimation Model Comparison Benchmarks
// ============================================================

func TestBenchPoseModels(t *testing.T) {
	img := mustOpenExampleImage(t, "person.jpg")
	useCuda := useCudaFromEnv(false)
	specs := PoseModelSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "Pose Estimation Models Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

func TestBenchPoseModelsCuda(t *testing.T) {
	img := mustOpenExampleImage(t, "person.jpg")
	specs := PoseModelSpecs(true)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = true
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("Pose Estimation Models Benchmark (CUDA)", summaries)
}

// ============================================================
// Batch Inference Benchmarks
// ============================================================

func TestBenchDetBatch(t *testing.T) {
	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")
	imgs := []image.Image{img1, img2}

	useCuda := useCudaFromEnv(false)
	specs := BatchDetectionModelSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	var summaries []BenchSummary
	for _, e := range engines {
		summaries = append(summaries, RunBatchBench(e, imgs, cfg))
	}

	title := "Detection Batch Inference Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBatchBenchTable(title, summaries, len(imgs))
}

func TestBenchDetBatchCuda(t *testing.T) {
	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")
	imgs := []image.Image{img1, img2}

	specs := BatchDetectionModelSpecs(true)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = true
	var summaries []BenchSummary
	for _, e := range engines {
		summaries = append(summaries, RunBatchBench(e, imgs, cfg))
	}

	PrintBatchBenchTable("Detection Batch Inference Benchmark (CUDA)", summaries, len(imgs))
}

// ============================================================
// Batch Size Sweep Benchmark
// ============================================================

func TestBenchDetBatchSweep(t *testing.T) {
	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")
	baseImgs := []image.Image{img1, img2}

	useCuda := useCudaFromEnv(false)
	specs := BatchDetectionModelSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda

	batchSizes := []int{1, 2, 4}
	RunBatchSweepBench(engines, baseImgs, batchSizes, cfg)
}

// ============================================================
// Same-Model Scale Comparison Benchmarks
// ============================================================

func TestBenchYOLO26Scales(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := YOLO26ScaleSpecs("det", useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "YOLO26 Scale Comparison (det)"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

func TestBenchYOLO26ScalesCuda(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	specs := YOLO26ScaleSpecs("det", true)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = true
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("YOLO26 Scale Comparison (det) (CUDA)", summaries)
}

func TestBenchRFDETRScales(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := RFDETRScaleSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "RF-DETR Scale Comparison"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

func TestBenchRFDETRScalesCuda(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	specs := RFDETRScaleSpecs(true)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = true
	summaries := RunImageBenchMulti(engines, img, cfg)
	PrintBenchSummaryTable("RF-DETR Scale Comparison (CUDA)", summaries)
}

// ============================================================
// YOLOv11 Benchmarks
// ============================================================

func TestBenchYOLOv11Det(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := YOLOv11ModelSpecs("det", useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "YOLOv11 Detection Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

func TestBenchYOLOv11Seg(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := YOLOv11ModelSpecs("seg", useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "YOLOv11 Segmentation Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

func TestBenchYOLOv11Pose(t *testing.T) {
	img := mustOpenExampleImage(t, "person.jpg")
	useCuda := useCudaFromEnv(false)
	specs := YOLOv11ModelSpecs("pose", useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "YOLOv11 Pose Estimation Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

// ============================================================
// Depth-Anything-3 Benchmark
// ============================================================

func TestBenchDepthAnything3(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	specs := DepthAnything3Specs()

	for _, spec := range specs {
		cfg := depth_anything3.DefaultConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath

		engine, err := depth_anything3.NewEngine(cfg)
		if err != nil {
			t.Logf("skipping %s: %v", spec.Name, err)
			continue
		}

		// Warmup
		for i := 0; i < 3; i++ {
			_, _ = engine.Predict(img)
		}

		latencies := make([]time.Duration, 0, 30)
		for i := 0; i < 30; i++ {
			start := time.Now()
			_, err := engine.Predict(img)
			latencies = append(latencies, time.Since(start))
			if err != nil {
				t.Logf("inference error: %v", err)
			}
		}

		summary := ComputeSummary(spec.Name, "depth", latencies, nil, 0)
		PrintBenchSummaryTable(fmt.Sprintf("Depth-Anything-3: %s", spec.Name), []BenchSummary{summary})
		engine.Destroy()
	}
}

// ============================================================
// Image Directory Benchmarks
// ============================================================

func TestBenchDetImageDir(t *testing.T) {
	dir := "."
	imagePaths, err := loadImagesFromDir(dir)
	if err != nil || len(imagePaths) == 0 {
		t.Skip("no image files found in current directory")
	}

	useCuda := useCudaFromEnv(false)
	specs := DetectionModelSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries, err := RunImageDirBench(engines, dir, cfg)
	if err != nil {
		t.Fatalf("image dir benchmark failed: %v", err)
	}

	title := "Detection Models - Image Directory Benchmark"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
	PrintBenchDetectionsTable("Detection Counts", summaries)
}

// ============================================================
// Video Benchmarks
// ============================================================

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

	useCuda := useCudaFromEnv(true)
	specs := DetectionModelSpecs(useCuda)
	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	summaries := RunVideoBench(t, engines, videoPath, 5.0)
	title := "Detection Models - Video Benchmark (5fps)"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
	PrintBenchDetectionsTable("Detection Counts per Frame", summaries)

	csvPath := exampleArtifactPath("bench_det_video_results.csv")
	if err := WriteBenchCSV(csvPath, summaries); err != nil {
		t.Logf("failed to write CSV: %v", err)
	}
}

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

	useCuda := useCudaFromEnv(true)
	specs := DetectionModelSpecs(useCuda)
	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	summaries := RunVideoStreamBench(t, engines, videoPath, 5.0)
	title := "Detection Models - Video Stream Benchmark (5fps)"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

// ============================================================
// Custom Comparison (pairwise)
// ============================================================

func TestBenchYOLO26VsLTDETR(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := DetectionPairSpecs("yolo26", "ltdetr", useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "YOLO26 vs LTDETR"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

func TestBenchRFDETRVsEdgeCrafter(t *testing.T) {
	img := mustOpenExampleImage(t, "test.png")
	useCuda := useCudaFromEnv(false)
	specs := DetectionPairSpecs("rfdetr", "edgecrafter", useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "RF-DETR vs EdgeCrafter"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}

func TestBenchYOLO26VsEdgeCrafterPose(t *testing.T) {
	img := mustOpenExampleImage(t, "person.jpg")
	useCuda := useCudaFromEnv(false)
	specs := PoseModelSpecs(useCuda)

	engines, err := createEnginesFromSpecs(specs)
	if err != nil {
		t.Fatalf("failed to create engines: %v", err)
	}
	defer destroyEngines(engines)

	cfg := DefaultBenchConfig()
	cfg.UseCuda = useCuda
	summaries := RunImageBenchMulti(engines, img, cfg)
	title := "YOLO26 vs EdgeCrafter Pose Estimation"
	if useCuda {
		title += " (CUDA)"
	}
	PrintBenchSummaryTable(title, summaries)
}
