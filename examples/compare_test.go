package examples

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/DavidSche/raven-onnxruntime/vision/rfdetr"
	"github.com/DavidSche/raven-onnxruntime/vision/yolo26"
	"github.com/up-zero/gotool/imageutil"
)

type detComparison struct {
	FileName string
	Yolo26   []yolo26.DetResult
	Rfdetr   []rfdetr.DetResult
	Yolo26Ms time.Duration
	RfdetrMs time.Duration
}

func drawDetResults(img image.Image, yoloResults []yolo26.DetResult, rfdetrResults []rfdetr.DetResult) *image.RGBA {
	dst := image.NewRGBA(img.Bounds())
	draw.Draw(dst, img.Bounds(), img, img.Bounds().Min, draw.Src)

	yoloColor := color.RGBA{R: 0, G: 255, B: 0, A: 255}
	rfdetrColor := color.RGBA{R: 0, G: 100, B: 255, A: 255}

	for _, res := range yoloResults {
		imageutil.DrawThickRectOutline(dst, res.Box, yoloColor, 3)
	}

	for _, res := range rfdetrResults {
		imageutil.DrawThickRectOutline(dst, res.Box, rfdetrColor, 3)
	}

	return dst
}

func TestCompareYOLO26VsRFDETR(t *testing.T) {
	yoloCfg := yolo26.DefaultDetConfig()
	yoloCfg.ModelPath = ExampleModelPath("yolo26", "yolo26s.onnx")
	yoloCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	yoloEngine, err := yolo26.NewDetEngine(yoloCfg)
	if err != nil {
		t.Fatalf("failed to initialize YOLO26 engine: %v", err)
	}
	defer yoloEngine.Destroy()

	rfdetrCfg := rfdetr.DefaultDetConfig()
	rfdetrCfg.ModelPath = ExampleModelPath("rf-detr", "rf-detr-small.onnx")
	rfdetrCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	rfdetrEngine, err := rfdetr.NewDetEngine(rfdetrCfg)
	if err != nil {
		t.Fatalf("failed to initialize RF-DETR engine: %v", err)
	}
	defer rfdetrEngine.Destroy()

	imageDir := "."
	imagePaths, err := loadImagesFromDir(imageDir)
	if err != nil {
		t.Fatalf("failed to load images from %s: %v", imageDir, err)
	}
	if len(imagePaths) == 0 {
		t.Skip("no image files found in examples directory")
	}

	comparisons := make([]detComparison, 0, len(imagePaths))

	for _, imgPath := range imagePaths {
		img, err := imageutil.Open(imgPath)
		if err != nil {
			t.Logf("skip %s: %v", imgPath, err)
			continue
		}

		fileName := filepath.Base(imgPath)

		yoloStart := time.Now()
		yoloResults, err := yoloEngine.Predict(img)
		if err != nil {
			t.Fatalf("YOLO26 prediction failed on %s: %v", fileName, err)
		}
		yoloElapsed := time.Since(yoloStart)

		rfdetrStart := time.Now()
		rfdetrResults, err := rfdetrEngine.Predict(img)
		if err != nil {
			t.Fatalf("RF-DETR prediction failed on %s: %v", fileName, err)
		}
		rfdetrElapsed := time.Since(rfdetrStart)

		comp := detComparison{
			FileName: fileName,
			Yolo26:   yoloResults,
			Rfdetr:   rfdetrResults,
			Yolo26Ms: yoloElapsed,
			RfdetrMs: rfdetrElapsed,
		}
		comparisons = append(comparisons, comp)

		overlayImg := drawDetResults(img, yoloResults, rfdetrResults)
		outName := fmt.Sprintf("compare_%s", fileName)
		ext := filepath.Ext(outName)
		outName = strings.TrimSuffix(outName, ext) + ".jpg"
		if err := imageutil.Save(exampleArtifactPath("compare", outName), overlayImg, 80); err != nil {
			t.Logf("failed to save overlay image %s: %v", outName, err)
		}
	}

	fmt.Println("============================================")
	fmt.Println("  YOLO26 vs RF-DETR Detection Comparison")
	fmt.Println("============================================")
	fmt.Printf("%-20s | %8s | %12s | %8s | %12s\n", "File", "YOLO#Det", "YOLO Latency", "RFDETR#Det", "RFDETR Latency")
	fmt.Println(strings.Repeat("-", 80))

	var totalYoloMs, totalRfdetrMs time.Duration
	var totalYoloDet, totalRfdetrDet int

	for _, comp := range comparisons {
		fmt.Printf("%-20s | %8d | %10v | %8d | %10v\n",
			comp.FileName,
			len(comp.Yolo26),
			comp.Yolo26Ms.Round(time.Microsecond),
			len(comp.Rfdetr),
			comp.RfdetrMs.Round(time.Microsecond),
		)

		totalYoloMs += comp.Yolo26Ms
		totalRfdetrMs += comp.RfdetrMs
		totalYoloDet += len(comp.Yolo26)
		totalRfdetrDet += len(comp.Rfdetr)
	}

	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("%-20s | %8d | %10v | %8d | %10v\n",
		"TOTAL/AVG",
		totalYoloDet,
		(totalYoloMs / time.Duration(len(comparisons))).Round(time.Microsecond),
		totalRfdetrDet,
		(totalRfdetrMs / time.Duration(len(comparisons))).Round(time.Microsecond),
	)
	fmt.Println(strings.Repeat("-", 80))

	fmt.Println("\nLegend: Green boxes = YOLO26, Blue boxes = RF-DETR")
	fmt.Println("Overlay images saved under artifacts/benchmarks/compare_*.jpg")
}

func TestCompareYOLO26VsRFDETRSingleImage(t *testing.T) {
	yoloCfg := yolo26.DefaultDetConfig()
	yoloCfg.ModelPath = ExampleModelPath("yolo26", "yolo26s.onnx")
	yoloCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	yoloEngine, err := yolo26.NewDetEngine(yoloCfg)
	if err != nil {
		t.Fatalf("failed to initialize YOLO26 engine: %v", err)
	}
	defer yoloEngine.Destroy()

	rfdetrCfg := rfdetr.DefaultDetConfig()
	rfdetrCfg.ModelPath = ExampleModelPath("rf-detr", "rf-detr-small.onnx")
	rfdetrCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	rfdetrEngine, err := rfdetr.NewDetEngine(rfdetrCfg)
	if err != nil {
		t.Fatalf("failed to initialize RF-DETR engine: %v", err)
	}
	defer rfdetrEngine.Destroy()

	img := mustOpenExampleImage(t, "test.png")

	yoloStart := time.Now()
	yoloResults, err := yoloEngine.Predict(img)
	if err != nil {
		t.Fatalf("YOLO26 prediction failed: %v", err)
	}
	yoloElapsed := time.Since(yoloStart)

	rfdetrStart := time.Now()
	rfdetrResults, err := rfdetrEngine.Predict(img)
	if err != nil {
		t.Fatalf("RF-DETR prediction failed: %v", err)
	}
	rfdetrElapsed := time.Since(rfdetrStart)

	fmt.Println("============================================")
	fmt.Println("  Single Image: YOLO26 vs RF-DETR")
	fmt.Println("============================================")

	fmt.Printf("\n[YOLO26] detected %d objects (%v)\n", len(yoloResults), yoloElapsed.Round(time.Microsecond))
	for i, res := range yoloResults {
		fmt.Printf("  #%d ClassID=%d Score=%.4f Box=%v\n", i, res.ClassID, res.Score, res.Box)
	}

	fmt.Printf("\n[RF-DETR] detected %d objects (%v)\n", len(rfdetrResults), rfdetrElapsed.Round(time.Microsecond))
	for i, res := range rfdetrResults {
		fmt.Printf("  #%d ClassID=%d Score=%.4f Box=%v\n", i, res.ClassID, res.Score, res.Box)
	}

	fmt.Printf("\nLatency comparison: YOLO26=%v | RF-DETR=%v\n", yoloElapsed.Round(time.Microsecond), rfdetrElapsed.Round(time.Microsecond))

	overlayImg := drawDetResults(img, yoloResults, rfdetrResults)
	if err := imageutil.Save(exampleArtifactPath("compare_single.jpg"), overlayImg, 80); err != nil {
		t.Logf("failed to save overlay image: %v", err)
	}
	fmt.Println("Overlay image saved under artifacts/benchmarks/compare_single.jpg (Green=YOLO26, Blue=RF-DETR)")
}

func TestCompareYOLO26VsRFDETRBenchmark(t *testing.T) {
	yoloCfg := yolo26.DefaultDetConfig()
	yoloCfg.ModelPath = ExampleModelPath("yolo26", "yolo26s.onnx")
	yoloCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	yoloEngine, err := yolo26.NewDetEngine(yoloCfg)
	if err != nil {
		t.Fatalf("failed to initialize YOLO26 engine: %v", err)
	}
	defer yoloEngine.Destroy()

	rfdetrCfg := rfdetr.DefaultDetConfig()
	rfdetrCfg.ModelPath = ExampleModelPath("rf-detr", "rf-detr-small.onnx")
	rfdetrCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	rfdetrEngine, err := rfdetr.NewDetEngine(rfdetrCfg)
	if err != nil {
		t.Fatalf("failed to initialize RF-DETR engine: %v", err)
	}
	defer rfdetrEngine.Destroy()

	img := mustOpenExampleImage(t, "test.png")

	const warmupRuns = 5
	const benchRuns = 30

	for i := 0; i < warmupRuns; i++ {
		if _, err := yoloEngine.Predict(img); err != nil {
			t.Fatalf("YOLO26 warmup failed: %v", err)
		}
		if _, err := rfdetrEngine.Predict(img); err != nil {
			t.Fatalf("RF-DETR warmup failed: %v", err)
		}
	}

	var yoloTotal time.Duration
	for i := 0; i < benchRuns; i++ {
		start := time.Now()
		if _, err := yoloEngine.Predict(img); err != nil {
			t.Fatalf("YOLO26 benchmark failed: %v", err)
		}
		yoloTotal += time.Since(start)
	}

	var rfdetrTotal time.Duration
	for i := 0; i < benchRuns; i++ {
		start := time.Now()
		if _, err := rfdetrEngine.Predict(img); err != nil {
			t.Fatalf("RF-DETR benchmark failed: %v", err)
		}
		rfdetrTotal += time.Since(start)
	}

	yoloAvg := yoloTotal / benchRuns
	rfdetrAvg := rfdetrTotal / benchRuns

	fmt.Println("============================================")
	fmt.Println("  Benchmark: YOLO26 vs RF-DETR")
	fmt.Println("============================================")
	fmt.Printf("Runs: %d (warmup: %d)\n\n", benchRuns, warmupRuns)
	fmt.Printf("%-12s | %12s | %12s | %12s\n", "Model", "Total", "Avg", "FPS")
	fmt.Println(strings.Repeat("-", 56))
	fmt.Printf("%-12s | %10v | %10v | %10.1f\n", "YOLO26", yoloTotal.Round(time.Millisecond), yoloAvg.Round(time.Microsecond), 1.0/yoloAvg.Seconds())
	fmt.Printf("%-12s | %10v | %10v | %10.1f\n", "RF-DETR", rfdetrTotal.Round(time.Millisecond), rfdetrAvg.Round(time.Microsecond), 1.0/rfdetrAvg.Seconds())
	fmt.Println(strings.Repeat("-", 56))

	speedup := float64(rfdetrAvg) / float64(yoloAvg)
	if speedup >= 1.0 {
		fmt.Printf("\nYOLO26 is %.2fx faster than RF-DETR\n", speedup)
	} else {
		fmt.Printf("\nRF-DETR is %.2fx faster than YOLO26\n", 1.0/speedup)
	}
}
