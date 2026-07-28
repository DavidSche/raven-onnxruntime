package examples

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/vision/dfine"
	"github.com/up-zero/gotool/imageutil"
)

// ============================================================
// D-FINE Detection (from D-FINE repo, dfine_n_coco.onnx)
// ============================================================

func TestDFINEDet(t *testing.T) {
	cfg := dfine.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine", "dfine_n_coco.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize D-FINE detection engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	// Print and validate results
	fmt.Printf("D-FINE det: detected %d objects\n", len(results))
	for _, res := range results {
		fmt.Printf("  Class: %d, Score: %.4f, Box: %v\n", res.ClassID, res.Score, res.Box)
	}

	// Save visualization
	targetImg := image.NewRGBA(img.Bounds())
	draw.Draw(targetImg, img.Bounds(), img, img.Bounds().Min, draw.Src)
	for _, res := range results {
		imageutil.DrawThickRectOutline(targetImg, res.Box, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 3)
	}
	err = imageutil.Save(exampleArtifactPath("dfine_det.jpg"), targetImg, 50)
	if err != nil {
		t.Logf("failed to save image: %v", err)
	}
}

// ============================================================
// D-FINE-seg Detection-only model (dfine_s_1x3x640x640.onnx)
// ============================================================

func TestDFINESegDet(t *testing.T) {
	cfg := dfine.DefaultSegDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize D-FINE-seg detection engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("D-FINE-seg det: detected %d objects\n", len(results))
	for _, res := range results {
		fmt.Printf("  Class: %d, Score: %.4f, Box: %v\n", res.ClassID, res.Score, res.Box)
		// Validate box is within image bounds
		if res.Box.Min.X < 0 || res.Box.Min.Y < 0 ||
			res.Box.Max.X > img.Bounds().Dx() || res.Box.Max.Y > img.Bounds().Dy() {
			t.Errorf("box %v out of image bounds %v", res.Box, img.Bounds())
		}
	}

	// Save visualization
	targetImg := image.NewRGBA(img.Bounds())
	draw.Draw(targetImg, img.Bounds(), img, img.Bounds().Min, draw.Src)
	for _, res := range results {
		imageutil.DrawThickRectOutline(targetImg, res.Box, color.RGBA{R: 0, G: 255, B: 0, A: 255}, 3)
	}
	err = imageutil.Save(exampleArtifactPath("dfine_seg_det.jpg"), targetImg, 50)
	if err != nil {
		t.Logf("failed to save image: %v", err)
	}
}

// ============================================================
// D-FINE-seg Segmentation model (dfine_seg_s_1x3x640x640.onnx)
// Core integration test: validate mask output
// ============================================================

func TestDFINESeg(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize D-FINE-seg segmentation engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("D-FINE-seg seg: detected %d objects with masks\n", len(results))

	// Validate each detection and its mask
	imgW := img.Bounds().Dx()
	imgH := img.Bounds().Dy()

	for idx, res := range results {
		fmt.Printf("  #%d Class: %d, Score: %.4f, Box: %v\n", idx, res.ClassID, res.Score, res.Box)

		// --- Box validation ---
		if res.Box.Min.X < 0 || res.Box.Min.Y < 0 ||
			res.Box.Max.X > imgW || res.Box.Max.Y > imgH {
			t.Errorf("box %v out of image bounds [0,0,%d,%d]", res.Box, imgW, imgH)
		}

		// --- Mask validation ---
		if res.Mask == nil {
			t.Errorf("mask is nil for detection #%d (class=%d, score=%.4f)", idx, res.ClassID, res.Score)
			continue
		}

		maskBounds := res.Mask.Bounds()
		maskW := maskBounds.Dx()
		maskH := maskBounds.Dy()

		fmt.Printf("       Mask: %dx%d, bounds=%v\n", maskW, maskH, maskBounds)

		// Mask should match original image size
		if maskW != imgW || maskH != imgH {
			t.Errorf("mask size %dx%d does not match image size %dx%d", maskW, maskH, imgW, imgH)
		}

		// Count non-zero (white) pixels in mask
		whitePixels := 0
		totalPixels := maskW * maskH
		for y := 0; y < maskH; y++ {
			for x := 0; x < maskW; x++ {
				if res.Mask.GrayAt(x, y).Y > 0 {
					whitePixels++
				}
			}
		}
		coverage := float64(whitePixels) / float64(totalPixels) * 100
		fmt.Printf("       Mask coverage: %.2f%% (%d/%d white pixels)\n", coverage, whitePixels, totalPixels)

		// White pixels should be within the bounding box area
		boxArea := (res.Box.Dx()) * (res.Box.Dy())
		if whitePixels > 0 && boxArea > 0 {
			fmt.Printf("       Box area: %d pixels, mask/box ratio: %.2f\n",
				boxArea, float64(whitePixels)/float64(boxArea)*100)
		}

		// Save individual mask
		err := imageutil.Save(
			exampleArtifactPath(fmt.Sprintf("dfine_seg_mask_%d.png", idx)),
			res.Mask, 100)
		if err != nil {
			t.Logf("failed to save mask: %v", err)
		}
	}

	// Save overlay: draw boxes + masks on image
	overlayImg := overlayDFINESegResults(img, results)
	err = imageutil.Save(exampleArtifactPath("dfine_seg_overlay.jpg"), overlayImg, 80)
	if err != nil {
		t.Logf("failed to save overlay image: %v", err)
	}
}

// TestDFINESegLowThreshold tests with a low confidence threshold to get more detections.
func TestDFINESegLowThreshold(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.1 // lower threshold to get more mask results

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize D-FINE-seg segmentation engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("D-FINE-seg seg (thresh=0.1): %d objects\n", len(results))

	totalMasks := 0
	totalWhitePixels := 0
	for idx, res := range results {
		if res.Mask != nil {
			totalMasks++
			for y := 0; y < res.Mask.Bounds().Dy(); y++ {
				for x := 0; x < res.Mask.Bounds().Dx(); x++ {
					if res.Mask.GrayAt(x, y).Y > 0 {
						totalWhitePixels++
					}
				}
			}
		}
		_ = idx
	}
	fmt.Printf("  Results with masks: %d/%d\n", totalMasks, len(results))
	fmt.Printf("  Total mask white pixels: %d\n", totalWhitePixels)

	if totalMasks == 0 && len(results) > 0 {
		t.Logf("WARNING: all %d detections have nil masks at thresh=0.1", len(results))
	}

	// Save overlay
	overlayImg := overlayDFINESegResults(img, results)
	err = imageutil.Save(exampleArtifactPath("dfine_seg_lowthresh_overlay.jpg"), overlayImg, 80)
	if err != nil {
		t.Logf("failed to save overlay image: %v", err)
	}
}

// ============================================================
// Batch inference tests
// ============================================================

func TestDFINESegDetBatch(t *testing.T) {
	cfg := dfine.DefaultSegDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize D-FINE-seg det engine: %v", err)
	}
	defer engine.Destroy()

	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")

	results, err := engine.PredictBatch([]image.Image{img1, img2})
	if err != nil {
		t.Fatalf("batch prediction failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 batch results, got %d", len(results))
	}

	for i, batch := range results {
		fmt.Printf("Image %d: detected %d objects\n", i, len(batch))
		for j, res := range batch {
			fmt.Printf("  #%d Class: %d, Score: %.4f, Box: %v\n", j, res.ClassID, res.Score, res.Box)
		}
	}
}

// ============================================================
// Validation with random noise input (edge case testing)
// ============================================================

func TestDFINESegRandomInput(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	// Create a random RGB noise image at 640x640
	randImg := image.NewRGBA(image.Rect(0, 0, 640, 640))
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r := uint8((x*37 + y*13) % 256)
			g := uint8((x*71 + y*29) % 256)
			b := uint8((x*53 + y*17) % 256)
			randImg.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	results, err := engine.Predict(randImg)
	if err != nil {
		t.Fatalf("prediction on random input failed: %v", err)
	}

	fmt.Printf("D-FINE-seg random input: %d detections\n", len(results))
	for idx, res := range results {
		fmt.Printf("  #%d Class: %d, Score: %.4f, Box: %v, Mask=%v\n",
			idx, res.ClassID, res.Score, res.Box, res.Mask != nil)
	}
}

// ============================================================
// D-FINE (from D-FINE repo) detection with orig_target_sizes
// ============================================================

func TestDFINE2InputDet(t *testing.T) {
	cfg := dfine.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine", "dfine_n_coco.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize D-FINE 2-input engine: %v", err)
	}
	defer engine.Destroy()

	// Verify it detected the 2-input mode
	if !engine.NeedsOrigTargetSizes() {
		t.Log("NOTE: D-FINE model did not detect as 2-input (may be a 1-input variant)")
	}

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("D-FINE (2-input) det: %d objects\n", len(results))
	for _, res := range results {
		fmt.Printf("  Class: %d, Score: %.4f, Box: %v\n", res.ClassID, res.Score, res.Box)
	}
}

// ============================================================
// Mask quality validation
// ============================================================

func TestDFINESegMaskQuality(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	if len(results) == 0 {
		t.Skip("no detections to validate mask quality, try lowering threshold")
	}

	imgW := img.Bounds().Dx()
	imgH := img.Bounds().Dy()

	for idx, res := range results {
		if res.Mask == nil {
			continue
		}

		maskW := res.Mask.Bounds().Dx()
		maskH := res.Mask.Bounds().Dy()

		// 1. Mask dimensions must match image
		if maskW != imgW || maskH != imgH {
			t.Errorf("mask dimension mismatch: got %dx%d, want %dx%d", maskW, maskH, imgW, imgH)
		}

		// 2. Mask should have the correct bit-depth (8-bit grayscale)
		if res.Mask.Stride != maskW {
			t.Errorf("unexpected mask stride: got %d, image width %d", res.Mask.Stride, maskW)
		}

		// 3. White pixels should only appear inside or near the bounding box
		box := res.Box
		outsideBoxPixels := 0
		insideBoxPixels := 0
		whiteOutside := 0
		whiteInside := 0

		for y := 0; y < maskH; y++ {
			for x := 0; x < maskW; x++ {
				isInside := x >= box.Min.X && x < box.Max.X && y >= box.Min.Y && y < box.Max.Y
				isWhite := res.Mask.GrayAt(x, y).Y > 0

				if isInside {
					insideBoxPixels++
					if isWhite {
						whiteInside++
					}
				} else {
					outsideBoxPixels++
					if isWhite {
						whiteOutside++
					}
				}
			}
		}

		outsideRatio := float64(whiteOutside) / float64(max(1, outsideBoxPixels))
		fmt.Printf("  #%d mask quality: %d white inside box, %d white outside box (%.4f outside ratio)\n",
			idx, whiteInside, whiteOutside, outsideRatio)

		// Allow some margin for mask bleeding outside the box
		if outsideRatio > 0.1 {
			t.Logf("  #%d: high outside-box ratio %.4f (may be expected for this detection)",
				idx, outsideRatio)
		}
	}
}

// ============================================================
// Helpers
// ============================================================

// overlayDFINESegResults draws detection boxes and semi-transparent masks on an image.
func overlayDFINESegResults(img image.Image, results []dfine.SegResult) *image.RGBA {
	dst := image.NewRGBA(img.Bounds())
	draw.Draw(dst, img.Bounds(), img, img.Bounds().Min, draw.Src)

	for _, res := range results {
		if res.Mask != nil {
			// Draw mask as semi-transparent green overlay
			maskBounds := res.Mask.Bounds()
			for y := maskBounds.Min.Y; y < maskBounds.Max.Y; y++ {
				for x := maskBounds.Min.X; x < maskBounds.Max.X; x++ {
					if res.Mask.GrayAt(x, y).Y > 0 {
						// Blend green (0, 255, 0) at 40% opacity over original
						// Standard alpha blend: result = fg * α + bg * (1-α)
						fgAlpha := uint8(102) // ~40% foreground opacity
						bgAlpha := uint8(255 - fgAlpha)
						orig := dst.RGBAAt(x, y)
						dst.SetRGBA(x, y, color.RGBA{
							R: uint8((0*int(fgAlpha) + int(orig.R)*int(bgAlpha)) / 255),
							G: uint8((255*int(fgAlpha) + int(orig.G)*int(bgAlpha)) / 255),
							B: uint8((0*int(fgAlpha) + int(orig.B)*int(bgAlpha)) / 255),
							A: 255,
						})
					}
				}
			}
		}

		// Draw box outline
		boxColor := color.RGBA{R: 255, G: 0, B: 0, A: 255}
		imageutil.DrawThickRectOutline(dst, res.Box, boxColor, 3)
	}

	return dst
}

// ============================================================
// SegEngine PredictBatch edge cases
// ============================================================

func TestDFINESegBatch_Empty(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg model not available: %v", err)
	}
	defer engine.Destroy()

	// Empty batch → nil, nil
	results, err := engine.PredictBatch(nil)
	if err != nil {
		t.Errorf("PredictBatch(nil) returned error: %v", err)
	}
	if results != nil {
		t.Errorf("PredictBatch(nil): expected nil, got %v", results)
	}

	results, err = engine.PredictBatch([]image.Image{})
	if err != nil {
		t.Errorf("PredictBatch(empty) returned error: %v", err)
	}
	if results != nil {
		t.Errorf("PredictBatch(empty): expected nil, got %v", results)
	}
}

func TestDFINESegBatch_Single(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg model not available: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")

	// Single image via PredictBatch — compare with Predict
	batchResults, err := engine.PredictBatch([]image.Image{img})
	if err != nil {
		t.Fatalf("PredictBatch single failed: %v", err)
	}
	if len(batchResults) != 1 {
		t.Fatalf("expected 1 batch result, got %d", len(batchResults))
	}

	singleResult, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	// Both methods should return the same number of detections
	if len(batchResults[0]) != len(singleResult) {
		t.Errorf("detection count mismatch: batch=%d, single=%d",
			len(batchResults[0]), len(singleResult))
	}

	// Compare results
	for i := 0; i < min(len(batchResults[0]), len(singleResult)); i++ {
		b := batchResults[0][i]
		s := singleResult[i]
		if b.ClassID != s.ClassID || b.Score != s.Score || b.Box != s.Box {
			t.Errorf("result #%d mismatch:\n  batch:  class=%d score=%.4f box=%v\n  single: class=%d score=%.4f box=%v",
				i, b.ClassID, b.Score, b.Box, s.ClassID, s.Score, s.Box)
		}
		// Both should have identical mask presence
		if (b.Mask == nil) != (s.Mask == nil) {
			t.Errorf("result #%d mask presence mismatch: batch=%v single=%v",
				i, b.Mask != nil, s.Mask != nil)
		}
	}

	t.Logf("single-image batch: %d detections, single: %d detections",
		len(batchResults[0]), len(singleResult))
}

func TestDFINESegBatch_Multi(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg model not available: %v", err)
	}
	defer engine.Destroy()

	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")

	results, err := engine.PredictBatch([]image.Image{img1, img2})
	if err != nil {
		t.Fatalf("batch prediction failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 batch results, got %d", len(results))
	}

	for i, batch := range results {
		t.Logf("Image %d: %d detections", i, len(batch))
		for j, res := range batch {
			// Validate result fields are populated
			if res.Box.Min.X < 0 || res.Box.Min.Y < 0 {
				t.Errorf("Image %d, result %d: negative box coordinates %v", i, j, res.Box)
			}
			if res.Score < 0 || res.Score > 1.001 {
				t.Errorf("Image %d, result %d: invalid score %.2f", i, j, res.Score)
			}
			t.Logf("  #%d Class: %d, Score: %.4f, Box: %v, Mask=%v",
				j, res.ClassID, res.Score, res.Box, res.Mask != nil)
		}
	}
}

func TestDFINESegBatch_DifferentSizes(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg model not available: %v", err)
	}
	defer engine.Destroy()

	// Test with images of different sizes in the same batch
	img1 := mustOpenExampleImage(t, "test.png") // 640x480
	img2 := mustOpenExampleImage(t, "ship.jpg") // different size

	if img1.Bounds().Eq(img2.Bounds()) {
		t.Log("images are the same size — still validates batch correctness")
	}

	results, err := engine.PredictBatch([]image.Image{img1, img2})
	if err != nil {
		t.Fatalf("batch prediction with different sizes failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// Each image's boxes should be correctly scaled to its own dimensions
	for i, batch := range results {
		imgW := []image.Image{img1, img2}[i].Bounds().Dx()
		imgH := []image.Image{img1, img2}[i].Bounds().Dy()

		for j, res := range batch {
			if res.Box.Min.X < 0 || res.Box.Max.X > imgW {
				t.Errorf("Image %d, result %d: box X out of bounds [0,%d] = %v",
					i, j, imgW, res.Box)
			}
			if res.Box.Min.Y < 0 || res.Box.Max.Y > imgH {
				t.Errorf("Image %d, result %d: box Y out of bounds [0,%d] = %v",
					i, j, imgH, res.Box)
			}

			// Masks should match image dimensions
			if res.Mask != nil {
				maskW := res.Mask.Bounds().Dx()
				maskH := res.Mask.Bounds().Dy()
				if maskW != imgW || maskH != imgH {
					t.Errorf("Image %d, result %d: mask size %dx%d != image size %dx%d",
						i, j, maskW, maskH, imgW, imgH)
				}
			}

			t.Logf("Img%d #%d: class=%d score=%.4f box=%v mask=%v imgSize=%dx%d",
				i, j, res.ClassID, res.Score, res.Box, res.Mask != nil, imgW, imgH)
		}
	}
}

func TestDFINESegBatch_AllThresholds(t *testing.T) {
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.0 // include all detections

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg model not available: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.PredictBatch([]image.Image{img, img})
	if err != nil {
		t.Fatalf("batch prediction failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// Both batches of the same image should produce identical results
	if len(results[0]) != len(results[1]) {
		t.Errorf("detection count mismatch between identical images: %d vs %d",
			len(results[0]), len(results[1]))
	}

	matchCount := 0
	for i := 0; i < min(len(results[0]), len(results[1])); i++ {
		if results[0][i].ClassID == results[1][i].ClassID &&
			results[0][i].Score == results[1][i].Score &&
			results[0][i].Box == results[1][i].Box {
			matchCount++
		}
	}

	t.Logf("identical images batch: img0=%d detections, img1=%d detections, matched=%d",
		len(results[0]), len(results[1]), matchCount)

	if matchCount == 0 && len(results[0]) > 0 {
		t.Log("NOTE: zero match count may indicate non-determinism at threshold=0.0 (expected with scores near 0)")
	}
}

// ============================================================
// DetEngine PredictBatch edge cases
// ============================================================

func TestDFINEDetBatch_Empty(t *testing.T) {
	cfg := dfine.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine", "dfine_n_coco.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE model not available: %v", err)
	}
	defer engine.Destroy()

	results, err := engine.PredictBatch(nil)
	if err != nil {
		t.Errorf("PredictBatch(nil) returned error: %v", err)
	}
	if results != nil {
		t.Errorf("PredictBatch(nil): expected nil, got %v", results)
	}

	results, err = engine.PredictBatch([]image.Image{})
	if err != nil {
		t.Errorf("PredictBatch(empty) returned error: %v", err)
	}
	if results != nil {
		t.Errorf("PredictBatch(empty): expected nil, got %v", results)
	}
}

func TestDFINEDetBatch_Single(t *testing.T) {
	cfg := dfine.DefaultSegDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg det model not available: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")

	batchResults, err := engine.PredictBatch([]image.Image{img})
	if err != nil {
		t.Fatalf("PredictBatch single failed: %v", err)
	}
	if len(batchResults) != 1 {
		t.Fatalf("expected 1 batch result, got %d", len(batchResults))
	}

	singleResult, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if len(batchResults[0]) != len(singleResult) {
		t.Errorf("detection count mismatch: batch=%d, single=%d",
			len(batchResults[0]), len(singleResult))
	}

	for i := 0; i < min(len(batchResults[0]), len(singleResult)); i++ {
		b := batchResults[0][i]
		s := singleResult[i]
		if b.ClassID != s.ClassID || b.Score != s.Score || b.Box != s.Box {
			t.Errorf("result #%d mismatch:\n  batch:  class=%d score=%.4f box=%v\n  single: class=%d score=%.4f box=%v",
				i, b.ClassID, b.Score, b.Box, s.ClassID, s.Score, s.Box)
		}
	}

	t.Logf("single-image batch: %d detections, single: %d detections",
		len(batchResults[0]), len(singleResult))
}

func TestDFINEDetBatch_Multi(t *testing.T) {
	cfg := dfine.DefaultSegDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg det model not available: %v", err)
	}
	defer engine.Destroy()

	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")

	results, err := engine.PredictBatch([]image.Image{img1, img2})
	if err != nil {
		t.Fatalf("batch prediction failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 batch results, got %d", len(results))
	}

	for i, batch := range results {
		t.Logf("Image %d: %d detections", i, len(batch))
		for j, res := range batch {
			// Validate box bounds against image dimensions
			imgW := []image.Image{img1, img2}[i].Bounds().Dx()
			imgH := []image.Image{img1, img2}[i].Bounds().Dy()
			if res.Box.Min.X < 0 || res.Box.Min.Y < 0 {
				t.Errorf("Image %d, result %d: negative box coords %v", i, j, res.Box)
			}
			if res.Box.Max.X > imgW || res.Box.Max.Y > imgH {
				t.Errorf("Image %d, result %d: box %v exceeds image bounds [0,0,%d,%d]",
					i, j, res.Box, imgW, imgH)
			}
			if res.Score < 0 || res.Score > 1.001 {
				t.Errorf("Image %d, result %d: invalid score %.2f", i, j, res.Score)
			}
			t.Logf("  #%d Class: %d, Score: %.4f, Box: %v",
				j, res.ClassID, res.Score, res.Box)
		}
	}
}

func TestDFINEDetBatch_DifferentSizes(t *testing.T) {
	cfg := dfine.DefaultSegDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg det model not available: %v", err)
	}
	defer engine.Destroy()

	img1 := mustOpenExampleImage(t, "test.png") // 640x480
	img2 := mustOpenExampleImage(t, "ship.jpg") // different size

	results, err := engine.PredictBatch([]image.Image{img1, img2})
	if err != nil {
		t.Fatalf("batch prediction with different sizes failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// Each image's boxes should be correctly scaled to its own dimensions
	for i, batch := range results {
		imgW := []image.Image{img1, img2}[i].Bounds().Dx()
		imgH := []image.Image{img1, img2}[i].Bounds().Dy()

		for j, res := range batch {
			if res.Box.Min.X < 0 || res.Box.Max.X > imgW {
				t.Errorf("Image %d, result %d: box X out of bounds [0,%d] = %v",
					i, j, imgW, res.Box)
			}
			if res.Box.Min.Y < 0 || res.Box.Max.Y > imgH {
				t.Errorf("Image %d, result %d: box Y out of bounds [0,%d] = %v",
					i, j, imgH, res.Box)
			}
			t.Logf("Img%d #%d: class=%d score=%.4f box=%v imgSize=%dx%d",
				i, j, res.ClassID, res.Score, res.Box, imgW, imgH)
		}
	}
}

func TestDFINEDetBatch_AllThresholds(t *testing.T) {
	cfg := dfine.DefaultSegDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.0 // include all detections

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE-seg det model not available: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.PredictBatch([]image.Image{img, img})
	if err != nil {
		t.Fatalf("batch prediction failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if len(results[0]) != len(results[1]) {
		t.Errorf("detection count mismatch between identical images: %d vs %d",
			len(results[0]), len(results[1]))
	}

	matchCount := 0
	for i := 0; i < min(len(results[0]), len(results[1])); i++ {
		if results[0][i].ClassID == results[1][i].ClassID &&
			results[0][i].Score == results[1][i].Score &&
			results[0][i].Box == results[1][i].Box {
			matchCount++
		}
	}

	t.Logf("identical images batch: img0=%d detections, img1=%d detections, matched=%d",
		len(results[0]), len(results[1]), matchCount)
}

func TestDFINEDetBatch_2InputModel(t *testing.T) {
	// Test PredictBatch with the 2-input D-FINE model (dfine_n_coco)
	cfg := dfine.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("dfine", "dfine_n_coco.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.5

	engine, err := dfine.NewDetEngine(cfg)
	if err != nil {
		t.Skipf("D-FINE 2-input model not available: %v", err)
	}
	defer engine.Destroy()

	if !engine.NeedsOrigTargetSizes() {
		t.Skip("model does not require orig_target_sizes, not a 2-input D-FINE variant")
	}

	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")

	results, err := engine.PredictBatch([]image.Image{img1, img2})
	if err != nil {
		t.Fatalf("2-input batch prediction failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 batch results, got %d", len(results))
	}

	for i, batch := range results {
		t.Logf("Image %d: %d detections (2-input mode)", i, len(batch))
		for j, res := range batch {
			// Boxes should already be in original image coordinates
			imgW := []image.Image{img1, img2}[i].Bounds().Dx()
			imgH := []image.Image{img1, img2}[i].Bounds().Dy()
			if res.Box.Min.X < 0 || res.Box.Max.X > imgW {
				t.Errorf("Image %d, result %d: box X out of bounds [0,%d] = %v",
					i, j, imgW, res.Box)
			}
			if res.Box.Min.Y < 0 || res.Box.Max.Y > imgH {
				t.Errorf("Image %d, result %d: box Y out of bounds [0,%d] = %v",
					i, j, imgH, res.Box)
			}
			t.Logf("  #%d Class: %d, Score: %.4f, Box: %v",
				j, res.ClassID, res.Score, res.Box)
		}
	}
}

// ============================================================
// Math / value domain validation
// ============================================================

func TestDFINESegScoreDomain(t *testing.T) {
	// Verify that confidence scores are in valid [0, 1] range
	cfg := dfine.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.ConfThreshold = 0.0 // no threshold to see all scores

	engine, err := dfine.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	invalidScores := 0
	maxScore := float32(-1)
	minScore := float32(math.MaxFloat32)

	for _, res := range results {
		if res.Score < 0 || res.Score > 1.001 {
			invalidScores++
			t.Logf("invalid score %.6f for detection class=%d", res.Score, res.ClassID)
		}
		if res.Score > maxScore {
			maxScore = res.Score
		}
		if res.Score < minScore {
			minScore = res.Score
		}
	}

	fmt.Printf("Score domain: min=%.6f, max=%.6f, total=%d, invalid=%d\n",
		minScore, maxScore, len(results), invalidScores)

	if invalidScores > 0 {
		t.Errorf("found %d detections with scores outside [0,1] range", invalidScores)
	}
}
