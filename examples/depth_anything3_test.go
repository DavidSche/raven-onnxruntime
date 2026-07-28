package examples

import (
	"fmt"
	"image"
	"os"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/vision/depth_anything3"
	"github.com/up-zero/gotool/imageutil"
)

func TestDepthAnything3(t *testing.T) {
	if os.Getenv("RUN_DEPTH_ANYTHING3_E2E") != "1" {
		t.Skip("set RUN_DEPTH_ANYTHING3_E2E=1 to run the depth inference smoke test")
	}

	cfg := depth_anything3.DefaultConfig()
	cfg.ModelPath = ExampleModelPath("da3-small", "da3-small_518x518.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := depth_anything3.NewEngine(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	result, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("depth map size: %dx%d\n", result.Width, result.Height)
	fmt.Printf("depth range: [%.4f, %.4f]\n", minDepthVal(result.Depth), maxDepthVal(result.Depth))
	fmt.Printf("confidence range: [%.4f, %.4f]\n", minDepthVal(result.Confidence), maxDepthVal(result.Confidence))

	// Save grayscale depth map
	grayImg := result.ToGrayImage()
	if err := imageutil.Save(exampleArtifactPath("depth_anything3", "da3_depth_gray.png"), grayImg, 100); err != nil {
		t.Fatalf("failed to save grayscale depth map: %v", err)
	}

	// Save pseudo-color depth map
	colorImg := depth_anything3.DepthToColormap(result)
	if err := imageutil.Save(exampleArtifactPath("depth_anything3", "da3_depth_colormap.png"), colorImg, 100); err != nil {
		t.Fatalf("failed to save colormap depth map: %v", err)
	}

	// Save heatmap
	heatmapImg := depth_anything3.DepthToHeatmap(result)
	if err := imageutil.Save(exampleArtifactPath("depth_anything3", "da3_depth_heatmap.png"), heatmapImg, 100); err != nil {
		t.Fatalf("failed to save heatmap depth map: %v", err)
	}

	// Save depth overlay
	overlayImg := depth_anything3.DrawDepthOverlay(img, result, 0.5)
	if err := imageutil.Save(exampleArtifactPath("depth_anything3", "da3_depth_overlay.jpg"), overlayImg, 50); err != nil {
		t.Fatalf("failed to save overlay image: %v", err)
	}

	// Save confidence-filtered depth map
	confImg := depth_anything3.DrawDepthWithConfidence(result, 0.3)
	if err := imageutil.Save(exampleArtifactPath("depth_anything3", "da3_depth_conf.png"), confImg, 100); err != nil {
		t.Fatalf("failed to save confidence depth map: %v", err)
	}
}

func minDepthVal(data []float32) float32 {
	m := float32(1e10)
	for _, v := range data {
		if v < m {
			m = v
		}
	}
	return m
}

func maxDepthVal(data []float32) float32 {
	m := float32(-1e10)
	for _, v := range data {
		if v > m {
			m = v
		}
	}
	return m
}

func TestDepthAnything3Preprocess(t *testing.T) {
	if os.Getenv("RUN_DEPTH_ANYTHING3_E2E") != "1" {
		t.Skip("set RUN_DEPTH_ANYTHING3_E2E=1 to run the depth inference smoke test")
	}

	// Use the model's native input size to verify the end-to-end smoke path.
	img := image.NewRGBA(image.Rect(0, 0, 518, 518))

	cfg := depth_anything3.DefaultConfig()
	cfg.ModelPath = ExampleModelPath("da3-small", "da3-small_518x518.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := depth_anything3.NewEngine(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Destroy()

	result, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	if result.Width != 518 || result.Height != 518 {
		t.Errorf("output dimensions mismatch: got %dx%d, want 518x518", result.Width, result.Height)
	}
}
