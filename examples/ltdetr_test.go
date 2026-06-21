package examples

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/vision/ltdetr"
	"github.com/up-zero/gotool/imageutil"
)

func TestLTDETRDet(t *testing.T) {
	cfg := ltdetr.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("ltdetr", "dinov3_vits16-ltdetr-coco.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := ltdetr.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	targetImg := image.NewRGBA(img.Bounds())
	draw.Draw(targetImg, img.Bounds(), img, img.Bounds().Min, draw.Src)
	fmt.Printf("detected objects: %d\n", len(results))
	for _, res := range results {
		fmt.Printf("Class: %d, Score: %.2f, Box: %v\n", res.ClassID, res.Score, res.Box)
		imageutil.DrawThickRectOutline(targetImg, res.Box, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 3)
	}
	err = imageutil.Save(exampleArtifactPath("ltdetr_det.jpg"), targetImg, 50)
	if err != nil {
		fmt.Printf("failed to save image: %v", err)
	}
}

func TestLTDETRDetBatch(t *testing.T) {
	cfg := ltdetr.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("ltdetr", "dinov3_vits16-ltdetr-coco.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	cfg.DynamicBatch = true

	engine, err := ltdetr.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img1 := mustOpenExampleImage(t, "test.png")
	img2 := mustOpenExampleImage(t, "ship.jpg")

	results, err := engine.PredictBatch([]image.Image{img1, img2})
	if err != nil {
		t.Fatalf("batch prediction failed: %v", err)
	}

	for i, batch := range results {
		fmt.Printf("Image %d: detected %d objects\n", i, len(batch))
		for _, res := range batch {
			fmt.Printf("  Class: %d, Score: %.2f, Box: %v\n", res.ClassID, res.Score, res.Box)
		}
	}
}
