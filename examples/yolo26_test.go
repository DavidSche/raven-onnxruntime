package examples

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/vision/yolo26"
	"github.com/up-zero/gotool/imageutil"
)

func TestYOLO26Det(t *testing.T) {
	cfg := yolo26.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("yolo26", "yolo26m.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolo26.NewDetEngine(cfg)
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
	err = imageutil.Save(exampleArtifactPath("yolo26_det.jpg"), targetImg, 50)
	if err != nil {
		fmt.Printf("failed to save image: %v", err)
	}
}

func TestYOLO26Seg(t *testing.T) {
	cfg := yolo26.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("yolo26", "yolo26s-seg.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolo26.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("detected objects: %d\n", len(results))
	for idx, res := range results {
		fmt.Printf("Class: %d, Score: %.2f, Box: %v\n", res.ClassID, res.Score, res.Box)
		err := imageutil.Save(exampleArtifactPath(fmt.Sprintf("yolo26_seg_mask_%d.png", idx)), res.Mask, 100)
		if err != nil {
			fmt.Printf("failed to save mask: %v", err)
		}
	}
}

func TestYOLO26Cls(t *testing.T) {
	cfg := yolo26.DefaultClsConfig()
	cfg.ModelPath = ExampleModelPath("yolo26", "yolo26m-cls.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolo26.NewClsEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "test.png")
	results, err := engine.Predict(img, 5)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	for _, res := range results {
		fmt.Printf("Class: %d, Score: %.5f\n", res.ClassID, res.Score)
	}
}

func TestYOLO26Pose(t *testing.T) {
	cfg := yolo26.DefaultPoseConfig()
	cfg.ModelPath = ExampleModelPath("yolo26", "yolo26m-pose.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolo26.NewPoseEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "person.jpg")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	dst := yolo26.DrawPoseResult(img, results)
	err = imageutil.Save(exampleArtifactPath("yolo26_pose.jpg"), dst, 50)
	if err != nil {
		fmt.Printf("failed to save image: %v", err)
	}
}

func TestYOLO26OBB(t *testing.T) {
	cfg := yolo26.DefaultOBBConfig()
	cfg.ModelPath = ExampleModelPath("yolo26", "yolo26m-obb.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolo26.NewOBBEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "ship.jpg")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	dst := image.NewRGBA(img.Bounds())
	draw.Draw(dst, img.Bounds(), img, img.Bounds().Min, draw.Src)
	for _, result := range results {
		imageutil.DrawThickPolygonOutline(dst, result.Corners[:], 3, color.RGBA{R: 255, G: 0, B: 0, A: 255})
	}
	err = imageutil.Save(exampleArtifactPath("yolo26_obb.jpg"), dst, 50)
	if err != nil {
		fmt.Printf("failed to save image: %v", err)
	}
}
