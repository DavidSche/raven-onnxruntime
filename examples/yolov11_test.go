package examples

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/vision/yolov11"
	"github.com/up-zero/gotool/imageutil"
)

func TestYOLOv11Det(t *testing.T) {
	cfg := yolov11.DefaultDetConfig()
	cfg.ModelPath = ExampleModelPath("yolo11", "yolo11m.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolov11.NewDetEngine(cfg)
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
	if err := imageutil.Save(exampleArtifactPath("yolov11_det.jpg"), targetImg, 50); err != nil {
		t.Fatalf("failed to save image: %v", err)
	}
}

func TestYOLOv11Seg(t *testing.T) {
	cfg := yolov11.DefaultSegConfig()
	cfg.ModelPath = ExampleModelPath("yolo11", "yolo11m-seg.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolov11.NewSegEngine(cfg)
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
		if err := imageutil.Save(exampleArtifactPath(fmt.Sprintf("yolov11_seg_mask_%d.png", idx)), res.Mask, 100); err != nil {
			t.Fatalf("failed to save mask: %v", err)
		}
	}
}

func TestYOLOv11Cls(t *testing.T) {
	cfg := yolov11.DefaultClsConfig()
	cfg.ModelPath = ExampleModelPath("yolo11", "yolo11m-cls.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolov11.NewClsEngine(cfg)
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

func TestYOLOv11Pose(t *testing.T) {
	cfg := yolov11.DefaultPoseConfig()
	cfg.ModelPath = ExampleModelPath("yolo11", "yolo11m-pose.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolov11.NewPoseEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img := mustOpenExampleImage(t, "person.jpg")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	dst := yolov11.DrawPoseResult(img, results)
	if err := imageutil.Save(exampleArtifactPath("yolov11_pose.jpg"), dst, 50); err != nil {
		t.Fatalf("failed to save image: %v", err)
	}
}

func TestYOLOv11OBB(t *testing.T) {
	cfg := yolov11.DefaultOBBConfig()
	cfg.ModelPath = ExampleModelPath("yolo11", "yolo11m-obb.onnx")
	cfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()

	engine, err := yolov11.NewOBBEngine(cfg)
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
	if err := imageutil.Save(exampleArtifactPath("yolov11_obb.jpg"), dst, 50); err != nil {
		t.Fatalf("failed to save image: %v", err)
	}
}
