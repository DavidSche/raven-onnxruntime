package examples

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/vision/edgecrafter"
	"github.com/up-zero/gotool/imageutil"
)

func TestEdgeCrafterDet(t *testing.T) {
	cfg := edgecrafter.DefaultDetConfig()
	cfg.ModelPath = "../models/edgecrafter/ecdet-s.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := edgecrafter.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./test.png")
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
	err = imageutil.Save("edgecrafter_det.jpg", targetImg, 50)
	if err != nil {
		fmt.Printf("failed to save image: %v", err)
	}
}

func TestEdgeCrafterDetBatch(t *testing.T) {
	cfg := edgecrafter.DefaultDetConfig()
	cfg.ModelPath = "../models/edgecrafter/ecdet-s.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"
	cfg.DynamicBatch = true

	engine, err := edgecrafter.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img1, _ := imageutil.Open("./test.png")
	img2, _ := imageutil.Open("./ship.jpg")

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

func TestEdgeCrafterSeg(t *testing.T) {
	cfg := edgecrafter.DefaultSegConfig()
	cfg.ModelPath = "../models/edgecrafter/ecseg-s.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := edgecrafter.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("detected objects: %d\n", len(results))
	for idx, res := range results {
		fmt.Printf("Class: %d, Score: %.2f, Box: %v\n", res.ClassID, res.Score, res.Box)
		err := imageutil.Save(fmt.Sprintf("edgecrafter_seg_mask_%d.png", idx), res.Mask, 100)
		if err != nil {
			fmt.Printf("failed to save mask: %v", err)
		}
	}
}

func TestEdgeCrafterPose(t *testing.T) {
	cfg := edgecrafter.DefaultPoseConfig()
	cfg.ModelPath = "../models/edgecrafter/ecpose-s.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := edgecrafter.NewPoseEngine(cfg)
	if err != nil {
		t.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./person.jpg")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	fmt.Printf("detected poses: %d\n", len(results))
	for i, res := range results {
		fmt.Printf("Pose %d: ClassID=%d, Score=%.2f, Box=%v\n", i, res.ClassID, res.Score, res.Box)
		for j, kp := range res.KeyPoints {
			fmt.Printf("  Keypoint %d: X=%d, Y=%d, Score=%.2f\n", j, kp.X, kp.Y, kp.Score)
		}
	}

	// Draw pose results
	targetImg := image.NewRGBA(img.Bounds())
	draw.Draw(targetImg, img.Bounds(), img, img.Bounds().Min, draw.Src)
	for _, res := range results {
		imageutil.DrawThickRectOutline(targetImg, res.Box, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 2)
		for _, kp := range res.KeyPoints {
			if kp.X >= 0 && kp.X < targetImg.Bounds().Dx() && kp.Y >= 0 && kp.Y < targetImg.Bounds().Dy() {
				for dy := -3; dy <= 3; dy++ {
					for dx := -3; dx <= 3; dx++ {
						px := kp.X + dx
						py := kp.Y + dy
						if px >= 0 && px < targetImg.Bounds().Dx() && py >= 0 && py < targetImg.Bounds().Dy() {
							targetImg.Set(px, py, color.RGBA{R: 0, G: 255, B: 0, A: 255})
						}
					}
				}
			}
		}
	}
	err = imageutil.Save("edgecrafter_pose.jpg", targetImg, 50)
	if err != nil {
		fmt.Printf("failed to save image: %v", err)
	}
}
