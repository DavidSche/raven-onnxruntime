package examples

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/DavidSche/raven-onnxruntime/vision/rfdetr"
	"github.com/DavidSche/raven-onnxruntime/vision/yolo26"
	"github.com/up-zero/gotool/imageutil"
)

type videoFrameResult struct {
	FrameIdx   int
	Yolo26Dets int
	RfdetrDets int
	Yolo26Ms   time.Duration
	RfdetrMs   time.Duration
}

func encodeVideo(t *testing.T, framesDir, outputPath string, fps float64) {
	t.Helper()

	pattern := filepath.Join(framesDir, "frame_%06d.jpg")
	args := []string{
		"-framerate", fmt.Sprintf("%.1f", fps),
		"-i", pattern,
		"-c:v", "libx264",
		"-pix_fmt", "yuv420p",
		"-crf", "23",
		"-y",
		outputPath,
	}

	cmd := exec.Command("ffmpeg", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Logf("ffmpeg encoding failed: %v\n%s", err, stderr.String())
	}
}

func drawLabeledDetResults(img image.Image, yoloResults []yolo26.DetResult, rfdetrResults []rfdetr.DetResult, label string) *image.RGBA {
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

	labelColor := color.RGBA{R: 255, G: 255, B: 255, A: 200}
	labelBg := color.RGBA{R: 0, G: 0, B: 0, A: 160}
	drawLabel(dst, label, 10, 10, labelColor, labelBg)

	return dst
}

func drawLabel(img *image.RGBA, text string, x, y int, fg, bg color.Color) {
	bounds := img.Bounds()
	textW := len(text) * 7
	textH := 14
	if x+textW > bounds.Dx() || y+textH > bounds.Dy() {
		return
	}
	for dy := 0; dy < textH; dy++ {
		for dx := 0; dx < textW; dx++ {
			img.Set(x+dx, y+dy, bg)
		}
	}
	for i, ch := range text {
		ox := x + i*7
		for dy := 0; dy < 10; dy++ {
			for dx := 0; dx < 6; dx++ {
				bit := uint(font5x7[byte(ch)][dy]) >> uint(7-dx) & 1
				if bit == 1 {
					img.Set(ox+dx, y+2+dy, fg)
				}
			}
		}
	}
}

var font5x7 = [256][10]byte{}

func initFont5x7() {
	font5x7['0'] = [10]byte{0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E, 0x00, 0x00, 0x00}
	font5x7['1'] = [10]byte{0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00, 0x00, 0x00}
	font5x7['2'] = [10]byte{0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F, 0x00, 0x00, 0x00}
	font5x7['3'] = [10]byte{0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E, 0x00, 0x00, 0x00}
	font5x7['4'] = [10]byte{0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02, 0x00, 0x00, 0x00}
	font5x7['5'] = [10]byte{0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E, 0x00, 0x00, 0x00}
	font5x7['6'] = [10]byte{0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E, 0x00, 0x00, 0x00}
	font5x7['7'] = [10]byte{0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00}
	font5x7['8'] = [10]byte{0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E, 0x00, 0x00, 0x00}
	font5x7['9'] = [10]byte{0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C, 0x00, 0x00, 0x00}
	font5x7[' '] = [10]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	font5x7[':'] = [10]byte{0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}
	font5x7['.'] = [10]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00}
	font5x7['-'] = [10]byte{0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	font5x7['Y'] = [10]byte{0x11, 0x11, 0x11, 0x0E, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00}
	font5x7['O'] = [10]byte{0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00, 0x00, 0x00}
	font5x7['L'] = [10]byte{0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1E, 0x00, 0x00, 0x00}
	font5x7['R'] = [10]byte{0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11, 0x00, 0x00, 0x00}
	font5x7['F'] = [10]byte{0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00}
	font5x7['D'] = [10]byte{0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E, 0x00, 0x00, 0x00}
	font5x7['E'] = [10]byte{0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F, 0x00, 0x00, 0x00}
	font5x7['T'] = [10]byte{0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00}
	font5x7['v'] = [10]byte{0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04, 0x00, 0x00, 0x00}
	font5x7['s'] = [10]byte{0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E, 0x00, 0x00, 0x00}
}

func saveJPEG(img image.Image, path string, quality int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, img, &jpeg.Options{Quality: quality})
}

func TestCompareYOLO26VsRFDETRVideo(t *testing.T) {
	initFont5x7()
	checkFFmpeg(t)

	videoPath := ""
	if v := os.Getenv("VIDEO_PATH"); v != "" {
		videoPath = v
	} else {
		found, err := findVideoFile(".")
		if err != nil {
			t.Skip("no video file found in examples directory. Set VIDEO_PATH env var or place a video file (.mp4/.avi/.mkv/.mov/.webm) in the examples directory")
		}
		videoPath = found
	}

	t.Logf("Using video: %s", videoPath)

	width, height, fps, err := getVideoInfo(videoPath)
	if err != nil {
		t.Fatalf("failed to get video info: %v", err)
	}
	t.Logf("Video info: %dx%d @ %.1ffps", width, height, fps)

	yoloCfg := yolo26.DefaultDetConfig()
	yoloCfg.ModelPath = ExampleModelPath("yolo26", "yolo26s.onnx")
	yoloCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	yoloCfg.UseCuda = true

	yoloEngine, err := yolo26.NewDetEngine(yoloCfg)
	if err != nil {
		t.Fatalf("failed to initialize YOLO26 engine: %v", err)
	}
	defer yoloEngine.Destroy()

	rfdetrCfg := rfdetr.DefaultDetConfig()
	rfdetrCfg.ModelPath = ExampleModelPath("rf-detr", "rf-detr-small.onnx")
	rfdetrCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	rfdetrCfg.UseCuda = true

	rfdetrEngine, err := rfdetr.NewDetEngine(rfdetrCfg)
	if err != nil {
		t.Fatalf("failed to initialize RF-DETR engine: %v", err)
	}
	defer rfdetrEngine.Destroy()

	tempDir, err := os.MkdirTemp("", "raven_video_compare_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	extractDir := filepath.Join(tempDir, "extracted")
	annotatedDir := filepath.Join(tempDir, "annotated")
	os.MkdirAll(extractDir, 0o755)
	os.MkdirAll(annotatedDir, 0o755)

	sampleFPS := 5.0
	if fps <= 5 {
		sampleFPS = 0
	}
	t.Logf("Extracting frames (sampleFPS=%.1f)...", sampleFPS)

	frames := extractFrames(t, videoPath, extractDir, sampleFPS)
	if len(frames) == 0 {
		t.Fatal("no frames extracted from video")
	}
	t.Logf("Extracted %d frames", len(frames))

	frameResults := make([]videoFrameResult, 0, len(frames))
	var totalYoloMs, totalRfdetrMs time.Duration
	var totalYoloDets, totalRfdetrDets int

	for i, framePath := range frames {
		img, err := imageutil.Open(framePath)
		if err != nil {
			t.Logf("skip frame %d: %v", i, err)
			continue
		}

		yoloStart := time.Now()
		yoloResults, err := yoloEngine.Predict(img)
		if err != nil {
			t.Fatalf("YOLO26 prediction failed on frame %d: %v", i, err)
		}
		yoloElapsed := time.Since(yoloStart)

		rfdetrStart := time.Now()
		rfdetrResults, err := rfdetrEngine.Predict(img)
		if err != nil {
			t.Fatalf("RF-DETR prediction failed on frame %d: %v", i, err)
		}
		rfdetrElapsed := time.Since(rfdetrStart)

		totalYoloMs += yoloElapsed
		totalRfdetrMs += rfdetrElapsed
		totalYoloDets += len(yoloResults)
		totalRfdetrDets += len(rfdetrResults)

		frameResults = append(frameResults, videoFrameResult{
			FrameIdx:   i,
			Yolo26Dets: len(yoloResults),
			RfdetrDets: len(rfdetrResults),
			Yolo26Ms:   yoloElapsed,
			RfdetrMs:   rfdetrElapsed,
		})

		label := fmt.Sprintf("F:%d YOLO:%d RF:%d", i, len(yoloResults), len(rfdetrResults))
		annotatedImg := drawLabeledDetResults(img, yoloResults, rfdetrResults, label)

		outPath := filepath.Join(annotatedDir, fmt.Sprintf("frame_%06d.jpg", i+1))
		if err := saveJPEG(annotatedImg, outPath, 85); err != nil {
			t.Logf("failed to save annotated frame %d: %v", i, err)
		}

		if (i+1)%10 == 0 || i == len(frames)-1 {
			t.Logf("Processed %d/%d frames", i+1, len(frames))
		}
	}

	videoBaseName := strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath))
	outputVideoPath := exampleArtifactPath("video", fmt.Sprintf("compare_%s.mp4", videoBaseName))

	actualFPS := sampleFPS
	if actualFPS <= 0 {
		actualFPS = fps
	}
	encodeVideo(t, annotatedDir, outputVideoPath, actualFPS)

	if _, err := os.Stat(outputVideoPath); err == nil {
		t.Logf("Output video saved: %s", outputVideoPath)
	}

	fmt.Println("====================================================")
	fmt.Println("  Video Comparison: YOLO26 vs RF-DETR")
	fmt.Println("====================================================")
	fmt.Printf("Video: %s (%dx%d @ %.1ffps)\n", filepath.Base(videoPath), width, height, fps)
	fmt.Printf("Frames processed: %d (sampled at %.1ffps)\n\n", len(frameResults), actualFPS)

	fmt.Printf("%-10s | %-12s | %-12s | %-12s | %-12s\n", "Frame", "YOLO#Det", "YOLO(ms)", "RFDETR#Det", "RFDETR(ms)")
	fmt.Println(strings.Repeat("-", 66))

	for _, fr := range frameResults {
		fmt.Printf("%-10d | %-12d | %-12v | %-12d | %-12v\n",
			fr.FrameIdx,
			fr.Yolo26Dets,
			fr.Yolo26Ms.Round(time.Microsecond),
			fr.RfdetrDets,
			fr.RfdetrMs.Round(time.Microsecond),
		)
	}

	fmt.Println(strings.Repeat("-", 66))
	n := len(frameResults)
	if n > 0 {
		yoloAvg := totalYoloMs / time.Duration(n)
		rfdetrAvg := totalRfdetrMs / time.Duration(n)
		fmt.Printf("%-10s | %-12d | %-12v | %-12d | %-12v\n",
			"AVG",
			totalYoloDets/n,
			yoloAvg.Round(time.Microsecond),
			totalRfdetrDets/n,
			rfdetrAvg.Round(time.Microsecond),
		)
		fmt.Println(strings.Repeat("-", 66))
		fmt.Printf("\nTotal YOLO26 detections: %d | Total RF-DETR detections: %d\n", totalYoloDets, totalRfdetrDets)
		fmt.Printf("Avg YOLO26 latency: %v | Avg RF-DETR latency: %v\n", yoloAvg.Round(time.Microsecond), rfdetrAvg.Round(time.Microsecond))
		fmt.Printf("YOLO26 FPS: %.1f | RF-DETR FPS: %.1f\n", 1.0/yoloAvg.Seconds(), 1.0/rfdetrAvg.Seconds())
	}

	fmt.Println("\nLegend: Green boxes = YOLO26, Blue boxes = RF-DETR")
}

func TestCompareYOLO26VsRFDETRVideoStream(t *testing.T) {
	initFont5x7()
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

	yoloCfg := yolo26.DefaultDetConfig()
	yoloCfg.ModelPath = ExampleModelPath("yolo26", "yolo26s.onnx")
	yoloCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	yoloCfg.UseCuda = true

	yoloEngine, err := yolo26.NewDetEngine(yoloCfg)
	if err != nil {
		t.Fatalf("failed to initialize YOLO26 engine: %v", err)
	}
	defer yoloEngine.Destroy()

	rfdetrCfg := rfdetr.DefaultDetConfig()
	rfdetrCfg.ModelPath = ExampleModelPath("rf-detr", "rf-detr-small.onnx")
	rfdetrCfg.OnnxRuntimeLibPath = ExampleORTLibraryPath()
	rfdetrCfg.UseCuda = true

	rfdetrEngine, err := rfdetr.NewDetEngine(rfdetrCfg)
	if err != nil {
		t.Fatalf("failed to initialize RF-DETR engine: %v", err)
	}
	defer rfdetrEngine.Destroy()

	tempDir, err := os.MkdirTemp("", "raven_video_stream_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	annotatedDir := filepath.Join(tempDir, "annotated")
	os.MkdirAll(annotatedDir, 0o755)

	extractCmd := exec.Command("ffmpeg",
		"-i", videoPath,
		"-f", "image2pipe",
		"-vcodec", "mjpeg",
		"-q:v", "5",
		"-r", "5",
		"-",
	)

	pipeR, err := extractCmd.StdoutPipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}
	var stderrBuf bytes.Buffer
	extractCmd.Stderr = &stderrBuf

	if err := extractCmd.Start(); err != nil {
		t.Fatalf("failed to start ffmpeg: %v", err)
	}
	defer func() {
		extractCmd.Process.Kill()
		extractCmd.Wait()
	}()

	frameIdx := 0
	var totalYoloMs, totalRfdetrMs time.Duration
	var totalYoloDets, totalRfdetrDets int
	buf := make([]byte, 0, 2*1024*1024)
	soi := []byte{0xFF, 0xD8}
	eoi := []byte{0xFF, 0xD9}

	chunk := make([]byte, 256*1024)
	for {
		n, err := pipeR.Read(chunk)
		if n > 0 {
			buf = append(buf, chunk[:n]...)
		}
		if err != nil {
			if err != io.EOF {
				t.Logf("pipe read error: %v", err)
			}
			break
		}

		for {
			startIdx := bytes.Index(buf, soi)
			if startIdx < 0 {
				if len(buf) > 4*1024*1024 {
					buf = buf[len(buf)-65536:]
				}
				break
			}

			endIdx := bytes.Index(buf[startIdx+2:], eoi)
			if endIdx < 0 {
				if len(buf) > 4*1024*1024 {
					buf = buf[len(buf)-65536:]
				}
				break
			}
			endIdx += startIdx + 2 + 2

			jpegData := buf[startIdx:endIdx]
			buf = buf[endIdx:]

			img, err := jpeg.Decode(bytes.NewReader(jpegData))
			if err != nil {
				continue
			}

			yoloStart := time.Now()
			yoloResults, err := yoloEngine.Predict(img)
			if err != nil {
				t.Fatalf("YOLO26 prediction failed on stream frame %d: %v", frameIdx, err)
			}
			yoloElapsed := time.Since(yoloStart)

			rfdetrStart := time.Now()
			rfdetrResults, err := rfdetrEngine.Predict(img)
			if err != nil {
				t.Fatalf("RF-DETR prediction failed on stream frame %d: %v", frameIdx, err)
			}
			rfdetrElapsed := time.Since(rfdetrStart)

			totalYoloMs += yoloElapsed
			totalRfdetrMs += rfdetrElapsed
			totalYoloDets += len(yoloResults)
			totalRfdetrDets += len(rfdetrResults)

			label := fmt.Sprintf("F:%d Y:%d R:%d", frameIdx, len(yoloResults), len(rfdetrResults))
			annotatedImg := drawLabeledDetResults(img, yoloResults, rfdetrResults, label)

			outPath := filepath.Join(annotatedDir, fmt.Sprintf("frame_%06d.jpg", frameIdx+1))
			saveJPEG(annotatedImg, outPath, 85)

			frameIdx++
			if frameIdx%10 == 0 {
				t.Logf("Stream: processed %d frames", frameIdx)
			}
		}
	}

	extractCmd.Wait()

	if frameIdx == 0 {
		t.Fatal("no frames decoded from video stream")
	}

	videoBaseName := strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath))
	outputVideoPath := exampleArtifactPath("video", fmt.Sprintf("compare_stream_%s.mp4", videoBaseName))
	encodeVideo(t, annotatedDir, outputVideoPath, 5.0)

	if _, err := os.Stat(outputVideoPath); err == nil {
		t.Logf("Output video saved: %s", outputVideoPath)
	}

	yoloAvg := totalYoloMs / time.Duration(frameIdx)
	rfdetrAvg := totalRfdetrMs / time.Duration(frameIdx)

	fmt.Println("====================================================")
	fmt.Println("  Video Stream Comparison: YOLO26 vs RF-DETR")
	fmt.Println("====================================================")
	fmt.Printf("Video: %s\n", filepath.Base(videoPath))
	fmt.Printf("Frames processed: %d (streaming at 5fps)\n\n", frameIdx)
	fmt.Printf("%-12s | %-12s | %-12s | %-12s\n", "Model", "Total Dets", "Avg Latency", "Avg FPS")
	fmt.Println(strings.Repeat("-", 56))
	fmt.Printf("%-12s | %-12d | %-12v | %-12.1f\n", "YOLO26", totalYoloDets, yoloAvg.Round(time.Microsecond), 1.0/yoloAvg.Seconds())
	fmt.Printf("%-12s | %-12d | %-12v | %-12.1f\n", "RF-DETR", totalRfdetrDets, rfdetrAvg.Round(time.Microsecond), 1.0/rfdetrAvg.Seconds())
	fmt.Println(strings.Repeat("-", 56))

	speedup := float64(rfdetrAvg) / float64(yoloAvg)
	if speedup >= 1.0 {
		fmt.Printf("\nYOLO26 is %.2fx faster than RF-DETR\n", speedup)
	} else {
		fmt.Printf("\nRF-DETR is %.2fx faster than YOLO26\n", 1.0/speedup)
	}

	fmt.Println("\nLegend: Green boxes = YOLO26, Blue boxes = RF-DETR")
}
