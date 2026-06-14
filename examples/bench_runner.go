package examples

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/up-zero/gotool/imageutil"
)

// BenchConfig controls benchmark behavior.
type BenchConfig struct {
	WarmupRuns int // number of warmup iterations (default 3)
	BenchRuns  int // number of benchmark iterations (default 30)
	UseCuda    bool
	LibPath    string // ONNX Runtime library path
}

// DefaultBenchConfig returns sensible defaults.
func DefaultBenchConfig() BenchConfig {
	return BenchConfig{
		WarmupRuns: 3,
		BenchRuns:  30,
		UseCuda:    false,
		LibPath:    "../lib/onnxruntime.dll",
	}
}

// --- Image Benchmark ---

// RunImageBench benchmarks a single engine against one image.
func RunImageBench(engine BenchEngine, img image.Image, cfg BenchConfig) BenchSummary {
	// Warmup
	for i := 0; i < cfg.WarmupRuns; i++ {
		engine.Predict(img)
	}

	latencies := make([]time.Duration, 0, cfg.BenchRuns)
	detCounts := make([]int, 0, cfg.BenchRuns)

	for i := 0; i < cfg.BenchRuns; i++ {
		start := time.Now()
		n, _ := engine.Predict(img)
		elapsed := time.Since(start)
		latencies = append(latencies, elapsed)
		detCounts = append(detCounts, n)
	}

	return ComputeSummary(engine.Name(), engine.Task(), latencies, detCounts)
}

// RunImageBenchMulti benchmarks multiple engines against one image.
func RunImageBenchMulti(engines []BenchEngine, img image.Image, cfg BenchConfig) []BenchSummary {
	summaries := make([]BenchSummary, 0, len(engines))
	for _, e := range engines {
		summary := RunImageBench(e, img, cfg)
		summaries = append(summaries, summary)
	}
	return summaries
}

// RunImageDirBench benchmarks engines against all images in a directory.
func RunImageDirBench(engines []BenchEngine, dir string, cfg BenchConfig) ([]BenchSummary, error) {
	imagePaths, err := loadImagesFromDir(dir)
	if err != nil {
		return nil, err
	}
	if len(imagePaths) == 0 {
		return nil, fmt.Errorf("no image files found in %s", dir)
	}

	// Collect all latencies per engine across all images
	engineLatencies := make([][]time.Duration, len(engines))
	engineDetCounts := make([][]int, len(engines))

	for _, imgPath := range imagePaths {
		img, err := imageutil.Open(imgPath)
		if err != nil {
			continue
		}

		for i, e := range engines {
			// Warmup per image
			for w := 0; w < cfg.WarmupRuns; w++ {
				e.Predict(img)
			}

			for r := 0; r < cfg.BenchRuns; r++ {
				start := time.Now()
				n, _ := e.Predict(img)
				engineLatencies[i] = append(engineLatencies[i], time.Since(start))
				engineDetCounts[i] = append(engineDetCounts[i], n)
			}
		}
	}

	summaries := make([]BenchSummary, 0, len(engines))
	for i, e := range engines {
		s := ComputeSummary(e.Name(), e.Task(), engineLatencies[i], engineDetCounts[i])
		summaries = append(summaries, s)
	}
	return summaries, nil
}

// --- Batch Benchmark ---

// RunBatchBench benchmarks batch inference for engines that support it.
func RunBatchBench(engine BenchEngine, imgs []image.Image, cfg BenchConfig) BenchSummary {
	if !engine.SupportsBatch() {
		return BenchSummary{EngineName: engine.Name(), Task: engine.Task()}
	}

	// Warmup
	for i := 0; i < cfg.WarmupRuns; i++ {
		engine.PredictBatch(imgs)
	}

	latencies := make([]time.Duration, 0, cfg.BenchRuns)
	detCounts := make([]int, 0, cfg.BenchRuns)

	for i := 0; i < cfg.BenchRuns; i++ {
		start := time.Now()
		n, _ := engine.PredictBatch(imgs)
		elapsed := time.Since(start)
		latencies = append(latencies, elapsed)
		detCounts = append(detCounts, n)
	}

	s := ComputeSummary(engine.Name(), engine.Task(), latencies, detCounts)
	return s
}

// --- Video Benchmark ---

// checkFFmpeg verifies ffmpeg is available.
func checkFFmpeg(t *testing.T) {
	t.Helper()
	_, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not found in PATH, skipping video test")
	}
}

// getVideoInfo extracts width, height, fps from a video file.
func getVideoInfo(videoPath string) (int, int, float64, error) {
	cmd := exec.Command("ffmpeg", "-i", videoPath, "-hide_banner")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	cmd.Run()

	output := stderr.String()
	var width, height int
	var fps float64

	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, "Video:") {
			continue
		}
		parts := strings.Split(line, ",")
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if strings.Contains(part, "x") && !strings.Contains(part, "fps") {
				dimParts := strings.Split(part, "x")
				if len(dimParts) == 2 {
					w, err1 := strconv.Atoi(strings.TrimSpace(dimParts[0]))
					hStr := dimParts[1]
					for i, c := range hStr {
						if c < '0' || c > '9' {
							hStr = hStr[:i]
							break
						}
					}
					h, err2 := strconv.Atoi(strings.TrimSpace(hStr))
					if err1 == nil && err2 == nil {
						width, height = w, h
					}
				}
			}
			if strings.Contains(part, "fps") {
				fpsStr := strings.TrimSuffix(part, " fps")
				fpsStr = strings.TrimSpace(fpsStr)
				if f, err := strconv.ParseFloat(fpsStr, 64); err == nil {
					fps = f
				}
			}
		}
	}

	if width == 0 || height == 0 || fps == 0 {
		return 0, 0, 0, fmt.Errorf("failed to parse video info")
	}
	return width, height, fps, nil
}

// extractFrames extracts frames from a video to a directory.
func extractFrames(t *testing.T, videoPath, outputDir string, sampleFPS float64) []string {
	t.Helper()
	pattern := filepath.Join(outputDir, "frame_%06d.jpg")
	args := []string{"-i", videoPath, "-q:v", "2"}
	if sampleFPS > 0 {
		args = append(args, "-r", fmt.Sprintf("%.1f", sampleFPS))
	}
	args = append(args, pattern)

	cmd := exec.Command("ffmpeg", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("ffmpeg frame extraction failed: %v\n%s", err, stderr.String())
	}

	var frames []string
	entries, err := os.ReadDir(outputDir)
	if err != nil {
		t.Fatalf("failed to read extracted frames: %v", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "frame_") && strings.HasSuffix(entry.Name(), ".jpg") {
			frames = append(frames, filepath.Join(outputDir, entry.Name()))
		}
	}
	sort.Strings(frames)
	return frames
}

// findVideoFile finds a video file in the given directory.
func findVideoFile(dir string) (string, error) {
	videoExts := map[string]bool{
		".mp4": true, ".avi": true, ".mkv": true, ".mov": true,
		".wmv": true, ".flv": true, ".webm": true, ".m4v": true,
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if videoExts[ext] {
			return filepath.Join(dir, entry.Name()), nil
		}
	}
	return "", fmt.Errorf("no video file found in %s", dir)
}

// RunVideoBench benchmarks engines against video frames.
func RunVideoBench(t *testing.T, engines []BenchEngine, videoPath string, sampleFPS float64) []BenchSummary {
	t.Helper()

	tempDir, err := os.MkdirTemp("", "raven_bench_video_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	extractDir := filepath.Join(tempDir, "frames")
	os.MkdirAll(extractDir, 0o755)

	frames := extractFrames(t, videoPath, extractDir, sampleFPS)
	if len(frames) == 0 {
		t.Fatal("no frames extracted from video")
	}
	t.Logf("Extracted %d frames", len(frames))

	engineLatencies := make([][]time.Duration, len(engines))
	engineDetCounts := make([][]int, len(engines))

	for _, framePath := range frames {
		img, err := imageutil.Open(framePath)
		if err != nil {
			continue
		}

		for i, e := range engines {
			start := time.Now()
			n, err := e.Predict(img)
			if err != nil {
				t.Fatalf("engine %s prediction failed: %v", e.Name(), err)
			}
			engineLatencies[i] = append(engineLatencies[i], time.Since(start))
			engineDetCounts[i] = append(engineDetCounts[i], n)
		}
	}

	summaries := make([]BenchSummary, 0, len(engines))
	for i, e := range engines {
		s := ComputeSummary(e.Name(), e.Task(), engineLatencies[i], engineDetCounts[i])
		summaries = append(summaries, s)
	}
	return summaries
}

// RunVideoStreamBench benchmarks engines against a video stream (pipe-based, no temp frames).
func RunVideoStreamBench(t *testing.T, engines []BenchEngine, videoPath string, streamFPS float64) []BenchSummary {
	t.Helper()

	extractCmd := exec.Command("ffmpeg",
		"-i", videoPath,
		"-f", "image2pipe",
		"-vcodec", "mjpeg",
		"-q:v", "5",
		"-r", fmt.Sprintf("%.1f", streamFPS),
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

	engineLatencies := make([][]time.Duration, len(engines))
	engineDetCounts := make([][]int, len(engines))

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

			for i, e := range engines {
				start := time.Now()
				n, _ := e.Predict(img)
				engineLatencies[i] = append(engineLatencies[i], time.Since(start))
				engineDetCounts[i] = append(engineDetCounts[i], n)
			}
		}
	}

	summaries := make([]BenchSummary, 0, len(engines))
	for i, e := range engines {
		s := ComputeSummary(e.Name(), e.Task(), engineLatencies[i], engineDetCounts[i])
		summaries = append(summaries, s)
	}
	return summaries
}

// --- Helper: load images from directory ---

var benchImageExts = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true, ".bmp": true, ".webp": true, ".tiff": true, ".tif": true,
}

func loadImagesFromDir(dir string) ([]string, error) {
	var files []string
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory %s: %w", dir, err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if benchImageExts[ext] {
			files = append(files, filepath.Join(dir, entry.Name()))
		}
	}
	return files, nil
}
