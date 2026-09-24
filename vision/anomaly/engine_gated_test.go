package anomaly

import (
	"errors"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func gatedLibraryPath(t *testing.T) string {
	t.Helper()
	if path := os.Getenv("RAVEN_ORT_LIB_PATH"); path != "" {
		return path
	}
	path := filepath.Join("..", "..", "lib", "onnxruntime.dll")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("ONNX Runtime library is unavailable: %v", err)
	}
	return path
}

func gatedFixtureConfig(t *testing.T, variant string) Config {
	t.Helper()
	cfg := loadFixtureConfig(t, variant)
	cfg.OnnxRuntimeLibPath = gatedLibraryPath(t)
	// Keep the contract smoke deterministic on CPU even when a GPU EP is
	// installed on the development machine.
	cfg.ExecutionProviderPolicy = map[string]string{"cpu": "required"}
	return cfg
}

func gatedRuntimeContext(cfg Config) AnomalyRuntimeContext {
	return AnomalyRuntimeContext{
		DeploymentBindingID:      "gated-test",
		CameraID:                 cfg.CalibrationContext.CameraID,
		StreamID:                 "gated-stream",
		CaptureResolution:        cfg.CalibrationContext.CaptureResolution,
		DecodedResolution:        cfg.InputSize,
		ROIResolution:            cfg.InputSize,
		ROIPolicyHash:            cfg.CalibrationContext.ROIPolicyHash,
		ROIPolicyVersion:         cfg.CalibrationContext.ROIPolicyVersion,
		LightingProfile:          cfg.CalibrationContext.LightingProfile,
		Environment:              cfg.CalibrationContext.Environment,
		FrameReceivedMonotonicNS: 100,
		DecodeMonotonicNS:        200,
	}
}

func gradientImage(size image.Point) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, size.X, size.Y))
	for y := 0; y < size.Y; y++ {
		for x := 0; x < size.X; x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8((x*255)/(size.X-1) + 1),
				G: uint8((y*255)/(size.Y-1) + 1),
				B: 128,
				A: 255,
			})
		}
	}
	return img
}

func TestAnomalyEngineLoadsProductionGraphs(t *testing.T) {
	for _, variant := range []string{"efficientad-s", "padim-r18"} {
		t.Run(variant, func(t *testing.T) {
			cfg := gatedFixtureConfig(t, variant)
			engine, err := NewEngine(cfg)
			if err != nil {
				t.Fatalf("NewEngine() error = %v", err)
			}
			engine.Destroy()
		})
	}
}

func TestAnomalyEnginePredictsSingleFrame(t *testing.T) {
	cfg := gatedFixtureConfig(t, "padim-r18")
	engine, err := NewEngine(cfg)
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}
	defer engine.Destroy()

	result, err := engine.PredictAnomaly(gradientImage(cfg.InputSize), gatedRuntimeContext(cfg))
	if err != nil {
		t.Fatalf("PredictAnomaly() error = %v", err)
	}
	if result == nil {
		t.Fatal("PredictAnomaly returned nil result")
	}
	if !isFiniteFloat32(result.Score) {
		t.Fatalf("non-finite score %v", result.Score)
	}
	if result.ScoreSemantics != ScoreSemanticsMapMax {
		t.Fatalf("ScoreSemantics = %q, want %q", result.ScoreSemantics, ScoreSemanticsMapMax)
	}
	if result.MapSourceWidth != cfg.InputSize.X || result.MapSourceHeight != cfg.InputSize.Y {
		t.Fatalf("source map size = %dx%d, want %v", result.MapSourceWidth, result.MapSourceHeight, cfg.InputSize)
	}
	if result.AnomalyMap != nil || result.MapWidth != 0 || result.MapHeight != 0 {
		t.Fatalf("production result unexpectedly returned anomaly map: %+v", result)
	}
	if result.RegionDecisionMode != "independent" || result.Regions == nil {
		t.Fatalf("unexpected region decision: %+v", result)
	}
	for _, region := range result.Regions {
		if region.Box.Empty() {
			t.Fatalf("empty region box: %+v", region)
		}
		if !region.Box.In(image.Rect(0, 0, cfg.InputSize.X, cfg.InputSize.Y)) {
			t.Fatalf("region outside source image: %+v", region)
		}
	}
}

func TestAnomalyEngineConcurrentPredictThenDestroy(t *testing.T) {
	cfg := gatedFixtureConfig(t, "efficientad-s")
	engine, err := NewEngine(cfg)
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}

	const routines = 4
	results := make([]*AnomalyResult, routines)
	errorsOut := make([]error, routines)
	var waitGroup sync.WaitGroup
	for index := 0; index < routines; index++ {
		waitGroup.Add(1)
		go func(index int) {
			defer waitGroup.Done()
			results[index], errorsOut[index] = engine.PredictAnomaly(gradientImage(cfg.InputSize), gatedRuntimeContext(cfg))
		}(index)
	}
	waitGroup.Wait()

	for index := range results {
		if errorsOut[index] != nil {
			t.Fatalf("concurrent PredictAnomaly[%d] error = %v", index, errorsOut[index])
		}
		if results[index] == nil || !isFiniteFloat32(results[index].Score) {
			t.Fatalf("concurrent PredictAnomaly[%d] returned invalid result", index)
		}
		if math.IsNaN(float64(results[index].Score)) {
			t.Fatalf("concurrent PredictAnomaly[%d] score is NaN", index)
		}
	}
	engine.Destroy()
}

func TestAnomalyEnginePredictAfterDestroyFailsFast(t *testing.T) {
	cfg := gatedFixtureConfig(t, "efficientad-s")
	engine, err := NewEngine(cfg)
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}
	engine.Destroy()
	if _, err := engine.PredictAnomaly(gradientImage(cfg.InputSize), gatedRuntimeContext(cfg)); !errors.Is(err, ErrEngineDestroyed) {
		t.Fatalf("PredictAnomaly after Destroy error = %v, want ErrEngineDestroyed", err)
	}
	engine.Destroy()
}

func TestAnomalyEngineRejectsRuntimeContextBeforeRun(t *testing.T) {
	cfg := gatedFixtureConfig(t, "efficientad-s")
	engine, err := NewEngine(cfg)
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}
	defer engine.Destroy()
	ctx := gatedRuntimeContext(cfg)
	ctx.ROIPolicyHash = "wrong"
	if _, err := engine.PredictAnomaly(gradientImage(cfg.InputSize), ctx); !errors.Is(err, ErrRuntimeContextMismatch) {
		t.Fatalf("wrong ROI hash error = %v, want ErrRuntimeContextMismatch", err)
	}
}
