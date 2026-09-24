package anomaly

import (
	"errors"
	"image"
	"image/color"
	"testing"
)

func contextConfig(t *testing.T, cameraGate, lightingGate, environmentGate AnomalyContextGate) Config {
	cfg := loadFixtureConfig(t, "efficientad-s")
	cfg.ContextPolicy.CameraID = cameraGate
	cfg.ContextPolicy.LightingProfile = lightingGate
	cfg.ContextPolicy.Environment = environmentGate
	cfg.CalibrationContext.ContextPolicy = cfg.ContextPolicy
	return cfg
}

func validRuntimeContext(cfg Config) AnomalyRuntimeContext {
	return AnomalyRuntimeContext{
		DeploymentBindingID:      "deployment-a",
		CameraID:                 cfg.CalibrationContext.CameraID,
		StreamID:                 "stream-1",
		CaptureResolution:        cfg.CalibrationContext.CaptureResolution,
		DecodedResolution:        cfg.CalibrationContext.DecodedResolution,
		ROIResolution:            cfg.CalibrationContext.DecodedResolution,
		ROIPolicyHash:            cfg.CalibrationContext.ROIPolicyHash,
		ROIPolicyVersion:         cfg.CalibrationContext.ROIPolicyVersion,
		LightingProfile:          cfg.CalibrationContext.LightingProfile,
		Environment:              cfg.CalibrationContext.Environment,
		FrameReceivedMonotonicNS: 100,
		DecodeMonotonicNS:        200,
	}
}

func contextImage(size image.Point) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, size.X, size.Y))
	for y := 0; y < size.Y; y++ {
		for x := 0; x < size.X; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 128, A: 255})
		}
	}
	return img
}

func TestValidateRuntimeContextAcceptsCanonicalFullFrame(t *testing.T) {
	cfg := contextConfig(t, ContextGateHard, ContextGateHard, ContextGateHard)
	ctx := validRuntimeContext(cfg)
	if observeOnly, err := validateRuntimeContext(contextImage(ctx.ROIResolution), ctx, cfg); err != nil || observeOnly {
		t.Fatalf("validateRuntimeContext() = %v, %v; want false, nil", observeOnly, err)
	}
}

func TestValidateRuntimeContextRejectsRequiredFieldsAndShape(t *testing.T) {
	cfg := contextConfig(t, ContextGateHard, ContextGateHard, ContextGateHard)
	img := contextImage(cfg.CalibrationContext.DecodedResolution)

	tests := []struct {
		name   string
		mutate func(*AnomalyRuntimeContext)
	}{
		{name: "nil image", mutate: func(*AnomalyRuntimeContext) {}},
		{name: "deployment", mutate: func(ctx *AnomalyRuntimeContext) { ctx.DeploymentBindingID = "" }},
		{name: "stream", mutate: func(ctx *AnomalyRuntimeContext) { ctx.StreamID = "" }},
		{name: "frame clock", mutate: func(ctx *AnomalyRuntimeContext) { ctx.FrameReceivedMonotonicNS = 0 }},
		{name: "decode clock", mutate: func(ctx *AnomalyRuntimeContext) { ctx.DecodeMonotonicNS = 0 }},
		{name: "ROI resolution", mutate: func(ctx *AnomalyRuntimeContext) { ctx.ROIResolution = image.Point{X: 1, Y: 1} }},
		{name: "decoded/ROI", mutate: func(ctx *AnomalyRuntimeContext) { ctx.DecodedResolution = image.Point{X: 100, Y: 100} }},
		{name: "ROI version", mutate: func(ctx *AnomalyRuntimeContext) { ctx.ROIPolicyVersion = "wrong" }},
		{name: "ROI hash", mutate: func(ctx *AnomalyRuntimeContext) { ctx.ROIPolicyHash = "wrong" }},
		{name: "capture resolution", mutate: func(ctx *AnomalyRuntimeContext) { ctx.CaptureResolution = image.Point{X: 100, Y: 100} }},
		{name: "decoded calibration", mutate: func(ctx *AnomalyRuntimeContext) {
			ctx.DecodedResolution = image.Point{X: 100, Y: 100}
			ctx.ROIResolution = ctx.DecodedResolution
		}},
		{name: "camera", mutate: func(ctx *AnomalyRuntimeContext) { ctx.CameraID = "wrong" }},
		{name: "lighting", mutate: func(ctx *AnomalyRuntimeContext) { ctx.LightingProfile = "wrong" }},
		{name: "environment", mutate: func(ctx *AnomalyRuntimeContext) { ctx.Environment = "wrong" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := validRuntimeContext(cfg)
			test.mutate(&ctx)
			input := img
			if test.name == "nil image" {
				input = nil
			}
			_, err := validateRuntimeContext(input, ctx, cfg)
			if !errors.Is(err, ErrRuntimeContextMismatch) {
				t.Fatalf("validateRuntimeContext() error = %v, want ErrRuntimeContextMismatch", err)
			}
		})
	}
}

func TestValidateRuntimeContextMarksObserveOnlyContexts(t *testing.T) {
	cfg := contextConfig(t, ContextGateObserve, ContextGateObserve, ContextGateObserve)
	ctx := validRuntimeContext(cfg)
	ctx.CameraID = "different-camera"
	ctx.LightingProfile = "different-light"
	ctx.Environment = "different-environment"
	observeOnly, err := validateRuntimeContext(contextImage(ctx.ROIResolution), ctx, cfg)
	if err != nil {
		t.Fatalf("validateRuntimeContext() error = %v", err)
	}
	if !observeOnly {
		t.Fatal("context mismatches did not mark result observe-only")
	}
}

func TestValidateRuntimeContextHardBlocksObservePolicyFields(t *testing.T) {
	cfg := contextConfig(t, ContextGateObserve, ContextGateObserve, ContextGateHard)
	ctx := validRuntimeContext(cfg)
	ctx.Environment = "different-environment"
	_, err := validateRuntimeContext(contextImage(ctx.ROIResolution), ctx, cfg)
	if !errors.Is(err, ErrRuntimeContextMismatch) {
		t.Fatalf("hard environment mismatch error = %v", err)
	}
}

func TestValidateRuntimeContextRequiresImageBounds(t *testing.T) {
	cfg := contextConfig(t, ContextGateHard, ContextGateHard, ContextGateHard)
	ctx := validRuntimeContext(cfg)
	ctx.ROIResolution = image.Point{X: 10, Y: 10}
	if _, err := validateRuntimeContext(contextImage(cfg.CalibrationContext.DecodedResolution), ctx, cfg); !errors.Is(err, ErrRuntimeContextMismatch) {
		t.Fatalf("bounds mismatch error = %v", err)
	}
}
