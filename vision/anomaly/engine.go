package anomaly

import (
	"fmt"
	"image"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/up-zero/gotool/convertutil"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

// Engine is the concurrent-safe single-frame anomaly inference engine.
//
// All fields other than the atomic counters are immutable after NewEngine
// returns. PredictAnomaly does not retain request-local state.
type Engine struct {
	session *ort.Session
	config  Config

	runCount        atomic.Uint64
	epFallbackTotal atomic.Uint64
	destroyed       atomic.Bool
	destroyOnce     sync.Once
}

// NewEngine validates cfg, initializes ONNX Runtime, creates the session and
// applies the frozen graph/output contract checks.
func NewEngine(cfg Config) (*Engine, error) {
	if err := validateRuntimeConfig(&cfg); err != nil {
		return nil, err
	}
	info, err := os.Stat(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("%w: cannot stat ONNX model %q: %v", ErrInvalidConfig, cfg.ModelPath, err)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("%w: model path is a directory: %q", ErrInvalidConfig, cfg.ModelPath)
	}

	onnxConfig := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, onnxConfig); err != nil {
		return nil, fmt.Errorf("failed to copy ONNX properties: %w", err)
	}
	// RMP EP policy wins over the legacy boolean compatibility fields.
	onnxConfig.UseCuda = false
	onnxConfig.UseCoreML = false
	if err := onnxConfig.New(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	sessionOptions, selectedProvider, epFallback, err := resolveExecutionProviders(onnxConfig, cfg)
	if err != nil {
		onnxConfig.Destroy()
		return nil, err
	}
	session, err := onnxConfig.OnnxEngine.NewSession(cfg.ModelPath, sessionOptions)
	if err != nil {
		sessionOptions.Destroy()
		onnxConfig.Destroy()
		return nil, fmt.Errorf("failed to create anomaly ONNX session: %w", err)
	}
	sessionOptions.Destroy()
	onnxConfig.Destroy()

	validatedConfig, err := validateONNXGraph(session, cfg)
	if err != nil {
		session.Destroy()
		return nil, err
	}

	ortlog.Infow("anomaly engine created",
		"model_path", cfg.ModelPath,
		"validated_variant", cfg.ValidatedVariant,
		"provider_selected", selectedProvider,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)
	engine := &Engine{session: session, config: validatedConfig}
	if epFallback {
		engine.epFallbackTotal.Store(1)
	}
	return engine, nil
}

func validateRuntimeConfig(cfg *Config) error {
	if cfg == nil {
		return fmt.Errorf("%w: configuration is nil", ErrInvalidConfig)
	}
	if cfg.ModelPath == "" || cfg.ModelSpec == "" {
		return fmt.Errorf("%w: model path and canonical model_spec are required", ErrInvalidConfig)
	}
	if cfg.InputSize.X <= 0 || cfg.InputSize.Y <= 0 {
		return fmt.Errorf("%w: ONNX input size must be positive", ErrInvalidConfig)
	}
	if !isFiniteFloat32(cfg.ScoreThreshold) || !isFiniteFloat32(cfg.MapThreshold) || cfg.ScoreThreshold < 0 || cfg.MapThreshold < 0 {
		return fmt.Errorf("%w: thresholds must be finite and non-negative", ErrInvalidConfig)
	}
	if cfg.MinRegionArea <= 0 || cfg.MaxRegions <= 0 {
		return fmt.Errorf("%w: region limits must be positive", ErrInvalidConfig)
	}
	if cfg.MapDownsample < 0 {
		return fmt.Errorf("%w: map_downsample must be zero or positive", ErrInvalidConfig)
	}
	if cfg.MapDownsample == 0 {
		cfg.MapDownsample = 1
	}
	if cfg.MapDownsample > 1 && !cfg.OutputAnomalyMap {
		return fmt.Errorf("%w: map_downsample requires output_anomaly_map", ErrInvalidConfig)
	}
	if cfg.RuntimeBatchMode != "single" || cfg.RuntimeMaxBatch != 1 {
		return fmt.Errorf("%w: anomaly-v1 permits single-frame runtime only", ErrInvalidConfig)
	}
	if cfg.InputLimits.MaxBatch < cfg.RuntimeMaxBatch {
		return fmt.Errorf("%w: runtime batch exceeds input limits", ErrInvalidConfig)
	}
	if cfg.InputSize.X < cfg.InputLimits.MinSide || cfg.InputSize.Y < cfg.InputLimits.MinSide ||
		cfg.InputSize.X > cfg.InputLimits.MaxSide || cfg.InputSize.Y > cfg.InputLimits.MaxSide ||
		int64(cfg.InputSize.X)*int64(cfg.InputSize.Y) > int64(cfg.InputLimits.MaxPixels) {
		return fmt.Errorf("%w: ONNX input exceeds input limits", ErrInvalidConfig)
	}
	if err := validateExecutionProviderPolicy(cfg.ExecutionProviderPolicy); err != nil {
		return err
	}
	if cfg.PreprocessContract.Normalize != normalizeDivide255 && cfg.PreprocessContract.Normalize != normalizeImageNet {
		return fmt.Errorf("%w: unsupported preprocess normalization %q", ErrInvalidConfig, cfg.PreprocessContract.Normalize)
	}
	if cfg.PreprocessContract.ResizeKind != "stretch" || cfg.PreprocessContract.TensorLayout != "NCHW" {
		return fmt.Errorf("%w: anomaly-v1 requires stretch resize and NCHW tensors", ErrInvalidConfig)
	}
	if cfg.ScoreSemantics != ScoreSemanticsMapMax && cfg.ScoreSemantics != ScoreSemanticsModelDefined {
		return fmt.Errorf("%w: unsupported score semantics %q", ErrInvalidConfig, cfg.ScoreSemantics)
	}
	if cfg.OutputContract.InputName == "" || len(cfg.OutputContract.RequiredOutputs) != 2 || !cfg.OutputContract.OutputNameStrict {
		return fmt.Errorf("%w: strict pred_score/anomaly_map output contract is required", ErrInvalidConfig)
	}
	for _, field := range []string{cfg.ThresholdVersion, cfg.RegionPolicyVersion, cfg.DecisionPolicyVersion, cfg.CalibrationContext.ROIPolicyHash} {
		if field == "" {
			return fmt.Errorf("%w: threshold, region, decision and ROI policy identities are required", ErrInvalidConfig)
		}
	}
	return nil
}

func resolveExecutionProviders(onnxConfig *vision.OnnxConfig, cfg Config) (*ort.SessionOptions, string, bool, error) {
	baseOptions := onnxConfig.SessionOptions
	if baseOptions == nil {
		return nil, "", false, fmt.Errorf("ONNX Runtime did not create base session options")
	}
	providers, err := onnxConfig.OnnxEngine.AvailableProviders()
	if err != nil {
		return nil, "", false, fmt.Errorf("failed to query ONNX Runtime providers: %w", err)
	}

	candidate := ""
	if cfg.ExecutionProviderPolicy["cuda"] == providerRequired || cfg.ExecutionProviderPolicy["cuda"] == providerPreferred {
		candidate = "cuda"
	} else if cfg.ExecutionProviderPolicy["coreml"] == providerRequired || cfg.ExecutionProviderPolicy["coreml"] == providerPreferred {
		candidate = "coreml"
	}
	if candidate == "" {
		return baseOptions, "cpu", false, nil
	}

	required := cfg.ExecutionProviderPolicy[candidate] == providerRequired
	providerName := "CUDAExecutionProvider"
	if candidate == "coreml" {
		providerName = "CoreMLExecutionProvider"
	}
	available := false
	for _, name := range providers {
		if name == providerName {
			available = true
			break
		}
	}
	if !available {
		if required {
			return nil, "", false, fmt.Errorf("%s execution provider is required but unavailable", providerName)
		}
		return baseOptions, "cpu", true, nil
	}

	options, err := buildExecutionProviderOptions(onnxConfig.OnnxEngine, cfg, candidate)
	if err != nil {
		if required {
			return nil, "", false, fmt.Errorf("required %s execution provider failed to initialize: %w", providerName, err)
		}
		ortlog.Warnw("preferred anomaly execution provider failed; falling back to CPU",
			"requested_provider", providerName,
			"effective_provider", "CPUExecutionProvider",
			"reason", err,
			"metric", cfg.ExecutionProviderFallbackMetric)
		return baseOptions, "cpu", true, nil
	}
	baseOptions.Destroy()
	return options, candidate, false, nil
}

// EPFallbackTotal reports the process-visible anomaly EP fallback count.
func (e *Engine) EPFallbackTotal() uint64 {
	if e == nil {
		return 0
	}
	return e.epFallbackTotal.Load()
}

func buildExecutionProviderOptions(engine *ort.Engine, cfg Config, provider string) (*ort.SessionOptions, error) {
	options, err := engine.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	destroyOnError := func(cause error) (*ort.SessionOptions, error) {
		options.Destroy()
		return nil, cause
	}
	if cfg.NumThreads > 0 {
		if err := options.SetIntraOpNumThreads(int32(cfg.NumThreads)); err != nil {
			return destroyOnError(fmt.Errorf("failed to set intra-op threads: %w", err))
		}
	}
	if err := options.SetInterOpNumThreads(1); err != nil {
		return destroyOnError(fmt.Errorf("failed to set inter-op threads: %w", err))
	}
	if err := options.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
		return destroyOnError(fmt.Errorf("failed to set sequential execution: %w", err))
	}
	if err := options.SetMemPattern(true); err != nil {
		return destroyOnError(fmt.Errorf("failed to enable memory pattern: %w", err))
	}
	if err := options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelAll); err != nil {
		return destroyOnError(fmt.Errorf("failed to set graph optimization: %w", err))
	}
	if err := options.SetCpuMemArena(cfg.EnableCpuMemArena); err != nil {
		return destroyOnError(fmt.Errorf("failed to set CPU memory arena: %w", err))
	}
	switch provider {
	case "cuda":
		if err := options.EnableCUDA(); err != nil {
			return destroyOnError(err)
		}
	case "coreml":
		if err := options.EnableCoreML(nil); err != nil {
			return destroyOnError(err)
		}
	default:
		return destroyOnError(fmt.Errorf("unsupported execution provider %q", provider))
	}
	return options, nil
}

func validateONNXGraph(session *ort.Session, cfg Config) (Config, error) {
	if len(session.InputNames) != 1 || session.InputNames[0] != cfg.OutputContract.InputName {
		return cfg, fmt.Errorf("%w: inputs %v, want exactly [%s]", ErrInvalidContract, session.InputNames, cfg.OutputContract.InputName)
	}
	if len(session.OutputNames) != cfg.OutputContract.OutputCount {
		return cfg, fmt.Errorf("%w: got %d outputs, want exactly %d", ErrInvalidContract, len(session.OutputNames), cfg.OutputContract.OutputCount)
	}
	for _, name := range cfg.OutputContract.ForbiddenOutputs {
		for _, output := range session.OutputNames {
			if output == name {
				return cfg, fmt.Errorf("%w: forbidden output %q is present", ErrInvalidContract, name)
			}
		}
	}
	for _, required := range cfg.OutputContract.RequiredOutputs {
		found := false
		for _, output := range session.OutputNames {
			if output == required {
				found = true
				break
			}
		}
		if !found {
			return cfg, fmt.Errorf("%w: required output %q is missing", ErrInvalidContract, required)
		}
	}

	shape, err := session.GetInputShape(0)
	if err != nil {
		return cfg, fmt.Errorf("failed to read ONNX input shape: %w", err)
	}
	if len(shape) != 4 {
		return cfg, fmt.Errorf("%w: input shape %v, want [B,3,H,W]", ErrInvalidContract, shape)
	}
	if !isDynamicDimension(shape[0]) || shape[1] != 3 || shape[2] < 1 || shape[3] < 1 {
		return cfg, fmt.Errorf("%w: input shape %v is not a dynamic batch RGB tensor", ErrInvalidContract, shape)
	}
	height, width := int(shape[2]), int(shape[3])
	if cfg.InputSize != (image.Point{X: width, Y: height}) {
		return cfg, fmt.Errorf("%w: manifest input size %v differs from ONNX input size %dx%d", ErrInvalidContract, cfg.InputSize, width, height)
	}
	cfg.InputSize = image.Point{X: width, Y: height}
	if height < cfg.InputLimits.MinSide || width < cfg.InputLimits.MinSide ||
		height > cfg.InputLimits.MaxSide || width > cfg.InputLimits.MaxSide ||
		int64(height)*int64(width) > int64(cfg.InputLimits.MaxPixels) {
		return cfg, fmt.Errorf("%w: ONNX input %dx%d exceeds input limits", ErrInvalidContract, width, height)
	}
	return cfg, nil
}

func isDynamicDimension(value int64) bool {
	return value <= 0
}

// PredictAnomaly runs one image. It is safe for concurrent use while the
// lifecycle owner has not called Destroy.
func (e *Engine) PredictAnomaly(img image.Image, runtimeCtx AnomalyRuntimeContext) (*AnomalyResult, error) {
	if e == nil {
		return nil, fmt.Errorf("%w: engine is not initialized", ErrInvalidConfig)
	}
	if e.destroyed.Load() {
		return nil, ErrEngineDestroyed
	}
	if e.session == nil {
		return nil, fmt.Errorf("%w: engine is not initialized", ErrInvalidConfig)
	}

	observeOnly, err := validateRuntimeContext(img, runtimeCtx, e.config)
	if err != nil {
		return nil, err
	}

	startedAt := time.Now()
	preprocessStart := time.Now()
	inputTensor, original, err := preprocessImage(e.session, img, e.config)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()
	preprocessDuration := time.Since(preprocessStart)

	runStart := time.Now()
	outputs, err := e.session.Run(map[string]*ort.Value{e.session.InputNames[0]: inputTensor})
	if err != nil {
		return nil, fmt.Errorf("anomaly inference failed: %w", err)
	}
	defer ort.DestroyValues(outputs)
	runDuration := time.Since(runStart)

	postprocessStart := time.Now()
	result, err := postprocess(outputs, e.config, original)
	if err != nil {
		return nil, err
	}
	result.ObserveOnly = observeOnly
	postprocessDuration := time.Since(postprocessStart)

	count := e.runCount.Add(1)
	if count%60 == 0 {
		ortlog.Infow("anomaly timings",
			"model_path", e.config.ModelPath,
			"preprocess", preprocessDuration.String(),
			"run", runDuration.String(),
			"postprocess", postprocessDuration.String(),
			"total", time.Since(startedAt).String(),
			"count", count)
	}
	return result, nil
}

func validateRuntimeContext(img image.Image, ctx AnomalyRuntimeContext, cfg Config) (bool, error) {
	if img == nil {
		return false, fmt.Errorf("%w: image is nil", ErrRuntimeContextMismatch)
	}
	if ctx.DeploymentBindingID == "" || ctx.StreamID == "" {
		return false, fmt.Errorf("%w: deployment binding and stream IDs are required", ErrRuntimeContextMismatch)
	}
	if ctx.FrameReceivedMonotonicNS == 0 || ctx.DecodeMonotonicNS == 0 {
		return false, fmt.Errorf("%w: frame received and decode monotonic timestamps are required", ErrRuntimeContextMismatch)
	}
	bounds := img.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return false, fmt.Errorf("%w: image has non-positive size", ErrRuntimeContextMismatch)
	}
	if ctx.ROIResolution != bounds.Size() {
		return false, fmt.Errorf("%w: ROI resolution %v differs from image bounds %v", ErrRuntimeContextMismatch, ctx.ROIResolution, bounds.Size())
	}
	if ctx.ROIResolution != ctx.DecodedResolution {
		return false, fmt.Errorf("%w: canonical full-frame context requires ROI resolution %v to equal decoded resolution %v", ErrRuntimeContextMismatch, ctx.ROIResolution, ctx.DecodedResolution)
	}
	if ctx.ROIPolicyVersion != cfg.CalibrationContext.ROIPolicyVersion {
		return false, fmt.Errorf("%w: ROI policy version mismatch: got %q, want %q", ErrRuntimeContextMismatch, ctx.ROIPolicyVersion, cfg.CalibrationContext.ROIPolicyVersion)
	}
	if ctx.ROIPolicyHash != cfg.CalibrationContext.ROIPolicyHash {
		return false, fmt.Errorf("%w: ROI policy hash mismatch", ErrRuntimeContextMismatch)
	}

	hardResolution := cfg.ContextPolicy.CaptureResolution == ContextGateHard
	if hardResolution && ctx.CaptureResolution != cfg.CalibrationContext.CaptureResolution {
		return false, fmt.Errorf("%w: capture resolution %v differs from calibrated %v", ErrRuntimeContextMismatch, ctx.CaptureResolution, cfg.CalibrationContext.CaptureResolution)
	}
	hardDecoded := cfg.ContextPolicy.DecodedResolution == ContextGateHard
	if hardDecoded && ctx.DecodedResolution != cfg.CalibrationContext.DecodedResolution {
		return false, fmt.Errorf("%w: decoded resolution %v differs from calibrated %v", ErrRuntimeContextMismatch, ctx.DecodedResolution, cfg.CalibrationContext.DecodedResolution)
	}

	observeOnly := false
	if cfg.ContextPolicy.CameraID == ContextGateHard {
		if ctx.CameraID != cfg.CalibrationContext.CameraID {
			return false, fmt.Errorf("%w: camera ID %q differs from calibrated %q", ErrRuntimeContextMismatch, ctx.CameraID, cfg.CalibrationContext.CameraID)
		}
	} else if ctx.CameraID != cfg.CalibrationContext.CameraID {
		observeOnly = true
	}
	if cfg.ContextPolicy.LightingProfile == ContextGateHard {
		if ctx.LightingProfile != cfg.CalibrationContext.LightingProfile {
			return false, fmt.Errorf("%w: lighting profile %q differs from calibrated %q", ErrRuntimeContextMismatch, ctx.LightingProfile, cfg.CalibrationContext.LightingProfile)
		}
	} else if ctx.LightingProfile != cfg.CalibrationContext.LightingProfile {
		observeOnly = true
	}
	if cfg.ContextPolicy.Environment == ContextGateHard {
		if ctx.Environment != cfg.CalibrationContext.Environment {
			return false, fmt.Errorf("%w: environment %q differs from calibrated %q", ErrRuntimeContextMismatch, ctx.Environment, cfg.CalibrationContext.Environment)
		}
	} else if ctx.Environment != cfg.CalibrationContext.Environment {
		observeOnly = true
	}
	return observeOnly, nil
}

// Destroy releases the ONNX session. It is a post-quiescence operation: the
// lifecycle owner must stop admission and wait for all PredictAnomaly calls
// before calling it.
func (e *Engine) Destroy() {
	if e == nil {
		return
	}
	e.destroyed.Store(true)
	e.destroyOnce.Do(func() {
		if e.session != nil {
			ortlog.Infow("destroying anomaly engine", "model_path", e.config.ModelPath)
			e.session.Destroy()
			e.session = nil
		}
	})
}
