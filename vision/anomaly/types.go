// Package anomaly contains the frozen single-frame anomaly runtime contract.
//
// The package consumes production RMP directories produced by
// anomaly-export-v1 and validates the manifest, model card, threshold report,
// ONNX graph and runtime context before inference.
package anomaly

import "image"

// Region is one connected anomaly candidate derived from the model map.
//
// Regions are not object instances and must not be assigned cross-frame
// object identity. Coordinates are expressed in the effective ROI image
// before raven-go maps them to the full frame.
type Region struct {
	Box       image.Rectangle
	PeakScore float32
	MeanScore float32
	Area      int
	Coverage  float32
}

// AnomalyRuntimeContext describes the deployment and per-frame context.
//
// CaptureResolution is the source-declared resolution, DecodedResolution is
// the decoder output, and ROIResolution is the cropped image actually sent
// to inference. ModelInputSize is intentionally absent because it is derived
// from the ONNX graph, not repeated here.
type AnomalyRuntimeContext struct {
	DeploymentBindingID string
	CameraID            string
	StreamID            string
	CaptureResolution   image.Point
	DecodedResolution   image.Point
	ROIResolution       image.Point
	ROIPolicyHash       string
	ROIPolicyVersion    string
	LightingProfile     string
	Environment         string

	// FrameReceivedMonotonicNS and DecodeMonotonicNS are Raven process
	// monotonic-clock values. Zero is rejected so queue wait and decode delay
	// cannot silently degrade to an unmeasured duration.
	FrameReceivedMonotonicNS uint64
	DecodeMonotonicNS        uint64
}

// AnomalyContextGate controls whether a context mismatch blocks inference.
type AnomalyContextGate string

const (
	// ContextGateHard blocks inference when context does not match.
	ContextGateHard AnomalyContextGate = "hard_gate"
	// ContextGateObserve allows inference but marks results observe-only.
	ContextGateObserve AnomalyContextGate = "observe"
)

// AnomalyContextPolicy mirrors the RMP context_policy.
type AnomalyContextPolicy struct {
	CameraID          AnomalyContextGate
	CaptureResolution AnomalyContextGate
	DecodedResolution AnomalyContextGate
	ROIPolicy         AnomalyContextGate
	LightingProfile   AnomalyContextGate
	Environment       AnomalyContextGate
}

// InputLimitsConfig bounds graph and runtime tensor dimensions.
type InputLimitsConfig struct {
	MinSide   int
	MaxSide   int
	MaxPixels int
	MaxBatch  int
}

// OutputContractConfig is the strict ONNX tensor-name and shape contract.
type OutputContractConfig struct {
	SchemaVersion          string
	InputName              string
	InputShape             []any
	Layout                 string
	RequiredOutputs        []string
	ForbiddenOutputs       []string
	OutputCount            int
	OutputNameStrict       bool
	AllowPositionalOutputs bool
}

// ScoreAssertionConfig is the allclose assertion used for map/score checks.
type ScoreAssertionConfig struct {
	Enabled bool
	Atol    float64
	Rtol    float64
}

// ScoreContractConfig freezes the score and anomaly-map value space.
type ScoreContractConfig struct {
	SchemaVersion        string
	Semantics            ScoreSemantics
	Space                string
	ThresholdSpace       string
	PixelMapSpace        string
	RuntimeNormalization string
	Source               string
	PreProcessorInGraph  bool
	PostProcessorInGraph bool
	Assertion            ScoreAssertionConfig
}

// PreprocessContractConfig freezes out-of-graph image preparation.
type PreprocessContractConfig struct {
	ColorSpace          string
	TensorLayout        string
	Normalize           string
	ResizeKind          string
	InputSize           image.Point
	ScaleToUnitInterval bool
	InGraph             bool
}

// RuntimeProfileConfig contains the single-frame execution profile.
type RuntimeProfileConfig struct {
	BatchMode                       string
	MaxRuntimeBatch                 int
	ExecutionProviderFallbackMetric string
	ExecutionProviderPolicy         map[string]string
	InputLimits                     InputLimitsConfig
}

// CalibrationContextConfig stores the immutable context used at calibration.
type CalibrationContextConfig struct {
	CameraID          string
	CameraSerial      string
	CaptureResolution image.Point
	DecodedResolution image.Point
	ROIPolicyVersion  string
	ROIPolicyHash     string
	ROIPolicyValue    []float64
	LightingProfile   string
	Environment       string
	ContextPolicy     AnomalyContextPolicy
}

// Config is the complete frozen engine configuration surface.
type Config struct {
	ModelPath          string
	OnnxRuntimeLibPath string
	UseCuda            bool
	UseCoreML          bool
	NumThreads         int
	EnableCpuMemArena  bool

	ModelKind                       string
	ModelVersion                    string
	ModelSpec                       string
	InputSize                       image.Point
	ScoreThreshold                  float32
	ScoreSemantics                  ScoreSemantics
	OutputContract                  OutputContractConfig
	ScoreContract                   ScoreContractConfig
	PreprocessContract              PreprocessContractConfig
	MapThreshold                    float32
	MinRegionArea                   int
	MaxRegions                      int
	OutputAnomalyMap                bool
	MapDownsample                   int
	RuntimeBatchMode                string
	RuntimeMaxBatch                 int
	InputLimits                     InputLimitsConfig
	ExecutionProviderFallbackMetric string
	ExecutionProviderPolicy         map[string]string
	ExportProfile                   string
	DeploymentProfile               string
	ValidatedVariant                string
	RegionDecisionMode              string
	RegionPolicyVersion             string
	DecisionPolicyVersion           string
	ThresholdVersion                string
	ContextPolicy                   AnomalyContextPolicy
	CalibrationContext              CalibrationContextConfig

	ApiVersion int
}

// ScoreSemantics identifies the meaning of the raw model score.
type ScoreSemantics string

const (
	// ScoreSemanticsMapMax means pred_score equals max(anomaly_map).
	ScoreSemanticsMapMax ScoreSemantics = "map_max"
	// ScoreSemanticsModelDefined is reserved for a new explicit contract.
	ScoreSemanticsModelDefined ScoreSemantics = "model_defined"
)

// AnomalyResult is the single-frame anomaly result.
//
// MapWidth and MapHeight describe the actual AnomalyMap array. When
// MapDownsample is greater than one, MapSourceWidth and MapSourceHeight still
// describe the original ONNX map used by region policy.
type AnomalyResult struct {
	Score                 float32
	ScoreSemantics        ScoreSemantics
	IsAnomalous           bool
	ObserveOnly           bool
	RegionDecisionMode    string
	MapWidth              int
	MapHeight             int
	MapSourceWidth        int
	MapSourceHeight       int
	MapDownsample         int
	AnomalyMap            []float32
	Regions               []Region
	DebugStats            AnomalyDebugStats
	ThresholdVersion      string
	RegionPolicyVersion   string
	DecisionPolicyVersion string
}

// RegionQualificationStats is qualification-only region telemetry.
type RegionQualificationStats struct {
	RegionIndex   int
	ScoreP90      float32
	ScoreP95      float32
	SampledScores []float32
}

// AnomalyDebugStats is qualification/debug telemetry and is not part of the
// production event payload.
type AnomalyDebugStats struct {
	RawComponentCount     int
	ThresholdedPixelCount int
	FilteredByArea        int
	FilteredByMaxRegions  int
	Regions               []RegionQualificationStats
}

// AnomalyPredictor is the ONNX runtime boundary for single-frame anomaly
// inference. It is intentionally independent of vision.Predictor.
type AnomalyPredictor interface {
	PredictAnomaly(img image.Image, ctx AnomalyRuntimeContext) (*AnomalyResult, error)
	Destroy()
}

// AnomalyBatchItem is one input in the optional batch interface.
type AnomalyBatchItem struct {
	Image image.Image
	Ctx   AnomalyRuntimeContext
}

// BatchAnomalyPredictor is a P2-only optional interface. It is declared for
// contract completeness and must not be implemented by the v1 engine.
type BatchAnomalyPredictor interface {
	PredictAnomalyBatch(items []AnomalyBatchItem) ([]*AnomalyResult, error)
	MaxBatch() int
}
