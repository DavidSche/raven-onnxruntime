package anomaly

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
)

const (
	manifestSchemaVersion       = "1.0"
	scoreContractSchemaVersion  = "rmp-score-contract-v1"
	outputContractSchemaVersion = "rmp-onnx-output-v1"
	thresholdReportSchema       = "anomaly-threshold-report-v1"
	modelCardSchema             = "anomaly-model-card-v1"
	dataSplitSchema             = "anomaly-data-split-v1"
	regionPolicyVersionV1       = "region-policy-v1"
	deploymentProduction        = "production"
	deploymentDevelopment       = "development"
	modelKindEfficientAD        = "efficient-ad"
	modelKindPadim              = "padim"
	normalizeDivide255          = "divide_255"
	normalizeImageNet           = "imagenet"
	providerRequired            = "required"
	providerPreferred           = "preferred"
)

// LoadConfig loads and deeply validates a production RMP directory. modelPath
// may point to the RMP directory or to any file inside it, including
// model.onnx. The returned Config is a detached copy suitable for NewEngine.
func LoadConfig(modelPath string) (Config, error) {
	if modelPath == "" {
		return Config{}, fmt.Errorf("%w: model path is empty", ErrInvalidConfig)
	}

	info, err := os.Stat(modelPath)
	if err != nil {
		return Config{}, fmt.Errorf("%w: cannot stat model path %q: %v", ErrInvalidConfig, modelPath, err)
	}
	dir := modelPath
	if !info.IsDir() {
		dir = filepath.Dir(modelPath)
	}

	manifest, err := readStrictJSON[manifestFile](filepath.Join(dir, "manifest.json"))
	if err != nil {
		return Config{}, err
	}
	thresholdReport, err := readStrictJSON[thresholdReportFile](filepath.Join(dir, "reference", "threshold_report.json"))
	if err != nil {
		return Config{}, err
	}
	modelCard, err := readStrictJSON[modelCardFile](filepath.Join(dir, "reference", "model_card.json"))
	if err != nil {
		return Config{}, err
	}

	cfg, err := manifest.config(dir)
	if err != nil {
		return Config{}, err
	}
	if err := validateStaticConfig(&cfg, &manifest, &thresholdReport, &modelCard); err != nil {
		return Config{}, err
	}
	if err := verifyRMPChecksums(dir); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// readStrictJSON rejects both unknown fields and trailing JSON values.
func readStrictJSON[T any](path string) (T, error) {
	var value T
	data, err := os.ReadFile(path)
	if err != nil {
		return value, fmt.Errorf("%w: cannot read %q: %v", ErrInvalidContract, path, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return value, fmt.Errorf("%w: cannot parse %q: %v", ErrInvalidContract, path, err)
	}
	if err := trailingJSON(decoder); err != nil {
		return value, fmt.Errorf("%w: %q: %v", ErrInvalidContract, path, err)
	}
	return value, nil
}

func trailingJSON(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON value")
		}
		return err
	}
	return nil
}

type manifestFile struct {
	FormatVersion   string         `json:"format_version"`
	InputSize       []int          `json:"input_size"`
	Labels          string         `json:"labels"`
	ModelType       string         `json:"model_type"`
	ModelVersion    string         `json:"model_version"`
	Task            string         `json:"task"`
	Runtime         string         `json:"runtime"`
	RuntimeContract string         `json:"runtime_contract"`
	Params          manifestParams `json:"params"`
}

type manifestParams struct {
	CalibDataset struct {
		Name        string `json:"name"`
		Fingerprint string `json:"fingerprint"`
		Normal      int    `json:"normal"`
		Anomalous   int    `json:"anomalous"`
	} `json:"calib_dataset"`
	CalibrationContext    calibrationContextFile `json:"calibration_context"`
	DecisionPolicyVersion string                 `json:"decision_policy_version"`
	DeploymentProfile     string                 `json:"deployment_profile"`
	ExportProfile         string                 `json:"export_profile"`
	InputNormalize        string                 `json:"input_normalize"`
	MapDownsample         int                    `json:"map_downsample"`
	ModelSpec             modelSpecJSON          `json:"model_spec"`
	OnnxInput             struct {
		Batch struct {
			Kind string `json:"kind"`
		} `json:"batch"`
	} `json:"onnx_input"`
	OutputAnomalyMap      bool                   `json:"output_anomaly_map"`
	OutputContract        outputContractFile     `json:"output_contract"`
	PreprocessContract    preprocessContractFile `json:"preprocess_contract"`
	RegionPolicy          regionPolicyFile       `json:"region_policy"`
	RegionPolicyVersion   string                 `json:"region_policy_version"`
	Resize                string                 `json:"resize"`
	RuntimeProfile        runtimeProfileFile     `json:"runtime_profile"`
	ScoreContract         scoreContractFile      `json:"score_contract"`
	ScoreSemantics        string                 `json:"score_semantics"`
	ScoreThreshold        float64                `json:"score_threshold"`
	ScoreThresholdFPR     float64                `json:"score_threshold_fpr"`
	ScoreThresholdMethod  string                 `json:"score_threshold_method"`
	ScoreThresholdRecall  float64                `json:"score_threshold_recall"`
	ThresholdCalibratedAt string                 `json:"threshold_calibrated_at"`
	ThresholdVersion      string                 `json:"threshold_version"`
}

type calibrationContextFile struct {
	CameraID          string            `json:"camera_id"`
	CameraSerial      string            `json:"camera_serial"`
	CaptureResolution []int             `json:"capture_resolution"`
	ContextPolicy     contextPolicyFile `json:"context_policy"`
	DecodedResolution []int             `json:"decoded_resolution"`
	Environment       string            `json:"environment"`
	LightingProfile   string            `json:"lighting_profile"`
	ROIPolicy         roiPolicyFile     `json:"roi_policy"`
	ROIPolicyHash     string            `json:"roi_policy_hash"`
}

type contextPolicyFile struct {
	CameraID          string `json:"camera_id"`
	CaptureResolution string `json:"capture_resolution"`
	DecodedResolution string `json:"decoded_resolution"`
	Environment       string `json:"environment"`
	LightingProfile   string `json:"lighting_profile"`
	ROIPolicy         string `json:"roi_policy"`
}

type roiPolicyFile struct {
	CoordinateSpace string    `json:"coordinate_space"`
	Shape           string    `json:"shape"`
	Value           []float64 `json:"value"`
	Version         string    `json:"version"`
}

type modelSpecJSON struct {
	Backbone           string   `json:"backbone,omitempty"`
	Family             string   `json:"family"`
	InputSize          []int    `json:"input_size"`
	Layers             []string `json:"layers,omitempty"`
	NFeatures          int      `json:"n_features,omitempty"`
	NFeaturesOriginal  int      `json:"n_features_original,omitempty"`
	PadMaps            bool     `json:"pad_maps,omitempty"`
	Padding            bool     `json:"padding,omitempty"`
	TeacherOutChannels int      `json:"teacher_out_channels,omitempty"`
	Variant            string   `json:"variant,omitempty"`
}

type outputContractFile struct {
	SchemaVersion string `json:"schema_version"`
	Input         struct {
		Layout string `json:"layout"`
		Name   string `json:"name"`
		Shape  []any  `json:"shape"`
	} `json:"input"`
	RequiredOutputs        []string `json:"required_outputs"`
	ForbiddenOutputs       []string `json:"forbidden_outputs"`
	OutputCount            int      `json:"output_count"`
	OutputNameStrict       bool     `json:"output_name_strict"`
	AllowPositionalOutputs bool     `json:"allow_positional_outputs"`
}

type preprocessContractFile struct {
	ColorSpace string `json:"color_space"`
	InGraph    bool   `json:"in_graph"`
	Normalize  string `json:"normalize"`
	Resize     struct {
		InputSize []int  `json:"input_size"`
		Kind      string `json:"kind"`
	} `json:"resize"`
	ScaleToUnitInterval bool   `json:"scale_to_unit_interval"`
	TensorLayout        string `json:"tensor_layout"`
}

type regionPolicyFile struct {
	AreaReferenceInputSize []int   `json:"area_reference_input_size"`
	DecisionMode           string  `json:"decision_mode"`
	MapThreshold           float64 `json:"map_threshold"`
	MaxRegions             int     `json:"max_regions"`
	MinRegionArea          int     `json:"min_region_area"`
	MinRegionAreaUnit      string  `json:"min_region_area_unit"`
	RegionScore            string  `json:"region_score"`
	Version                string  `json:"version"`
}

type runtimeProfileFile struct {
	BatchMode                       string            `json:"batch_mode"`
	ExecutionProviderFallbackMetric string            `json:"execution_provider_fallback_metric"`
	ExecutionProviderPolicy         map[string]string `json:"execution_provider_policy"`
	InputLimits                     struct {
		MaxBatch  int `json:"max_batch"`
		MaxPixels int `json:"max_pixels"`
		MaxSide   int `json:"max_side"`
		MinSide   int `json:"min_side"`
	} `json:"input_limits"`
	MaxRuntimeBatch int `json:"max_runtime_batch"`
}

type scoreContractFile struct {
	SchemaVersion        string `json:"schema_version"`
	Semantics            string `json:"semantics"`
	Space                string `json:"space"`
	ThresholdSpace       string `json:"threshold_space"`
	PixelMapSpace        string `json:"pixel_map_space"`
	RuntimeNormalization string `json:"runtime_normalization"`
	Source               string `json:"source"`
	PreProcessorInGraph  bool   `json:"pre_processor_in_graph"`
	PostProcessorInGraph bool   `json:"post_processor_in_graph"`
	Assertion            struct {
		Atol    float64 `json:"atol"`
		Enabled bool    `json:"enabled"`
		Rtol    float64 `json:"rtol"`
	} `json:"assertion"`
}

type thresholdReportFile struct {
	Calibration struct {
		AnomalyCount       int                    `json:"anomaly_count"`
		Context            calibrationContextFile `json:"context"`
		DatasetFingerprint string                 `json:"dataset_fingerprint"`
		DatasetName        string                 `json:"dataset_name"`
		NormalCount        int                    `json:"normal_count"`
	} `json:"calibration"`
	DataSplitContract dataSplitFile   `json:"data_split_contract"`
	Metrics           json.RawMessage `json:"metrics"`
	ModelIdentity     struct {
		DecisionPolicyVersion string        `json:"decision_policy_version"`
		ExportHash            string        `json:"export_hash"`
		ExportProfile         string        `json:"export_profile"`
		ModelHash             string        `json:"model_hash"`
		ModelSpec             modelSpecJSON `json:"model_spec"`
		RegionPolicyVersion   string        `json:"region_policy_version"`
		ThresholdVersion      string        `json:"threshold_version"`
		TrainingConfigHash    string        `json:"training_config_hash"`
	} `json:"model_identity"`
	SchemaVersion string `json:"schema_version"`
	Threshold     struct {
		CalibratedAt string `json:"calibrated_at"`
		Image        struct {
			Method struct {
				SelectionObjective string  `json:"selection_objective"`
				TargetFPR          float64 `json:"target_fpr"`
				Type               string  `json:"type"`
			} `json:"method"`
			Value float64 `json:"value"`
		} `json:"image"`
		Pixel struct {
			Method struct {
				Quantile           float64 `json:"quantile"`
				SelectionObjective string  `json:"selection_objective"`
				Type               string  `json:"type"`
			} `json:"method"`
			Value float64 `json:"value"`
		} `json:"pixel"`
		RegionPolicy regionPolicyFile `json:"region_policy"`
	} `json:"threshold"`
}

type dataSplitFile struct {
	SchemaVersion            string `json:"schema_version"`
	TrainingFingerprint      string `json:"training_fingerprint"`
	CalibrationFingerprint   string `json:"calibration_fingerprint"`
	QualificationFingerprint string `json:"qualification_fingerprint"`
	OverlapPolicy            string `json:"overlap_policy"`
	QualificationImmutable   bool   `json:"qualification_immutable"`
}

type modelCardFile struct {
	CheckpointHash    string        `json:"checkpoint_hash"`
	DataSplitContract dataSplitFile `json:"data_split_contract"`
	ExportProfile     struct {
		Name     string `json:"name"`
		Exporter struct {
			Backend string `json:"backend"`
			Dynamo  bool   `json:"dynamo"`
			Name    string `json:"name"`
		} `json:"exporter"`
		Opset     int               `json:"opset"`
		Toolchain map[string]string `json:"toolchain"`
	} `json:"export_profile"`
	ModelArtifactHash string        `json:"model_artifact_hash"`
	ModelSpec         modelSpecJSON `json:"model_spec"`
	ModelType         string        `json:"model_type"`
	ModelVersion      string        `json:"model_version"`
	ONNX              struct {
		ExecutionCoverage string `json:"execution_coverage"`
		Input             struct {
			DType string `json:"dtype"`
			Name  string `json:"name"`
			Shape []any  `json:"shape"`
		} `json:"input"`
		Outputs struct {
			AnomalyMap struct {
				DType string `json:"dtype"`
				Shape []any  `json:"shape"`
			} `json:"anomaly_map"`
			PredScore struct {
				DType string `json:"dtype"`
				Shape []any  `json:"shape"`
			} `json:"pred_score"`
		} `json:"outputs"`
		ProviderSelected []string `json:"provider_selected"`
	} `json:"onnx"`
	TrainingConfigHash string `json:"training_config_hash"`
	SchemaVersion      string `json:"schema_version"`
	ValidatedVariant   string `json:"validated_variant"`
	Validation         struct {
		Comparisons json.RawMessage `json:"comparisons"`
		Status      string          `json:"status"`
	} `json:"validation"`
}

func (m *manifestFile) config(dir string) (Config, error) {
	contextPolicy := AnomalyContextPolicy{
		CameraID:          AnomalyContextGate(m.Params.CalibrationContext.ContextPolicy.CameraID),
		CaptureResolution: AnomalyContextGate(m.Params.CalibrationContext.ContextPolicy.CaptureResolution),
		DecodedResolution: AnomalyContextGate(m.Params.CalibrationContext.ContextPolicy.DecodedResolution),
		ROIPolicy:         AnomalyContextGate(m.Params.CalibrationContext.ContextPolicy.ROIPolicy),
		LightingProfile:   AnomalyContextGate(m.Params.CalibrationContext.ContextPolicy.LightingProfile),
		Environment:       AnomalyContextGate(m.Params.CalibrationContext.ContextPolicy.Environment),
	}
	cfg := Config{
		ModelPath:        filepath.Join(dir, "model.onnx"),
		ModelKind:        m.ModelType,
		ModelVersion:     m.ModelVersion,
		ScoreThreshold:   float32(m.Params.ScoreThreshold),
		MapThreshold:     float32(m.Params.RegionPolicy.MapThreshold),
		MinRegionArea:    m.Params.RegionPolicy.MinRegionArea,
		MaxRegions:       m.Params.RegionPolicy.MaxRegions,
		OutputAnomalyMap: m.Params.OutputAnomalyMap,
		MapDownsample:    m.Params.MapDownsample,
		RuntimeBatchMode: m.Params.RuntimeProfile.BatchMode,
		RuntimeMaxBatch:  m.Params.RuntimeProfile.MaxRuntimeBatch,
		InputLimits: InputLimitsConfig{
			MinSide:   m.Params.RuntimeProfile.InputLimits.MinSide,
			MaxSide:   m.Params.RuntimeProfile.InputLimits.MaxSide,
			MaxPixels: m.Params.RuntimeProfile.InputLimits.MaxPixels,
			MaxBatch:  m.Params.RuntimeProfile.InputLimits.MaxBatch,
		},
		ExecutionProviderFallbackMetric: m.Params.RuntimeProfile.ExecutionProviderFallbackMetric,
		ExecutionProviderPolicy:         cloneStringMap(m.Params.RuntimeProfile.ExecutionProviderPolicy),
		ExportProfile:                   m.Params.ExportProfile,
		DeploymentProfile:               m.Params.DeploymentProfile,
		RegionDecisionMode:              m.Params.RegionPolicy.DecisionMode,
		RegionPolicyVersion:             m.Params.RegionPolicyVersion,
		DecisionPolicyVersion:           m.Params.DecisionPolicyVersion,
		ThresholdVersion:                m.Params.ThresholdVersion,
		ScoreSemantics:                  ScoreSemantics(m.Params.ScoreSemantics),
		OutputContract: OutputContractConfig{
			SchemaVersion:          m.Params.OutputContract.SchemaVersion,
			InputName:              m.Params.OutputContract.Input.Name,
			InputShape:             append([]any(nil), m.Params.OutputContract.Input.Shape...),
			Layout:                 m.Params.OutputContract.Input.Layout,
			RequiredOutputs:        append([]string(nil), m.Params.OutputContract.RequiredOutputs...),
			ForbiddenOutputs:       append([]string(nil), m.Params.OutputContract.ForbiddenOutputs...),
			OutputCount:            m.Params.OutputContract.OutputCount,
			OutputNameStrict:       m.Params.OutputContract.OutputNameStrict,
			AllowPositionalOutputs: m.Params.OutputContract.AllowPositionalOutputs,
		},
		ScoreContract: ScoreContractConfig{
			SchemaVersion:        m.Params.ScoreContract.SchemaVersion,
			Semantics:            ScoreSemantics(m.Params.ScoreContract.Semantics),
			Space:                m.Params.ScoreContract.Space,
			ThresholdSpace:       m.Params.ScoreContract.ThresholdSpace,
			PixelMapSpace:        m.Params.ScoreContract.PixelMapSpace,
			RuntimeNormalization: m.Params.ScoreContract.RuntimeNormalization,
			Source:               m.Params.ScoreContract.Source,
			PreProcessorInGraph:  m.Params.ScoreContract.PreProcessorInGraph,
			PostProcessorInGraph: m.Params.ScoreContract.PostProcessorInGraph,
			Assertion: ScoreAssertionConfig{
				Enabled: m.Params.ScoreContract.Assertion.Enabled,
				Atol:    m.Params.ScoreContract.Assertion.Atol,
				Rtol:    m.Params.ScoreContract.Assertion.Rtol,
			},
		},
		PreprocessContract: PreprocessContractConfig{
			ColorSpace:          m.Params.PreprocessContract.ColorSpace,
			TensorLayout:        m.Params.PreprocessContract.TensorLayout,
			Normalize:           m.Params.PreprocessContract.Normalize,
			ResizeKind:          m.Params.PreprocessContract.Resize.Kind,
			InputSize:           pointFromSize(m.Params.PreprocessContract.Resize.InputSize),
			ScaleToUnitInterval: m.Params.PreprocessContract.ScaleToUnitInterval,
			InGraph:             m.Params.PreprocessContract.InGraph,
		},
		InputSize:     pointFromSize(m.InputSize),
		ContextPolicy: contextPolicy,
		CalibrationContext: CalibrationContextConfig{
			CameraID:          m.Params.CalibrationContext.CameraID,
			CameraSerial:      m.Params.CalibrationContext.CameraSerial,
			CaptureResolution: pointFromSize(m.Params.CalibrationContext.CaptureResolution),
			DecodedResolution: pointFromSize(m.Params.CalibrationContext.DecodedResolution),
			ROIPolicyVersion:  m.Params.CalibrationContext.ROIPolicy.Version,
			ROIPolicyHash:     m.Params.CalibrationContext.ROIPolicyHash,
			ROIPolicyValue:    append([]float64(nil), m.Params.CalibrationContext.ROIPolicy.Value...),
			LightingProfile:   m.Params.CalibrationContext.LightingProfile,
			Environment:       m.Params.CalibrationContext.Environment,
			ContextPolicy:     contextPolicy,
		},
	}

	spec, err := json.Marshal(m.Params.ModelSpec)
	if err != nil {
		return Config{}, fmt.Errorf("%w: cannot canonicalize model_spec: %v", ErrInvalidContract, err)
	}
	cfg.ModelSpec = string(spec)
	return cfg, nil
}

func cloneStringMap(input map[string]string) map[string]string {
	if input == nil {
		return nil
	}
	output := make(map[string]string, len(input))
	for key, value := range input {
		output[key] = value
	}
	return output
}

func pointFromSize(size []int) image.Point {
	if len(size) != 2 {
		return image.Point{}
	}
	return image.Point{X: size[1], Y: size[0]}
}

func validateStaticConfig(cfg *Config, manifest *manifestFile, report *thresholdReportFile, card *modelCardFile) error {
	if manifest.FormatVersion != manifestSchemaVersion {
		return contractError("manifest format_version", manifest.FormatVersion, manifestSchemaVersion)
	}
	if manifest.Task != "anomaly" {
		return contractError("manifest task", manifest.Task, "anomaly")
	}
	if manifest.Runtime != "onnxruntime" {
		return contractError("manifest runtime", manifest.Runtime, "onnxruntime")
	}
	if manifest.RuntimeContract != RuntimeContract {
		return contractError("runtime_contract", manifest.RuntimeContract, RuntimeContract)
	}
	if manifest.ModelType != modelKindEfficientAD && manifest.ModelType != modelKindPadim {
		return fmt.Errorf("%w: unsupported model_type %q", ErrInvalidContract, manifest.ModelType)
	}
	if manifest.Labels != "labels.txt" {
		return contractError("labels reference", manifest.Labels, "labels.txt")
	}
	if err := validateLabels(filepath.Dir(cfg.ModelPath), "anomaly"); err != nil {
		return err
	}

	if len(manifest.InputSize) != 2 {
		return fmt.Errorf("%w: manifest input_size must contain exactly height and width", ErrInvalidContract)
	}
	if manifest.InputSize[0] < 1 || manifest.InputSize[1] < 1 {
		return fmt.Errorf("%w: manifest input_size is non-positive", ErrInvalidContract)
	}
	if cfg.InputSize != cfg.PreprocessContract.InputSize {
		return fmt.Errorf("%w: manifest and preprocess input sizes differ", ErrInvalidContract)
	}
	if len(manifest.Params.ModelSpec.InputSize) != 2 || manifest.Params.ModelSpec.InputSize[0] != manifest.InputSize[0] || manifest.Params.ModelSpec.InputSize[1] != manifest.InputSize[1] {
		return fmt.Errorf("%w: model_spec input_size differs from manifest input_size", ErrInvalidContract)
	}

	if err := validateModelSpec(manifest, card); err != nil {
		return err
	}
	if err := validateModelSpecAgainstReport(manifest, report); err != nil {
		return err
	}
	if cfg.ModelVersion != card.ModelVersion {
		return contractError("model_version", cfg.ModelVersion, card.ModelVersion)
	}
	if manifest.ModelType != card.ModelType {
		return contractError("model_type", manifest.ModelType, card.ModelType)
	}
	if manifest.Params.ExportProfile != card.ExportProfile.Name {
		return contractError("export_profile", manifest.Params.ExportProfile, card.ExportProfile.Name)
	}

	if err := validateProfileAndVariant(cfg, card); err != nil {
		return err
	}
	if err := validateScoreAndThreshold(cfg, manifest, report); err != nil {
		return err
	}
	if err := validateRegionPolicy(cfg, manifest, report); err != nil {
		return err
	}
	if err := validateRuntimeProfile(cfg, manifest); err != nil {
		return err
	}
	if err := validateContext(cfg); err != nil {
		return err
	}
	if err := validateIdentity(cfg, report, card); err != nil {
		return err
	}
	if err := validateONNXCard(manifest, card); err != nil {
		return err
	}
	if report.DataSplitContract != card.DataSplitContract {
		return fmt.Errorf("%w: threshold report and model card data split contracts differ", ErrInvalidContract)
	}
	if report.DataSplitContract.SchemaVersion != dataSplitSchema || report.DataSplitContract.OverlapPolicy != "forbidden" || !report.DataSplitContract.QualificationImmutable {
		return fmt.Errorf("%w: invalid data split contract", ErrInvalidContract)
	}
	if len(report.DataSplitContract.TrainingFingerprint) == 0 || len(report.DataSplitContract.CalibrationFingerprint) == 0 || len(report.DataSplitContract.QualificationFingerprint) == 0 {
		return fmt.Errorf("%w: data split contract has an empty fingerprint", ErrInvalidContract)
	}
	return nil
}

func validateLabels(dir, expected string) error {
	data, err := os.ReadFile(filepath.Join(dir, "labels.txt"))
	if err != nil {
		return fmt.Errorf("%w: cannot read labels: %v", ErrInvalidContract, err)
	}
	lines := strings.Fields(string(data))
	if len(lines) != 1 || lines[0] != expected {
		return fmt.Errorf("%w: anomaly-v1 requires labels.txt to contain exactly %q", ErrInvalidContract, expected)
	}
	return nil
}

func validateModelSpec(manifest *manifestFile, card *modelCardFile) error {
	if !reflect.DeepEqual(manifest.Params.ModelSpec, card.ModelSpec) {
		return fmt.Errorf("%w: manifest and model card model specs differ", ErrInvalidContract)
	}
	return nil
}

func validateModelSpecAgainstReport(manifest *manifestFile, report *thresholdReportFile) error {
	if !reflect.DeepEqual(manifest.Params.ModelSpec, report.ModelIdentity.ModelSpec) {
		return fmt.Errorf("%w: manifest and threshold report model specs differ", ErrInvalidContract)
	}
	return nil
}

func validateProfileAndVariant(cfg *Config, card *modelCardFile) error {
	switch cfg.DeploymentProfile {
	case deploymentProduction:
		if cfg.ExportProfile != ExportProfile {
			return fmt.Errorf("%w: production export_profile must be %q", ErrInvalidContract, ExportProfile)
		}
		if card.ValidatedVariant != ValidatedEfficientAdSmall && card.ValidatedVariant != ValidatedPadimR18 {
			return fmt.Errorf("%w: production model uses unvalidated variant %q", ErrInvalidContract, card.ValidatedVariant)
		}
		if card.Validation.Status != "PASS" {
			return contractError("model validation status", card.Validation.Status, "PASS")
		}
		cfg.ValidatedVariant = card.ValidatedVariant
	case deploymentDevelopment:
		if cfg.ExportProfile != ExportProfile {
			ortlog.Warnw("development anomaly model does not use the production export profile",
				"export_profile", cfg.ExportProfile)
		}
	default:
		return fmt.Errorf("%w: unsupported deployment_profile %q", ErrInvalidContract, cfg.DeploymentProfile)
	}
	return nil
}

func validateScoreAndThreshold(cfg *Config, manifest *manifestFile, report *thresholdReportFile) error {
	if cfg.ScoreSemantics != ScoreSemanticsMapMax {
		return fmt.Errorf("%w: anomaly-v1 requires score semantics map_max", ErrInvalidContract)
	}
	if manifest.Params.ScoreSemantics != manifest.Params.ScoreContract.Semantics || string(cfg.ScoreSemantics) != manifest.Params.ScoreContract.Semantics {
		return contractError("score semantics", manifest.Params.ScoreSemantics, manifest.Params.ScoreContract.Semantics)
	}
	if manifest.Params.ScoreContract.SchemaVersion != scoreContractSchemaVersion {
		return contractError("score contract schema", manifest.Params.ScoreContract.SchemaVersion, scoreContractSchemaVersion)
	}
	if manifest.Params.ScoreContract.Space != "raw_model" || manifest.Params.ScoreContract.ThresholdSpace != "raw_model" || manifest.Params.ScoreContract.PixelMapSpace != "raw_model" {
		return fmt.Errorf("%w: anomaly-v1 requires raw_model score, threshold and pixel-map spaces", ErrInvalidContract)
	}
	if manifest.Params.ScoreContract.RuntimeNormalization != "none" || manifest.Params.ScoreContract.PreProcessorInGraph || manifest.Params.ScoreContract.PostProcessorInGraph {
		return fmt.Errorf("%w: anomaly-v1 forbids in-graph preprocessing/postprocessing or runtime normalization", ErrInvalidContract)
	}
	if manifest.Params.InputNormalize != manifest.Params.PreprocessContract.Normalize || manifest.Params.PreprocessContract.InGraph {
		return fmt.Errorf("%w: manifest and preprocess normalization contracts differ", ErrInvalidContract)
	}
	if manifest.Params.PreprocessContract.Normalize != normalizeDivide255 && manifest.Params.PreprocessContract.Normalize != normalizeImageNet {
		return fmt.Errorf("%w: unsupported normalization %q", ErrInvalidContract, manifest.Params.PreprocessContract.Normalize)
	}
	if manifest.Params.PreprocessContract.Resize.Kind != "stretch" || manifest.Params.Resize != "stretch" {
		return fmt.Errorf("%w: anomaly-v1 requires stretch resize", ErrInvalidContract)
	}
	if !manifest.Params.PreprocessContract.ScaleToUnitInterval || manifest.Params.PreprocessContract.ColorSpace != "RGB" || manifest.Params.PreprocessContract.TensorLayout != "NCHW" {
		return fmt.Errorf("%w: invalid RGB/NCHW unit-interval preprocess contract", ErrInvalidContract)
	}

	if err := finiteFloat64("score_threshold", manifest.Params.ScoreThreshold); err != nil {
		return err
	}
	if err := finiteFloat64("map_threshold", manifest.Params.RegionPolicy.MapThreshold); err != nil {
		return err
	}
	if manifest.Params.ScoreThreshold < 0 || manifest.Params.RegionPolicy.MapThreshold < 0 {
		return fmt.Errorf("%w: thresholds must be non-negative", ErrInvalidContract)
	}
	if manifest.Params.ScoreThresholdMethod != "calibrated_policy" {
		return contractError("score threshold method", manifest.Params.ScoreThresholdMethod, "calibrated_policy")
	}
	if !floatsClose(manifest.Params.ScoreThreshold, report.Threshold.Image.Value) {
		return fmt.Errorf("%w: manifest score threshold differs from threshold report", ErrInvalidContract)
	}
	if !floatsClose(manifest.Params.RegionPolicy.MapThreshold, report.Threshold.Pixel.Value) {
		return fmt.Errorf("%w: manifest map threshold differs from threshold report", ErrInvalidContract)
	}
	if cfg.ScoreThreshold < 0 || math.IsNaN(float64(cfg.ScoreThreshold)) || math.IsInf(float64(cfg.ScoreThreshold), 0) {
		return fmt.Errorf("%w: converted score threshold is not finite", ErrInvalidContract)
	}
	if cfg.MapThreshold < 0 || math.IsNaN(float64(cfg.MapThreshold)) || math.IsInf(float64(cfg.MapThreshold), 0) {
		return fmt.Errorf("%w: converted map threshold is not finite", ErrInvalidContract)
	}
	if cfg.ThresholdVersion != report.ModelIdentity.ThresholdVersion {
		return contractError("threshold version", cfg.ThresholdVersion, report.ModelIdentity.ThresholdVersion)
	}
	if cfg.DecisionPolicyVersion != report.ModelIdentity.DecisionPolicyVersion {
		return contractError("decision policy version", cfg.DecisionPolicyVersion, report.ModelIdentity.DecisionPolicyVersion)
	}
	if _, err := time.Parse(time.RFC3339, manifest.Params.ThresholdCalibratedAt); err != nil {
		return fmt.Errorf("%w: invalid threshold_calibrated_at: %v", ErrInvalidContract, err)
	}
	if manifest.Params.ThresholdCalibratedAt != report.Threshold.CalibratedAt {
		return fmt.Errorf("%w: manifest and threshold report calibration timestamps differ", ErrInvalidContract)
	}
	if manifest.Params.CalibDataset.Name != report.Calibration.DatasetName ||
		manifest.Params.CalibDataset.Fingerprint != report.Calibration.DatasetFingerprint ||
		manifest.Params.CalibDataset.Normal != report.Calibration.NormalCount ||
		manifest.Params.CalibDataset.Anomalous != report.Calibration.AnomalyCount {
		return fmt.Errorf("%w: manifest and threshold report calibration datasets differ", ErrInvalidContract)
	}
	if !reflect.DeepEqual(manifest.Params.CalibrationContext, report.Calibration.Context) {
		return fmt.Errorf("%w: manifest and threshold report calibration contexts differ", ErrInvalidContract)
	}
	return nil
}

func validateRegionPolicy(cfg *Config, manifest *manifestFile, report *thresholdReportFile) error {
	policy := manifest.Params.RegionPolicy
	if policy.Version != regionPolicyVersionV1 || cfg.RegionPolicyVersion != regionPolicyVersionV1 || report.ModelIdentity.RegionPolicyVersion != regionPolicyVersionV1 {
		return fmt.Errorf("%w: anomaly-v1 requires region-policy-v1", ErrInvalidContract)
	}
	if policy.DecisionMode != "independent" || policy.RegionScore != "peak_score" || policy.MinRegionAreaUnit != "model_pixel" {
		return fmt.Errorf("%w: unsupported region policy semantics", ErrInvalidContract)
	}
	if policy.MinRegionArea < 1 || policy.MaxRegions < 1 {
		return fmt.Errorf("%w: region policy limits must be positive", ErrInvalidContract)
	}
	if len(policy.AreaReferenceInputSize) != 2 || policy.AreaReferenceInputSize[0] != manifest.InputSize[0] || policy.AreaReferenceInputSize[1] != manifest.InputSize[1] {
		return fmt.Errorf("%w: region policy area reference differs from model input", ErrInvalidContract)
	}
	if !reflect.DeepEqual(policy, report.Threshold.RegionPolicy) {
		return fmt.Errorf("%w: manifest and threshold report region policies differ", ErrInvalidContract)
	}
	if cfg.MinRegionArea != policy.MinRegionArea || cfg.MaxRegions != policy.MaxRegions || cfg.MapThreshold != float32(policy.MapThreshold) {
		return fmt.Errorf("%w: region policy values were not copied consistently", ErrInvalidContract)
	}
	if cfg.MapDownsample < 0 {
		return fmt.Errorf("%w: map_downsample must be zero or positive", ErrInvalidContract)
	}
	if cfg.MapDownsample > 1 && !cfg.OutputAnomalyMap {
		return fmt.Errorf("%w: map_downsample requires output_anomaly_map", ErrInvalidContract)
	}
	return nil
}

func validateRuntimeProfile(cfg *Config, manifest *manifestFile) error {
	if cfg.RuntimeBatchMode != "single" || cfg.RuntimeMaxBatch != 1 || manifest.Params.RuntimeProfile.MaxRuntimeBatch != 1 {
		return fmt.Errorf("%w: anomaly-v1 requires runtime batch mode single and max_runtime_batch 1", ErrInvalidContract)
	}
	if manifest.Params.OnnxInput.Batch.Kind != "dynamic" {
		return fmt.Errorf("%w: ONNX dynamic batch declaration is required for anomaly-export-v1", ErrInvalidContract)
	}
	limits := manifest.Params.RuntimeProfile.InputLimits
	if limits.MinSide < 1 || limits.MaxSide < limits.MinSide || limits.MaxPixels < 1 || limits.MaxBatch < 1 {
		return fmt.Errorf("%w: invalid input limits", ErrInvalidContract)
	}
	if cfg.InputSize.Y < limits.MinSide || cfg.InputSize.X < limits.MinSide || cfg.InputSize.Y > limits.MaxSide || cfg.InputSize.X > limits.MaxSide || int64(cfg.InputSize.Y)*int64(cfg.InputSize.X) > int64(limits.MaxPixels) {
		return fmt.Errorf("%w: ONNX input exceeds declared input limits", ErrInvalidContract)
	}
	if limits.MaxBatch < cfg.RuntimeMaxBatch {
		return fmt.Errorf("%w: runtime max batch exceeds input limits", ErrInvalidContract)
	}
	if cfg.ExecutionProviderFallbackMetric == "" {
		return fmt.Errorf("%w: execution provider fallback metric is empty", ErrInvalidContract)
	}
	return validateExecutionProviderPolicy(cfg.ExecutionProviderPolicy)
}

func validateExecutionProviderPolicy(policy map[string]string) error {
	if len(policy) == 0 {
		return fmt.Errorf("%w: execution provider policy is empty", ErrInvalidContract)
	}
	if policy["cpu"] != providerRequired {
		return fmt.Errorf("%w: CPU must be a required execution provider fallback", ErrInvalidContract)
	}
	for name, mode := range policy {
		switch name {
		case "cpu":
		case "cuda", "coreml":
			if mode != providerRequired && mode != providerPreferred {
				return fmt.Errorf("%w: execution provider %q has invalid mode %q", ErrInvalidContract, name, mode)
			}
		default:
			return fmt.Errorf("%w: unsupported execution provider %q", ErrInvalidContract, name)
		}
	}
	return nil
}

func validateContext(cfg *Config) error {
	roi := cfg.CalibrationContext
	if roi.ContextPolicy != cfg.ContextPolicy {
		return fmt.Errorf("%w: calibration context policy differs from runtime policy", ErrInvalidContract)
	}
	if roi.CaptureResolution.X <= 0 || roi.CaptureResolution.Y <= 0 || roi.DecodedResolution.X <= 0 || roi.DecodedResolution.Y <= 0 {
		return fmt.Errorf("%w: calibration context resolutions must be positive", ErrInvalidContract)
	}
	for _, gate := range []AnomalyContextGate{cfg.ContextPolicy.CameraID, cfg.ContextPolicy.CaptureResolution, cfg.ContextPolicy.DecodedResolution, cfg.ContextPolicy.ROIPolicy, cfg.ContextPolicy.LightingProfile, cfg.ContextPolicy.Environment} {
		if gate != ContextGateHard && gate != ContextGateObserve {
			return fmt.Errorf("%w: invalid context gate %q", ErrInvalidContract, gate)
		}
	}
	if cfg.ContextPolicy.ROIPolicy != ContextGateHard {
		return fmt.Errorf("%w: effective ROI policy must be hard_gate", ErrInvalidContract)
	}
	if roi.ROIPolicyVersion == "" {
		return fmt.Errorf("%w: ROI policy version is empty", ErrInvalidContract)
	}
	if !isSHA256(roi.ROIPolicyHash) {
		return fmt.Errorf("%w: ROI policy hash is not a valid sha256 digest", ErrInvalidContract)
	}
	fullFrame := roi.ROIPolicyVersion == "rect-v1"
	if !fullFrame {
		return fmt.Errorf("%w: unsupported ROI policy version %q", ErrInvalidContract, roi.ROIPolicyVersion)
	}
	return validateCanonicalFullFrameROI(cfg)
}

func validateCanonicalFullFrameROI(cfg *Config) error {
	// The manifest's ROI policy value is not exposed on Config, so compare it
	// through the strict manifest before Config is returned. This helper is
	// invoked from validateStaticConfig with the original manifest context.
	if len(cfg.CalibrationContext.ROIPolicyValue) != 4 {
		return fmt.Errorf("%w: canonical full-frame ROI policy requires four values", ErrInvalidContract)
	}
	expected := []float64{0, 0, 1, 1}
	for index, value := range cfg.CalibrationContext.ROIPolicyValue {
		if value != expected[index] {
			return fmt.Errorf("%w: canonical full-frame ROI policy is not [0,0,1,1]", ErrInvalidContract)
		}
	}
	return nil
}

func validateIdentity(cfg *Config, report *thresholdReportFile, card *modelCardFile) error {
	if !isSHA256(report.ModelIdentity.ExportHash) || !isSHA256(report.ModelIdentity.ModelHash) || !isSHA256(card.CheckpointHash) || !isSHA256(card.ModelArtifactHash) {
		return fmt.Errorf("%w: model identity contains an invalid sha256 digest", ErrInvalidContract)
	}
	if report.ModelIdentity.ExportHash != card.ModelArtifactHash {
		return fmt.Errorf("%w: threshold report export hash differs from model artifact hash", ErrInvalidContract)
	}
	if report.ModelIdentity.ModelHash != card.CheckpointHash {
		return fmt.Errorf("%w: threshold report model hash differs from checkpoint hash", ErrInvalidContract)
	}
	if !isSHA256(report.ModelIdentity.TrainingConfigHash) || !isSHA256(card.TrainingConfigHash) {
		return fmt.Errorf("%w: training config hash is not a valid sha256 digest", ErrInvalidContract)
	}
	if report.ModelIdentity.TrainingConfigHash != card.TrainingConfigHash {
		return fmt.Errorf("%w: threshold report training config hash differs from model card", ErrInvalidContract)
	}
	if report.ModelIdentity.ExportProfile != cfg.ExportProfile {
		return contractError("threshold report export profile", report.ModelIdentity.ExportProfile, cfg.ExportProfile)
	}
	if report.SchemaVersion != thresholdReportSchema {
		return contractError("threshold report schema", report.SchemaVersion, thresholdReportSchema)
	}
	if card.SchemaVersion != modelCardSchema {
		return contractError("model card schema", card.SchemaVersion, modelCardSchema)
	}
	if card.ExportProfile.Opset < 1 || len(card.ExportProfile.Toolchain) == 0 {
		return fmt.Errorf("%w: model card export toolchain is incomplete", ErrInvalidContract)
	}
	for _, key := range []string{"anomalib", "onnx", "onnxruntime", "python", "torch", "torchvision"} {
		if card.ExportProfile.Toolchain[key] == "" {
			return fmt.Errorf("%w: model card export toolchain is missing %q", ErrInvalidContract, key)
		}
	}
	return nil
}

func validateONNXCard(manifest *manifestFile, card *modelCardFile) error {
	contract := manifest.Params.OutputContract
	if contract.SchemaVersion != outputContractSchemaVersion {
		return contractError("output contract schema", contract.SchemaVersion, outputContractSchemaVersion)
	}
	if !contract.OutputNameStrict || contract.AllowPositionalOutputs || contract.OutputCount != 2 {
		return fmt.Errorf("%w: anomaly-v1 requires exactly two strict named outputs", ErrInvalidContract)
	}
	if contract.Input.Layout != "NCHW" || contract.Input.Name != "input" {
		return fmt.Errorf("%w: anomaly-v1 requires NCHW input named input", ErrInvalidContract)
	}
	required := map[string]bool{"pred_score": false, "anomaly_map": false}
	if len(contract.RequiredOutputs) != 2 {
		return fmt.Errorf("%w: output contract must require pred_score and anomaly_map", ErrInvalidContract)
	}
	for _, name := range contract.RequiredOutputs {
		if _, ok := required[name]; !ok {
			return fmt.Errorf("%w: unexpected required output %q", ErrInvalidContract, name)
		}
		required[name] = true
	}
	if !required["pred_score"] || !required["anomaly_map"] {
		return fmt.Errorf("%w: output contract must require both pred_score and anomaly_map", ErrInvalidContract)
	}
	for _, forbidden := range contract.ForbiddenOutputs {
		if required[forbidden] {
			return fmt.Errorf("%w: output %q is both required and forbidden", ErrInvalidContract, forbidden)
		}
	}
	if len(contract.ForbiddenOutputs) != 2 || contract.ForbiddenOutputs[0] != "pred_label" || contract.ForbiddenOutputs[1] != "pred_mask" {
		return fmt.Errorf("%w: output contract must explicitly forbid pred_label and pred_mask", ErrInvalidContract)
	}

	cardInputShape := []any{"B", float64(3), float64(manifest.InputSize[0]), float64(manifest.InputSize[1])}
	if !shapeEqual(card.ONNX.Input.Shape, cardInputShape) {
		return fmt.Errorf("%w: model card ONNX input shape differs from manifest", ErrInvalidContract)
	}
	if !shapeEqual(contract.Input.Shape, cardInputShape) {
		return fmt.Errorf("%w: output contract input shape differs from manifest", ErrInvalidContract)
	}
	cardMapShape := []any{"B", float64(1), float64(manifest.InputSize[0]), float64(manifest.InputSize[1])}
	if !shapeEqual(card.ONNX.Outputs.AnomalyMap.Shape, cardMapShape) {
		return fmt.Errorf("%w: model card anomaly_map shape differs from manifest", ErrInvalidContract)
	}
	cardScoreShape := []any{"B"}
	if !shapeEqual(card.ONNX.Outputs.PredScore.Shape, cardScoreShape) {
		return fmt.Errorf("%w: model card pred_score shape differs from manifest", ErrInvalidContract)
	}
	if card.ONNX.Input.DType != "float32" || card.ONNX.Outputs.AnomalyMap.DType != "float32" || card.ONNX.Outputs.PredScore.DType != "float32" {
		return fmt.Errorf("%w: anomaly-v1 requires float32 ONNX tensors", ErrInvalidContract)
	}
	return nil
}

func shapeEqual(actual, expected []any) bool {
	if len(actual) != len(expected) {
		return false
	}
	for i, value := range actual {
		switch typed := value.(type) {
		case string:
			text, ok := expected[i].(string)
			if !ok || typed != text {
				return false
			}
		case float64:
			number, ok := expected[i].(float64)
			if !ok || typed != number {
				return false
			}
		case int:
			number, ok := expected[i].(float64)
			if !ok || float64(typed) != number {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func verifyRMPChecksums(dir string) error {
	data, err := os.ReadFile(filepath.Join(dir, "checksums.txt"))
	if err != nil {
		return fmt.Errorf("%w: cannot read checksums: %v", ErrInvalidChecksum, err)
	}
	expected := map[string]string{}
	for _, line := range strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 || !strings.HasPrefix(fields[0], "sha256:") {
			return fmt.Errorf("%w: malformed checksum line %q", ErrInvalidChecksum, line)
		}
		digest := strings.TrimPrefix(fields[0], "sha256:")
		rel := filepath.Clean(filepath.FromSlash(fields[1]))
		if filepath.IsAbs(rel) || rel == "." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || rel == ".." {
			return fmt.Errorf("%w: checksum path escapes RMP: %s", ErrInvalidChecksum, fields[1])
		}
		if len(digest) != 64 {
			return fmt.Errorf("%w: invalid sha256 digest for %s", ErrInvalidChecksum, fields[1])
		}
		if _, err := hex.DecodeString(digest); err != nil {
			return fmt.Errorf("%w: invalid hex digest for %s", ErrInvalidChecksum, fields[1])
		}
		if _, exists := expected[rel]; exists {
			return fmt.Errorf("%w: duplicate checksum entry for %s", ErrInvalidChecksum, fields[1])
		}
		expected[rel] = digest
	}

	expectedFiles := map[string]bool{}
	for _, name := range fixtureNames {
		if name != "checksums.txt" {
			expectedFiles[filepath.FromSlash(name)] = true
		}
	}
	if len(expected) != len(expectedFiles) {
		return fmt.Errorf("%w: checksums.txt has %d entries, want %d", ErrInvalidChecksum, len(expected), len(expectedFiles))
	}
	for name := range expected {
		if !expectedFiles[name] {
			return fmt.Errorf("%w: checksums.txt contains unexpected file %s", ErrInvalidChecksum, name)
		}
	}
	for name, want := range expected {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("%w: cannot read %s: %v", ErrInvalidChecksum, name, err)
		}
		sum := sha256.Sum256(data)
		got := hex.EncodeToString(sum[:])
		if got != want {
			return fmt.Errorf("%w: checksum mismatch for %s", ErrInvalidChecksum, name)
		}
	}
	return nil
}

func isSHA256(value string) bool {
	if !strings.HasPrefix(value, "sha256:") {
		return false
	}
	digest := strings.TrimPrefix(value, "sha256:")
	if len(digest) != 64 {
		return false
	}
	_, err := hex.DecodeString(digest)
	return err == nil
}

func floatsClose(left, right float64) bool {
	return math.Abs(left-right) <= 1e-6
}

func finiteFloat64(name string, value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return fmt.Errorf("%w: %s is not finite", ErrInvalidContract, name)
	}
	return nil
}

func contractError(field, got, want string) error {
	return fmt.Errorf("%w: %s is %q, want %q", ErrInvalidContract, field, got, want)
}
