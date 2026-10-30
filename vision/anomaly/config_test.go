package anomaly

import (
	"encoding/json"
	"errors"
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadFixtureConfig(t *testing.T, variant string) Config {
	t.Helper()
	cfg, err := LoadConfig(filepath.Join("..", "..", "fixtures", "anomaly", variant))
	if err != nil {
		t.Fatalf("LoadConfig(%s) failed: %v", variant, err)
	}
	return cfg
}

func TestLoadConfigAcceptsProductionValidatedFixtures(t *testing.T) {
	expected := []struct {
		variant          string
		modelKind        string
		validatedVariant string
		scoreThreshold   float32
		mapThreshold     float32
		normalize        string
		inputSize        image.Point
	}{
		{variant: "efficientad-s", modelKind: "efficient-ad", validatedVariant: ValidatedEfficientAdSmall, scoreThreshold: 0.28832966, mapThreshold: 0.20989762, normalize: "divide_255", inputSize: image.Point{X: 256, Y: 256}},
		{variant: "padim-r18", modelKind: "padim", validatedVariant: ValidatedPadimR18, scoreThreshold: 48.214661, mapThreshold: 64.004524, normalize: "imagenet", inputSize: image.Point{X: 256, Y: 256}},
		{variant: "patchcore-r18-pcm25-v3", modelKind: "patchcore", validatedVariant: ValidatedPatchcoreR18, scoreThreshold: 9.837318, mapThreshold: 16.197048, normalize: "imagenet", inputSize: image.Point{X: 256, Y: 256}},
	}
	patchCoreExpectedNames := append(append([]string{}, fixtureNames...), patchCoreExtraFixtureNames...)
	for _, want := range expected {
		t.Run(want.variant, func(t *testing.T) {
			cfg := loadFixtureConfig(t, want.variant)
			if cfg.ModelKind != want.modelKind {
				t.Fatalf("ModelKind = %q, want %q", cfg.ModelKind, want.modelKind)
			}
			if cfg.ValidatedVariant != want.validatedVariant {
				t.Fatalf("ValidatedVariant = %q, want %q", cfg.ValidatedVariant, want.validatedVariant)
			}
			if cfg.ScoreThreshold != want.scoreThreshold || cfg.MapThreshold != want.mapThreshold {
				t.Fatalf("thresholds = %v/%v, want %v/%v", cfg.ScoreThreshold, cfg.MapThreshold, want.scoreThreshold, want.mapThreshold)
			}
			if cfg.PreprocessContract.Normalize != want.normalize {
				t.Fatalf("Normalize = %q, want %q", cfg.PreprocessContract.Normalize, want.normalize)
			}
			if cfg.InputSize != want.inputSize {
				t.Fatalf("InputSize = %v, want %v", cfg.InputSize, want.inputSize)
			}
			if cfg.RuntimeBatchMode != "single" || cfg.RuntimeMaxBatch != 1 {
				t.Fatalf("runtime profile is not single-frame: %q/%d", cfg.RuntimeBatchMode, cfg.RuntimeMaxBatch)
			}
			expectedNames := fixtureNames
			if strings.HasPrefix(want.variant, "patchcore-r18") {
				expectedNames = patchCoreExpectedNames
			}
			if err := verifyRMPChecksums(filepath.Dir(cfg.ModelPath), expectedNames); err != nil {
				t.Fatalf("verifyRMPChecksums() = %v, want nil", err)
			}
		})
	}
}

func TestReadStrictJSONRejectsUnknownFieldsAndTrailingValue(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "invalid.json")
	if err := writeFile(path, []byte(`{"format_version":"1.0","unexpected":true}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := readStrictJSON[manifestFile](path); !errors.Is(err, ErrInvalidContract) || !strings.Contains(err.Error(), "unexpected") {
		t.Fatalf("unknown field error = %v, want ErrInvalidContract mentioning unexpected", err)
	}
	if err := writeFile(path, []byte(`{"format_version":"1.0"} {"task":"anomaly"}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := readStrictJSON[manifestFile](path); !errors.Is(err, ErrInvalidContract) || !strings.Contains(err.Error(), "trailing") {
		t.Fatalf("trailing value error = %v, want ErrInvalidContract mentioning trailing", err)
	}
}

func TestVerifyRMPChecksumsRejectsUnexpectedEntryAndMismatch(t *testing.T) {
	source := loadFixtureConfig(t, "efficientad-s")
	dir := t.TempDir()
	for _, name := range fixtureNames {
		data, err := readFile(filepath.Join(filepath.Dir(source.ModelPath), filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		destination := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := writeFile(destination, data); err != nil {
			t.Fatal(err)
		}
	}
	data, err := readFile(filepath.Join(dir, "checksums.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if err := writeFile(filepath.Join(dir, "checksums.txt"), append(data, []byte("sha256:"+strings.Repeat("0", 64)+"  labels.txt\n")...)); err != nil {
		t.Fatal(err)
	}
	err = verifyRMPChecksums(dir, fixtureNames)
	if !errors.Is(err, ErrInvalidChecksum) || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate entry error = %v, want ErrInvalidChecksum mentioning duplicate", err)
	}
}

func TestCheckPatchCoreFixtureGateAcceptsRealFixture(t *testing.T) {
	err := CheckPatchCoreFixtureGate(filepath.Join("..", "..", "fixtures", "anomaly"))
	if err != nil {
		t.Fatalf("CheckPatchCoreFixtureGate() = %v, want nil", err)
	}
}

func TestPatchCoreTrustedCapacityProfileFailsFastWithoutRealEvidence(t *testing.T) {
	manifest := manifestFile{ModelType: modelKindPatchcore, RuntimeContract: RuntimeContract}
	manifest.Params.ModelSpec.MemoryBankHash = "sha256:" + strings.Repeat("0", 64)
	err := validatePatchCoreTrustedCapacityProfile(&Config{}, t.TempDir(), &manifest, &thresholdReportFile{}, &modelCardFile{})
	if !errors.Is(err, ErrInvalidContract) || !strings.Contains(err.Error(), "PC-M2.5") {
		t.Fatalf("missing trusted profile error = %v, want PC-M2.5 fail-fast", err)
	}
}

func TestPatchCoreRuntimeConfigProfileCanonicalHash(t *testing.T) {
	cfg := Config{
		RuntimeBatchMode:        "single",
		RuntimeMaxBatch:         1,
		NumThreads:              1,
		EnableCpuMemArena:       false,
		ExecutionProviderPolicy: map[string]string{"cpu": "required"},
	}
	expectedConfig := patchCoreRuntimeConfig(&cfg, "cpu", false)
	expectedHash, err := canonicalRawJSONHash(expectedConfig)
	if err != nil {
		t.Fatalf("canonicalRawJSONHash() = %v", err)
	}
	manifest := manifestFile{ModelType: modelKindPatchcore}
	manifest.Params.RuntimeConfig = expectedConfig
	manifest.Params.RuntimeConfigHash = expectedHash
	encoded, err := json.Marshal(expectedConfig)
	if err != nil {
		t.Fatalf("Marshal() = %v", err)
	}
	profile := patchCoreTrustedCapacityProfile{RuntimeConfig: json.RawMessage(encoded), RuntimeConfigHash: expectedHash}
	if err := validatePatchCoreRuntimeConfigProfile(&cfg, &manifest, &profile); err != nil {
		t.Fatalf("validatePatchCoreRuntimeConfigProfile() = %v, want nil", err)
	}
	profile.RuntimeConfigHash = "sha256:" + strings.Repeat("0", 64)
	if err := validatePatchCoreRuntimeConfigProfile(&cfg, &manifest, &profile); !errors.Is(err, ErrInvalidContract) {
		t.Fatalf("mismatched runtime hash accepted")
	}
}

func TestRuntimeProfileAcceptsPatchCoreStaticBatchAndKeepsOthersDynamic(t *testing.T) {
	newConfig := func() (Config, manifestFile) {
		cfg := Config{
			RuntimeBatchMode:                "single",
			RuntimeMaxBatch:                 1,
			InputSize:                       image.Point{X: 256, Y: 256},
			ExecutionProviderFallbackMetric: "anomaly_ep_fallback_total",
			ExecutionProviderPolicy:         map[string]string{"cpu": providerRequired},
		}
		manifest := manifestFile{}
		manifest.Params.RuntimeProfile.BatchMode = "single"
		manifest.Params.RuntimeProfile.MaxRuntimeBatch = 1
		manifest.Params.RuntimeProfile.InputLimits.MinSide = 16
		manifest.Params.RuntimeProfile.InputLimits.MaxSide = 2048
		manifest.Params.RuntimeProfile.InputLimits.MaxPixels = 1048576
		manifest.Params.RuntimeProfile.InputLimits.MaxBatch = 1
		return cfg, manifest
	}

	patchcoreCfg, patchcoreManifest := newConfig()
	patchcoreManifest.ModelType = modelKindPatchcore
	patchcoreManifest.Params.OnnxInput.Batch.Kind = "static"
	patchcoreManifest.Params.OnnxInput.Batch.Size = 1
	if err := validateRuntimeProfile(&patchcoreCfg, &patchcoreManifest); err != nil {
		t.Fatalf("PatchCore static batch rejected: %v", err)
	}

	padimCfg, padimManifest := newConfig()
	padimManifest.ModelType = modelKindPadim
	padimManifest.Params.OnnxInput.Batch.Kind = "dynamic"
	if err := validateRuntimeProfile(&padimCfg, &padimManifest); err != nil {
		t.Fatalf("dynamic batch PaDiM rejected: %v", err)
	}
	padimManifest.Params.OnnxInput.Batch.Kind = "static"
	if err := validateRuntimeProfile(&padimCfg, &padimManifest); !errors.Is(err, ErrInvalidContract) {
		t.Fatalf("non-PatchCore static batch accepted")
	}
}

func TestPatchCoreRuntimeRejectsDivide255AndMissingConfigHash(t *testing.T) {
	cfg := loadFixtureConfig(t, "padim-r18")
	cfg.ModelKind = modelKindPatchcore
	cfg.PreprocessContract.Normalize = normalizeDivide255
	cfg.RuntimeConfigHash = "sha256:" + strings.Repeat("0", 64)
	if err := validateRuntimeConfig(&cfg); !errors.Is(err, ErrInvalidContract) || !strings.Contains(err.Error(), "ImageNet") {
		t.Fatalf("PatchCore divide_255 validation = %v, want ImageNet error", err)
	}
	cfg.PreprocessContract.Normalize = normalizeImageNet
	cfg.RuntimeConfigHash = ""
	if err := validateRuntimeConfig(&cfg); !errors.Is(err, ErrInvalidContract) || !strings.Contains(err.Error(), "runtime_config_hash") {
		t.Fatalf("PatchCore missing runtime hash validation = %v", err)
	}
}

func TestPatchCoreModelSpecRejectsMultiNeighborAndWrongPostprocess(t *testing.T) {
	manifest := manifestFile{ModelType: modelKindPatchcore}
	spec := &manifest.Params.ModelSpec
	spec.Backbone = "resnet18"
	spec.Variant = "r18"
	spec.EmbeddingDim = 384
	spec.Layers = []string{"layer2", "layer3"}
	spec.MemoryBankCount = 8
	spec.MemoryBankHash = "sha256:" + strings.Repeat("1", 64)
	spec.MemoryBankLimit.MaxValidated = 8
	spec.MemoryBankLimit.MaxValidatedModelBytes = 1024
	spec.CoresetSamplingRatio = 0.1
	spec.MapNumNeighbors = 1
	spec.EmbeddingUpsample = "bilinear"
	spec.AnomalyMapUpsample = "nearest"
	spec.AnomalyMapSigma = 4
	spec.AnomalyMapKernelSize = 33
	spec.AnomalyMapBlurBorderType = "reflect"
	spec.AnomalyMapBlurNormalize = true
	spec.NNChunkSize = 8
	report := thresholdReportFile{}
	identity := &report.ModelIdentity
	identity.MemoryBankHash = spec.MemoryBankHash
	identity.MemoryBankCount = spec.MemoryBankCount
	identity.MemoryBankLimit = spec.MemoryBankLimit
	identity.CoresetSamplingRatio = spec.CoresetSamplingRatio
	identity.MapNumNeighbors = 1
	identity.MapMaxSource = "final_model_anomaly_map"
	identity.ScoreSemantics = "map_max"
	identity.AnomalyMapSigma = 4
	identity.AnomalyMapKernelSize = 33
	identity.NNChunkSize = 8
	card := modelCardFile{}
	card.ONNXOperatorAudit.MemoryBankSHA256 = spec.MemoryBankHash
	card.ONNXOperatorAudit.MemoryBankCount = spec.MemoryBankCount
	card.ONNXOperatorAudit.RuntimeLoad = "PASS"
	card.ONNXOperatorAudit.RuntimeInference = "PASS"
	if err := validatePatchCoreModelSpec(&manifest, &report, &card); err != nil {
		t.Fatalf("valid PatchCore static spec rejected: %v", err)
	}
	spec.MapNumNeighbors = 9
	if err := validatePatchCoreModelSpec(&manifest, &report, &card); !errors.Is(err, ErrInvalidContract) {
		t.Fatalf("multi-neighbor PatchCore accepted: %v", err)
	}
}

func TestShapeEqualSupportsDynamicBatchAndDimensions(t *testing.T) {
	if !shapeEqual([]any{"B", float64(3), float64(256), float64(256)}, []any{"B", float64(3), float64(256), float64(256)}) {
		t.Fatal("equal dynamic shapes rejected")
	}
	if shapeEqual([]any{float64(1), float64(3), float64(256), float64(256)}, []any{"B", float64(3), float64(256), float64(256)}) {
		t.Fatal("static batch accepted as dynamic")
	}
}
