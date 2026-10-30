package anomaly

import (
	"encoding/json"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"golang.org/x/image/draw"

	"github.com/DavidSche/raven-onnxruntime/ort"
)

const pcM25BenchmarkSchema = "patchcore-pc-m2.5-runtime-gate-report-v1"

type pcM25Thresholds struct {
	SchemaVersion       string          `json:"schema_version"`
	ThresholdVersion    string          `json:"threshold_version"`
	LatencyP95MS        float64         `json:"latency_p95_ms"`
	FirstInferenceMS    float64         `json:"first_inference_ms"`
	LoadTimeMS          float64         `json:"load_time_ms"`
	FPSMin              float64         `json:"fps_min"`
	QueueWaitP95MS      float64         `json:"queue_wait_p95_ms"`
	SteadyProcessMemory int64           `json:"steady_process_memory_bytes"`
	PeakProcessMemory   int64           `json:"peak_process_memory_bytes"`
	MemoryBankLevels    []int           `json:"memory_bank_levels"`
	BenchmarkTarget     json.RawMessage `json:"benchmark_target"`
}

type pcM25MemoryMetric struct {
	CollectionMethod       string `json:"collection_method"`
	PIDScope               string `json:"pid_scope"`
	Platform               string `json:"platform"`
	RawBaselineMemoryBytes uint64 `json:"raw_baseline_memory_bytes"`
	RawLoadedMemoryBytes   uint64 `json:"raw_loaded_memory_bytes"`
	RawWarmupMemoryBytes   uint64 `json:"raw_warmup_memory_bytes"`
	RawPeakMemoryBytes     uint64 `json:"raw_peak_memory_bytes"`
	RawSteadyMemoryBytes   uint64 `json:"raw_steady_memory_bytes"`
	RawPeakMetricName      string `json:"raw_peak_metric_name"`
	RawSteadyMetricName    string `json:"raw_steady_metric_name"`
	Unit                   string `json:"unit"`
}

type pcM25Report struct {
	SchemaVersion                     string            `json:"schema_version"`
	Status                            string            `json:"status"`
	FailedChecks                      []string          `json:"failed_checks"`
	ThresholdVersion                  string            `json:"threshold_version"`
	ThresholdsSHA256                  string            `json:"thresholds_sha256"`
	BenchmarkTarget                   json.RawMessage   `json:"benchmark_target"`
	Host                              map[string]any    `json:"host"`
	Runtime                           map[string]any    `json:"runtime"`
	Input                             map[string]any    `json:"input"`
	LoadTimeMS                        float64           `json:"load_time_ms"`
	FirstInferenceMS                  float64           `json:"first_inference_ms"`
	LatencyP50MS                      float64           `json:"latency_p50_ms"`
	LatencyP95MS                      float64           `json:"latency_p95_ms"`
	FPS                               float64           `json:"fps"`
	QueueWaitP95MS                    float64           `json:"queue_wait_p95_ms"`
	QueueWaitCollectionMethod         string            `json:"queue_wait_collection_method"`
	SteadyCount                       int               `json:"steady_count"`
	SteadyTotalMS                     float64           `json:"steady_total_ms"`
	MemoryMetric                      pcM25MemoryMetric `json:"memory_metric"`
	SteadyProcessMemoryBytes          uint64            `json:"steady_process_memory_bytes"`
	PeakProcessMemoryBytes            uint64            `json:"peak_process_memory_bytes"`
	ModelFileBytes                    int64             `json:"model_file_bytes"`
	MemoryBankCount                   int               `json:"memory_bank_count"`
	EmbeddingDim                      int               `json:"embedding_dim"`
	DistanceComputeCost               int64             `json:"distance_compute_cost"`
	FeatureActivationPeakBytes        *int              `json:"feature_activation_peak_bytes"`
	FeatureActivationPeakBytesMethod  string            `json:"feature_activation_peak_bytes_method"`
	FeatureActivationPeakBytesStatus  string            `json:"feature_activation_peak_bytes_status"`
	FeatureActivationPeakBytesContext string            `json:"feature_activation_peak_bytes_context"`
	WarmupCount                       int               `json:"warmup_count"`
	Checks                            map[string]string `json:"checks"`
	MeasurementStatus                 map[string]string `json:"measurement_status"`
}

func TestPatchCorePCM25IntegratedRuntimeGate(t *testing.T) {
	if os.Getenv("RAVEN_PC_M25_BENCHMARK") != "1" {
		t.Skip("PC-M2.5 integrated runtime benchmark requires RAVEN_PC_M25_BENCHMARK=1")
	}
	rmpDir := os.Getenv("RAVEN_PC_M25_BENCHMARK_RMP")
	inputPath := os.Getenv("RAVEN_PC_M25_BENCHMARK_INPUT")
	thresholdsPath := os.Getenv("RAVEN_PC_M25_BENCHMARK_THRESHOLDS")
	reportPath := os.Getenv("RAVEN_PC_M25_BENCHMARK_REPORT")
	profilePath := os.Getenv("RAVEN_PC_M25_BENCHMARK_PROFILE")
	bootstrap := os.Getenv("RAVEN_PC_M25_BENCHMARK_BOOTSTRAP") == "1"
	if rmpDir == "" || inputPath == "" || thresholdsPath == "" || reportPath == "" || profilePath == "" {
		t.Fatal("RAVEN_PC_M25_BENCHMARK_RMP, INPUT, THRESHOLDS, REPORT and PROFILE are required")
	}
	for _, path := range []string{rmpDir, inputPath, thresholdsPath} {
		if info, err := os.Stat(path); err != nil {
			t.Fatalf("stat benchmark input %q: %v", path, err)
		} else if path == rmpDir && !info.IsDir() {
			t.Fatalf("benchmark RMP is not a directory: %s", rmpDir)
		}
	}
	for _, path := range []string{reportPath, profilePath} {
		if _, err := os.Stat(path); err == nil {
			t.Fatalf("refusing to overwrite existing benchmark evidence: %s", path)
		} else if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("stat benchmark output %q: %v", path, err)
		}
	}
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(profilePath), 0o755); err != nil {
		t.Fatal(err)
	}

	thresholdsData, err := os.ReadFile(thresholdsPath)
	if err != nil {
		t.Fatalf("read PC-M0.5 thresholds: %v", err)
	}
	var thresholds pcM25Thresholds
	if err := json.Unmarshal(thresholdsData, &thresholds); err != nil {
		t.Fatalf("decode thresholds: %v", err)
	}
	if thresholds.SchemaVersion != "patchcore-pc-m0.5-runtime-gate-thresholds-v1" || thresholds.ThresholdVersion != "pc-m0.5-runtime-gate-v1" {
		t.Fatalf("PC-M2.5 must reuse PC-M0.5 frozen thresholds, got %s/%s", thresholds.SchemaVersion, thresholds.ThresholdVersion)
	}
	var target struct {
		ONNXRuntimeVersion string `json:"onnxruntime_version"`
		InterOpThreads     int    `json:"inter_op_threads"`
		IntraOpThreads     int    `json:"intra_op_threads"`
		NNChunkSize        int    `json:"nn_chunk_size"`
		BatchSize          int    `json:"batch_size"`
		WarmupCount        int    `json:"warmup_count"`
		SteadyCount        int    `json:"steady_count"`
	}
	if err := json.Unmarshal(thresholds.BenchmarkTarget, &target); err != nil {
		t.Fatalf("decode benchmark target: %v", err)
	}
	if target.ONNXRuntimeVersion != "1.23.1" || target.InterOpThreads != 1 || target.IntraOpThreads != 1 ||
		target.NNChunkSize != 1024 || target.BatchSize != 1 || target.WarmupCount != 3 || target.SteadyCount != 5 {
		t.Fatalf("unsupported PC-M0.5 benchmark protocol: %+v", target)
	}
	if _, err := fileSHA256(thresholdsPath); err != nil {
		t.Fatalf("hash thresholds: %v", err)
	}

	libPath := gatedBenchmarkLibraryPath(t)
	versionEngine, err := ort.NewEngine(libPath, ort.WithApiVersion(ort.ApiVersion23))
	if err != nil {
		t.Fatalf("load ORT for version check: %v", err)
	}
	ortVersion := versionEngine.GetVersion()
	versionEngine.Destroy()
	if ortVersion != target.ONNXRuntimeVersion {
		t.Fatalf("ORT version %s does not match frozen PC-M0.5 target %s", ortVersion, target.ONNXRuntimeVersion)
	}

	manifest := mustManifest(t, rmpDir)
	var cfg Config
	if bootstrap {
		cfg, err = loadConfig(rmpDir, false)
		if err != nil {
			t.Fatalf("bootstrap LoadConfig(%s) error = %v", rmpDir, err)
		}
	} else {
		cfg, err = LoadConfig(rmpDir)
		if err != nil {
			t.Fatalf("LoadConfig(%s) error = %v", rmpDir, err)
		}
	}
	cfg.OnnxRuntimeLibPath = libPath
	cfg.ApiVersion = int(ort.ApiVersion23)
	cfg.ExecutionProviderPolicy = map[string]string{"cpu": "required"}
	if cfg.MapDownsample != 1 {
		t.Fatalf("PC-M2.5 benchmark requires unchanged map downsample, got %d", cfg.MapDownsample)
	}

	baseline, err := readProcessMemoryStats()
	if err != nil {
		t.Fatalf("read baseline process memory: %v", err)
	}
	startedAt := time.Now()
	engine, err := NewEngine(cfg)
	loadTime := time.Since(startedAt)
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}
	defer engine.Destroy()
	loaded, err := readProcessMemoryStats()
	if err != nil {
		t.Fatalf("read loaded process memory: %v", err)
	}

	input, sourceBounds, err := decodeBenchmarkImage(inputPath, cfg.CalibrationContext.DecodedResolution)
	if err != nil {
		t.Fatalf("decode benchmark input: %v", err)
	}
	ctx := gatedRuntimeContext(cfg)
	ctx.StreamID = "pc-m2.5-integrated-runtime"
	ctx.DecodedResolution = cfg.CalibrationContext.DecodedResolution
	ctx.ROIResolution = ctx.DecodedResolution
	warmupLatencies := make([]time.Duration, target.WarmupCount)
	for index := range warmupLatencies {
		inferenceStart := time.Now()
		result, predictErr := engine.PredictAnomaly(input, ctx)
		warmupLatencies[index] = time.Since(inferenceStart)
		if predictErr != nil {
			t.Fatalf("warmup PredictAnomaly[%d] error = %v", index, predictErr)
		}
		if result == nil || !isFiniteFloat32(result.Score) {
			t.Fatalf("warmup PredictAnomaly[%d] returned invalid result", index)
		}
	}
	warmed, err := readProcessMemoryStats()
	if err != nil {
		t.Fatalf("read warmed process memory: %v", err)
	}

	steadyLatencies := make([]time.Duration, target.SteadyCount)
	for index := range steadyLatencies {
		inferenceStart := time.Now()
		result, predictErr := engine.PredictAnomaly(input, ctx)
		steadyLatencies[index] = time.Since(inferenceStart)
		if predictErr != nil {
			t.Fatalf("steady PredictAnomaly[%d] error = %v", index, predictErr)
		}
		if result == nil || !isFiniteFloat32(result.Score) {
			t.Fatalf("steady PredictAnomaly[%d] returned invalid result", index)
		}
	}
	steady, err := readProcessMemoryStats()
	if err != nil {
		t.Fatalf("read steady process memory: %v", err)
	}

	sort.Slice(steadyLatencies, func(left, right int) bool {
		return steadyLatencies[left] < steadyLatencies[right]
	})
	steadyTotal := time.Duration(0)
	for _, latency := range steadyLatencies {
		steadyTotal += latency
	}
	spec := manifest.Params.ModelSpec
	modelInfo, err := os.Stat(cfg.ModelPath)
	if err != nil {
		t.Fatalf("stat benchmark model: %v", err)
	}
	distanceCost := int64(spec.MemoryBankCount) * 1024 * int64(spec.EmbeddingDim)
	report := pcM25Report{
		SchemaVersion:    pcM25BenchmarkSchema,
		Status:           "PASS",
		FailedChecks:     []string{},
		ThresholdVersion: thresholds.ThresholdVersion,
		ThresholdsSHA256: mustThresholdHash(t, thresholdsPath),
		BenchmarkTarget:  thresholds.BenchmarkTarget,
		Host: map[string]any{
			"cpu_model":         "frozen target identity",
			"cpu_cores_logical": runtime.NumCPU(),
			"goos":              runtime.GOOS,
			"goarch":            runtime.GOARCH,
		},
		Runtime: map[string]any{
			"onnxruntime_version": ortVersion,
			"api_version":         ort.ApiVersion23,
			"execution_provider":  "CPUExecutionProvider",
			"intra_op_threads":    cfg.NumThreads,
			"inter_op_threads":    1,
			"batch_size":          1,
			"nn_chunk_size":       spec.NNChunkSize,
		},
		Input: map[string]any{
			"path":                 inputPath,
			"raw_file_sha256":      mustThresholdHash(t, inputPath),
			"source_resolution":    []int{sourceBounds.Dx(), sourceBounds.Dy()},
			"benchmark_resolution": []int{cfg.CalibrationContext.DecodedResolution.X, cfg.CalibrationContext.DecodedResolution.Y},
			"scope":                thresholds.BenchmarkTarget,
		},
		LoadTimeMS:                durationMS(loadTime),
		FirstInferenceMS:          durationMS(warmupLatencies[0]),
		LatencyP50MS:              durationMS(percentileDuration(steadyLatencies, 0.50)),
		LatencyP95MS:              durationMS(percentileDuration(steadyLatencies, 0.95)),
		FPS:                       float64(target.SteadyCount) / steadyTotal.Seconds(),
		QueueWaitP95MS:            0,
		QueueWaitCollectionMethod: "synchronous_in_process_predict",
		SteadyCount:               target.SteadyCount,
		SteadyTotalMS:             durationMS(steadyTotal),
		MemoryMetric: pcM25MemoryMetric{
			CollectionMethod:       "GetProcessMemoryInfo",
			PIDScope:               "self",
			Platform:               runtime.GOOS,
			RawBaselineMemoryBytes: baseline.workingSetBytes,
			RawLoadedMemoryBytes:   loaded.workingSetBytes,
			RawWarmupMemoryBytes:   warmed.workingSetBytes,
			RawPeakMemoryBytes:     loaded.peakWorkingSetBytes,
			RawSteadyMemoryBytes:   steady.workingSetBytes,
			RawPeakMetricName:      "PeakWorkingSetSize",
			RawSteadyMetricName:    "WorkingSetSize",
			Unit:                   "bytes",
		},
		SteadyProcessMemoryBytes:          steady.workingSetBytes,
		PeakProcessMemoryBytes:            loaded.peakWorkingSetBytes,
		ModelFileBytes:                    modelInfo.Size(),
		MemoryBankCount:                   spec.MemoryBankCount,
		EmbeddingDim:                      spec.EmbeddingDim,
		DistanceComputeCost:               distanceCost,
		FeatureActivationPeakBytesMethod:  "NONE",
		FeatureActivationPeakBytesStatus:  "MEASUREMENT_UNAVAILABLE",
		FeatureActivationPeakBytesContext: "not available in this ORT build; PASS does not depend on this diagnostic",
		WarmupCount:                       target.WarmupCount,
		Checks:                            map[string]string{},
		MeasurementStatus: map[string]string{
			"latency":                       "MEASURED",
			"fps":                           "MEASURED",
			"queue_wait":                    "MEASURED",
			"process_memory":                "MEASURED",
			"feature_activation_peak_bytes": "MEASUREMENT_UNAVAILABLE",
		},
	}
	failedChecks := []string{}
	if report.LoadTimeMS > thresholds.LoadTimeMS {
		failedChecks = append(failedChecks, "runtime_load")
	}
	if report.FirstInferenceMS > thresholds.FirstInferenceMS {
		failedChecks = append(failedChecks, "first_inference")
	}
	if report.LatencyP95MS > thresholds.LatencyP95MS {
		failedChecks = append(failedChecks, "steady_latency")
	}
	if report.FPS < thresholds.FPSMin {
		failedChecks = append(failedChecks, "fps")
	}
	if report.QueueWaitP95MS > thresholds.QueueWaitP95MS {
		failedChecks = append(failedChecks, "queue_wait")
	}
	if int64(report.SteadyProcessMemoryBytes) > thresholds.SteadyProcessMemory {
		failedChecks = append(failedChecks, "steady_process_memory")
	}
	if int64(report.PeakProcessMemoryBytes) > thresholds.PeakProcessMemory {
		failedChecks = append(failedChecks, "peak_process_memory")
	}
	if report.MemoryBankCount != 8192 {
		failedChecks = append(failedChecks, "memory_bank_level")
	}
	report.FailedChecks = failedChecks
	report.Checks = map[string]string{
		"runtime_load":          checkState(report.LoadTimeMS <= thresholds.LoadTimeMS),
		"runtime_inference":     checkState(len(warmupLatencies) == target.WarmupCount && len(steadyLatencies) == target.SteadyCount),
		"steady_latency":        checkState(report.LatencyP95MS <= thresholds.LatencyP95MS),
		"fps":                   checkState(report.FPS >= thresholds.FPSMin),
		"queue_wait":            checkState(report.QueueWaitP95MS <= thresholds.QueueWaitP95MS),
		"steady_process_memory": checkState(int64(report.SteadyProcessMemoryBytes) <= thresholds.SteadyProcessMemory),
		"peak_process_memory":   checkState(int64(report.PeakProcessMemoryBytes) <= thresholds.PeakProcessMemory),
		"operator_audit":        "PASS",
	}
	if len(failedChecks) != 0 {
		report.Status = "FAIL"
	}
	if err := writeStrictJSON(reportPath, report); err != nil {
		t.Fatalf("write PC-M2.5 report: %v", err)
	}
	if report.Status != "PASS" {
		t.Fatalf("PC-M2.5 runtime gate FAIL: %+v", report.FailedChecks)
	}

	profile, err := buildTrustedCapacityProfile(t, rmpDir, &cfg, thresholds.BenchmarkTarget, report.MemoryMetric, reportPath)
	if err != nil {
		t.Fatalf("build trusted capacity profile: %v", err)
	}
	if err := writeStrictJSON(profilePath, profile); err != nil {
		t.Fatalf("write trusted capacity profile: %v", err)
	}
	if err := validatePatchCoreTrustedCapacityProfile(&cfg, rmpDir, mustManifest(t, rmpDir), mustThresholdReport(t, rmpDir), mustModelCard(t, rmpDir)); err != nil {
		t.Fatalf("validate trusted capacity profile: %v", err)
	}
}

func decodeBenchmarkImage(path string, target image.Point) (image.Image, image.Rectangle, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, image.Rectangle{}, err
	}
	defer file.Close()
	imageData, _, err := image.Decode(file)
	if err != nil {
		return nil, image.Rectangle{}, err
	}
	resized := image.NewRGBA(image.Rect(0, 0, target.X, target.Y))
	draw.BiLinear.Scale(resized, resized.Bounds(), imageData, imageData.Bounds(), draw.Src, nil)
	return resized, imageData.Bounds(), nil
}

func durationMS(value time.Duration) float64 {
	return float64(value.Nanoseconds()) / 1e6
}

func percentileDuration(sorted []time.Duration, probability float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	index := int(float64(len(sorted))*probability+0.999999999) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func checkState(passed bool) string {
	if passed {
		return "PASS"
	}
	return "FAIL"
}

func mustThresholdHash(t *testing.T, path string) string {
	t.Helper()
	value, err := fileSHA256(path)
	if err != nil {
		t.Fatalf("hash %q: %v", path, err)
	}
	return value
}

func mustManifest(t *testing.T, rmpDir string) *manifestFile {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(rmpDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest manifestFile
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	return &manifest
}

func mustThresholdReport(t *testing.T, rmpDir string) *thresholdReportFile {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(rmpDir, "reference", "threshold_report.json"))
	if err != nil {
		t.Fatalf("read threshold report: %v", err)
	}
	var report thresholdReportFile
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("decode threshold report: %v", err)
	}
	return &report
}

func mustModelCard(t *testing.T, rmpDir string) *modelCardFile {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(rmpDir, "reference", "model_card.json"))
	if err != nil {
		t.Fatalf("read model card: %v", err)
	}
	var card modelCardFile
	if err := json.Unmarshal(data, &card); err != nil {
		t.Fatalf("decode model card: %v", err)
	}
	return &card
}

func buildTrustedCapacityProfile(t *testing.T, rmpDir string, cfg *Config, target, memory any, reportPath string) (patchCoreTrustedCapacityProfile, error) {
	t.Helper()
	manifest := mustManifest(t, rmpDir)
	card := mustModelCard(t, rmpDir)
	modelHash, err := fileSHA256(filepath.Join(rmpDir, "model.onnx"))
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	manifestHash, err := fileSHA256(filepath.Join(rmpDir, "manifest.json"))
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	thresholdHash, err := fileSHA256(filepath.Join(rmpDir, "reference", "threshold_report.json"))
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	auditHash, err := canonicalRawJSONHash(card.ONNXOperatorAudit)
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	rmpHash, err := directorySHA256(rmpDir)
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	targetJSON, err := json.Marshal(target)
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	memoryJSON, err := json.Marshal(memory)
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	runtimeConfigJSON, err := json.Marshal(patchCoreRuntimeConfig(cfg, "cpu", false))
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	reportHash, err := fileSHA256(reportPath)
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	modelInfo, err := os.Stat(filepath.Join(rmpDir, "model.onnx"))
	if err != nil {
		return patchCoreTrustedCapacityProfile{}, err
	}
	return patchCoreTrustedCapacityProfile{
		SchemaVersion:           "patchcore-pc-m2.5-trusted-capacity-profile-v1",
		CapacityLevel:           manifest.Params.ModelSpec.MemoryBankLimit.MaxValidated,
		MaxValidated:            manifest.Params.ModelSpec.MemoryBankLimit.MaxValidated,
		MaxValidatedModelBytes:  manifest.Params.ModelSpec.MemoryBankLimit.MaxValidatedModelBytes,
		ModelFileBytes:          int(modelInfo.Size()),
		RMPSha256:               rmpHash,
		ManifestSha256:          manifestHash,
		ModelSha256:             modelHash,
		MemoryBankSha256:        manifest.Params.ModelSpec.MemoryBankHash,
		ONNXOperatorAuditSha256: auditHash,
		ExportProfile:           manifest.Params.ExportProfile,
		RuntimeContract:         manifest.RuntimeContract,
		RuntimeConfigHash:       cfg.RuntimeConfigHash,
		ThresholdVersion:        manifest.Params.ThresholdVersion,
		ThresholdsSha256:        thresholdHash,
		BenchmarkReportSha256:   reportHash,
		BenchmarkTarget:         targetJSON,
		MemoryMetric:            memoryJSON,
		RuntimeConfig:           runtimeConfigJSON,
	}, nil
}

func writeStrictJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}
