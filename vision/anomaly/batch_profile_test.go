package anomaly

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestBatchProfileOptionsValidation(t *testing.T) {
	cfg := Config{
		RuntimeBatchMode: "single",
		RuntimeMaxBatch:  1,
		InputLimits:      InputLimitsConfig{MaxBatch: 1},
	}

	if _, err := normalizeBatchProfileOptions(BatchProfileOptions{TimedIterations: 1}, 0, cfg); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("zero items error = %v, want ErrInvalidConfig", err)
	}
	if _, err := normalizeBatchProfileOptions(BatchProfileOptions{TimedIterations: 0}, 1, cfg); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("zero timed iterations error = %v, want ErrInvalidConfig", err)
	}
	if _, err := normalizeBatchProfileOptions(BatchProfileOptions{TimedIterations: 1, WarmupIterations: -1}, 1, cfg); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("negative warmup error = %v, want ErrInvalidConfig", err)
	}
	if _, err := normalizeBatchProfileOptions(BatchProfileOptions{
		TimedIterations: 1,
		BatchSizes:      []int{2},
	}, 1, cfg); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("batch above max error = %v, want ErrInvalidConfig", err)
	}
	if _, err := normalizeBatchProfileOptions(BatchProfileOptions{
		TimedIterations: 1,
		BatchSizes:      []int{1, 1},
	}, 1, cfg); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("duplicate batch error = %v, want ErrInvalidConfig", err)
	}
	normalized, err := normalizeBatchProfileOptions(BatchProfileOptions{
		TimedIterations: 1,
		BatchSizes:      []int{1},
	}, 1, cfg)
	if err != nil {
		t.Fatalf("valid options error = %v", err)
	}
	if len(normalized.BatchSizes) != 1 || normalized.BatchSizes[0] != 1 {
		t.Fatalf("normalized batch sizes = %v, want [1]", normalized.BatchSizes)
	}
}

func TestBatchProfileRejectsMixedDeploymentContext(t *testing.T) {
	cfg := gatedFixtureConfig(t, "padim-r18")
	firstCtx := gatedRuntimeContext(cfg)
	secondCtx := gatedRuntimeContext(cfg)
	secondCtx.DeploymentBindingID = "other-deployment"
	items := []AnomalyBatchItem{
		{Image: gradientImage(cfg.InputSize), Ctx: firstCtx},
		{Image: gradientImage(cfg.InputSize), Ctx: secondCtx},
	}
	if _, err := prepareBatchProfileItems(items, cfg); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("prepareBatchProfileItems() error = %v, want ErrInvalidConfig", err)
	}
}

func TestAnomalyBatchProfileMatchesSingleFrame(t *testing.T) {
	for _, variant := range []string{"padim-r18", "efficientad-s"} {
		t.Run(variant, func(t *testing.T) {
			cfg := batchEligibleFixtureConfig(t, variant)
			engine, err := NewEngine(cfg)
			if err != nil {
				t.Fatalf("NewEngine() error = %v", err)
			}
			defer engine.Destroy()

			items := []AnomalyBatchItem{
				{Image: gradientImage(cfg.InputSize), Ctx: gatedRuntimeContext(cfg)},
				{Image: gradientImage(cfg.InputSize), Ctx: gatedRuntimeContext(cfg)},
			}
			items[1].Ctx.StreamID = "batch-profile-stream-2"

			single, err := engine.PredictAnomaly(items[0].Image, items[0].Ctx)
			if err != nil {
				t.Fatalf("PredictAnomaly() error = %v", err)
			}
			prepared, err := prepareBatchProfileItems(items, cfg)
			if err != nil {
				t.Fatalf("prepareBatchProfileItems() error = %v", err)
			}
			batched, err := runBatchedAnomaly(engine, prepared, 2)
			if err != nil {
				t.Fatalf("runBatchedAnomaly() error = %v", err)
			}
			if len(batched) != 2 {
				t.Fatalf("batch results = %d, want 2", len(batched))
			}
			for index, result := range batched {
				if result == nil || math.IsNaN(float64(result.Score)) {
					t.Fatalf("batch result %d is invalid", index)
				}
				if result.ObserveOnly {
					t.Fatalf("batch result %d unexpectedly observe-only", index)
				}
				if single.Score != result.Score {
					t.Fatalf("score mismatch: single=%v batch[%d]=%v", single.Score, index, result.Score)
				}
			}

			report, err := engine.ProfileBatch(items, BatchProfileOptions{
				BatchSizes:        []int{1, 2},
				WarmupIterations:  1,
				TimedIterations:   3,
				IncludeConcurrent: true,
			})
			if err != nil {
				t.Fatalf("ProfileBatch() error = %v", err)
			}
			if report == nil || len(report.Metrics) != 4 {
				t.Fatalf("ProfileBatch() returned %v, want 4 metrics", report)
			}
			for _, metric := range report.Metrics {
				if metric.TimedIterations != 3 || metric.FramesPerIteration != 2 {
					t.Fatalf("unexpected profile dimensions: %+v", metric)
				}
				if metric.WallP50MS <= 0 || metric.FramesPerSecond <= 0 {
					t.Fatalf("non-positive timing in metric: %+v", metric)
				}
				if math.IsNaN(metric.FrameP50MS) || metric.FrameP50MS < 0 {
					t.Fatalf("invalid frame latency: %+v", metric)
				}
			}
			foundBatch := false
			for _, metric := range report.Metrics {
				if metric.Mode == BatchProfileTensorBatch && metric.BatchSize == 2 {
					foundBatch = true
				}
			}
			if !foundBatch {
				t.Fatalf("ProfileBatch() did not include tensor batch 2: %+v", report.Metrics)
			}
		})
	}
}
func batchEligibleFixtureConfig(t *testing.T, variant string) Config {
	t.Helper()
	source := filepath.Join("..", "..", "fixtures", "anomaly", variant)
	destination := t.TempDir()

	for _, name := range fixtureNames {
		if name == "checksums.txt" {
			continue
		}
		sourcePath := filepath.Join(source, filepath.FromSlash(name))
		destinationPath := filepath.Join(destination, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(destinationPath), 0o755); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(sourcePath)
		if err != nil {
			t.Fatal(err)
		}
		if name == "manifest.json" {
			var manifest map[string]any
			if err := json.Unmarshal(data, &manifest); err != nil {
				t.Fatal(err)
			}
			params := manifest["params"].(map[string]any)
			params["deployment_profile"] = "development"
			runtimeProfile := params["runtime_profile"].(map[string]any)
			inputLimits := runtimeProfile["input_limits"].(map[string]any)
			inputLimits["max_batch"] = 2
			data, err = json.MarshalIndent(manifest, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(destinationPath, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	var checksums []byte
	for _, name := range fixtureNames {
		if name == "checksums.txt" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(destination, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(data)
		checksums = append(checksums, []byte("sha256:"+hex.EncodeToString(digest[:])+" "+name+"\n")...)
	}
	if err := os.WriteFile(filepath.Join(destination, "checksums.txt"), checksums, 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(destination)
	if err != nil {
		t.Fatalf("LoadConfig(%s) error = %v", destination, err)
	}
	cfg.OnnxRuntimeLibPath = gatedLibraryPath(t)
	cfg.ExecutionProviderPolicy = map[string]string{"cpu": "required"}
	return cfg
}
