package anomaly

import (
	"errors"
	"image"
	"math"
	"testing"
)

func postprocessingConfig(score float32, mapThreshold float32) Config {
	return Config{
		InputSize:      image.Point{X: 4, Y: 4},
		ScoreThreshold: score,
		MapThreshold:   mapThreshold,
		MinRegionArea:  2,
		MaxRegions:     2,
		MapDownsample:  1,
		ScoreSemantics: ScoreSemanticsMapMax,
		ScoreContract: ScoreContractConfig{
			Semantics: ScoreSemanticsMapMax,
			Assertion: ScoreAssertionConfig{Enabled: true, Atol: 1e-6, Rtol: 1e-3},
		},
		RegionDecisionMode:    "independent",
		ThresholdVersion:      "threshold-v1",
		RegionPolicyVersion:   "region-policy-v1",
		DecisionPolicyVersion: "decision-v1",
	}
}

func sampleAnomalyMap() []float32 {
	values := make([]float32, 16)
	set := func(x, y int, value float32) { values[y*4+x] = value }
	set(0, 0, 0.6)
	set(1, 0, 0.8)
	set(2, 0, 0.7)
	set(1, 1, 0.65)
	set(3, 3, 0.60)
	set(2, 3, 0.55)
	return values
}

func TestPostprocessDataBuildsSortedRegionsAndQualificationStats(t *testing.T) {
	cfg := postprocessingConfig(0.8, 0.5)
	result, err := postprocessData(0.8, sampleAnomalyMap(), 4, 4, cfg, preprocessResult{originalWidth: 8, originalHeight: 8})
	if err != nil {
		t.Fatalf("postprocessData() error = %v", err)
	}
	if !result.IsAnomalous || result.Score != 0.8 || result.ObserveOnly {
		t.Fatalf("unexpected image decision: %+v", result)
	}
	if len(result.Regions) != 2 {
		t.Fatalf("got %d regions, want 2", len(result.Regions))
	}
	first, second := result.Regions[0], result.Regions[1]
	if first.Box != image.Rect(0, 0, 6, 4) || first.Area != 4 || first.PeakScore != 0.8 || math.Abs(float64(first.MeanScore-0.6875)) > 1e-6 {
		t.Fatalf("first region = %+v", first)
	}
	if second.Box != image.Rect(4, 6, 8, 8) || second.Area != 2 || second.PeakScore != 0.6 || math.Abs(float64(second.MeanScore-0.575)) > 1e-6 {
		t.Fatalf("second region = %+v", second)
	}
	if result.DebugStats.ThresholdedPixelCount != 6 || result.DebugStats.RawComponentCount != 2 || result.DebugStats.FilteredByArea != 0 {
		t.Fatalf("unexpected debug stats: %+v", result.DebugStats)
	}
	if len(result.DebugStats.Regions) != 2 || result.DebugStats.Regions[0].ScoreP90 != 0.8 || result.DebugStats.Regions[0].ScoreP95 != 0.8 {
		t.Fatalf("unexpected qualification stats: %+v", result.DebugStats.Regions)
	}
	if len(result.DebugStats.Regions[0].SampledScores) != 4 {
		t.Fatalf("got %d sampled scores, want 4", len(result.DebugStats.Regions[0].SampledScores))
	}
}

func TestPostprocessDataFiltersSmallRegionsAndLimitsMaxRegions(t *testing.T) {
	cfg := postprocessingConfig(0.8, 0.5)
	cfg.MinRegionArea = 3
	result, err := postprocessData(0.8, sampleAnomalyMap(), 4, 4, cfg, preprocessResult{originalWidth: 8, originalHeight: 8})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Regions) != 1 || result.DebugStats.FilteredByArea != 1 {
		t.Fatalf("area filtering failed: regions=%d stats=%+v", len(result.Regions), result.DebugStats)
	}

	cfg.MinRegionArea = 1
	cfg.MaxRegions = 1
	result, err = postprocessData(0.8, sampleAnomalyMap(), 4, 4, cfg, preprocessResult{originalWidth: 8, originalHeight: 8})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Regions) != 1 || result.DebugStats.FilteredByMaxRegions != 1 {
		t.Fatalf("max region filtering failed: regions=%d stats=%+v", len(result.Regions), result.DebugStats)
	}
}

func TestPostprocessDataZeroRegionsAndModelDefinedScore(t *testing.T) {
	cfg := postprocessingConfig(0.8, 0.9)
	cfg.ScoreSemantics = ScoreSemanticsModelDefined
	cfg.ScoreContract.Semantics = ScoreSemanticsModelDefined
	cfg.ScoreContract.Assertion.Enabled = false
	result, err := postprocessData(0.8, make([]float32, 16), 4, 4, cfg, preprocessResult{originalWidth: 8, originalHeight: 8})
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsAnomalous || len(result.Regions) != 0 {
		t.Fatalf("image-level anomaly with zero regions failed: %+v", result)
	}
}

func TestPostprocessDataRejectsMapMaxMismatchAndNonFiniteMap(t *testing.T) {
	cfg := postprocessingConfig(0.8, 0.5)
	_, err := postprocessData(0.7, sampleAnomalyMap(), 4, 4, cfg, preprocessResult{originalWidth: 8, originalHeight: 8})
	if !errors.Is(err, ErrInvalidContract) {
		t.Fatalf("map max mismatch error = %v", err)
	}
	invalid := sampleAnomalyMap()
	invalid[0] = float32(math.NaN())
	if _, err := postprocessData(float32(math.NaN()), invalid, 4, 4, cfg, preprocessResult{originalWidth: 8, originalHeight: 8}); err == nil {
		t.Fatal("non-finite map accepted")
	}
}

func TestDownsampleMapAveragePooling(t *testing.T) {
	source := []float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15,
	}
	output := downsampleMap(source, 4, 4, 2)
	expected := []float32{2.5, 4.5, 10.5, 12.5}
	for i, value := range expected {
		if output[i] != value {
			t.Fatalf("output[%d] = %v, want %v", i, output[i], value)
		}
	}
	output = downsampleMap(source, 4, 4, 1)
	for i, value := range source {
		if output[i] != value {
			t.Fatalf("identity downsample changed value at %d", i)
		}
	}
}

func TestMapModelBoxClampsAndNeverReturnsEmptyBox(t *testing.T) {
	if got := mapModelBox(image.Rect(-2, -2, 1, 1), 4, 4, 8, 8); got != image.Rect(0, 0, 2, 2) {
		t.Fatalf("negative box = %v", got)
	}
	if got := mapModelBox(image.Rect(3, 3, 4, 4), 4, 4, 8, 8); got != image.Rect(6, 6, 8, 8) {
		t.Fatalf("edge box = %v", got)
	}
}
