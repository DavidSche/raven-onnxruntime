package anomaly

import (
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
	}
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
			if err := verifyRMPChecksums(filepath.Dir(cfg.ModelPath)); err != nil {
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
	err = verifyRMPChecksums(dir)
	if !errors.Is(err, ErrInvalidChecksum) || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate entry error = %v, want ErrInvalidChecksum mentioning duplicate", err)
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
