package anomaly

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func writeFixtureLayout(t *testing.T, root, variant string) {
	t.Helper()
	dir := filepath.Join(root, variant)
	if err := os.MkdirAll(filepath.Join(dir, "reference"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range fixtureNames {
		path := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.WriteFile(path, []byte("fixture"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCheckFixtureGateAcceptsCompleteRealFixtureLayout(t *testing.T) {
	root := t.TempDir()
	for _, variant := range []string{"efficientad-s", "padim-r18"} {
		writeFixtureLayout(t, root, variant)
	}
	if err := CheckFixtureGate(root); err != nil {
		t.Fatalf("CheckFixtureGate() = %v, want nil", err)
	}
}

func TestCheckFixtureGateRejectsMissingRequiredFile(t *testing.T) {
	root := t.TempDir()
	for _, variant := range []string{"efficientad-s", "padim-r18"} {
		writeFixtureLayout(t, root, variant)
	}
	if err := os.Remove(filepath.Join(root, "padim-r18", "reference", "golden_outputs.npz")); err != nil {
		t.Fatal(err)
	}
	err := CheckFixtureGate(root)
	if !errors.Is(err, ErrM1G0NotSatisfied) {
		t.Fatalf("CheckFixtureGate() error = %v, want ErrM1G0NotSatisfied", err)
	}
}

func TestCheckFixtureGateRejectsMissingLayoutAndEmptyFile(t *testing.T) {
	if err := CheckFixtureGate(t.TempDir()); !errors.Is(err, ErrM1G0NotSatisfied) {
		t.Fatalf("missing layout error = %v, want ErrM1G0NotSatisfied", err)
	}
	root := t.TempDir()
	for _, variant := range []string{"efficientad-s", "padim-r18"} {
		writeFixtureLayout(t, root, variant)
	}
	if err := os.WriteFile(filepath.Join(root, "efficientad-s", "model.onnx"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckFixtureGate(root); !errors.Is(err, ErrM1G0NotSatisfied) {
		t.Fatalf("empty file error = %v, want ErrM1G0NotSatisfied", err)
	}
}
