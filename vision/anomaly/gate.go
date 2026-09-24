package anomaly

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// RuntimeContract is the frozen v1 runtime contract name.
const RuntimeContract = "anomaly-v1"

// ExportProfile is the only production export profile for v1.
const ExportProfile = "anomaly-export-v1"

// Production validated variants are explicitly enumerated.
const (
	ValidatedEfficientAdSmall = "efficientad-s-256"
	ValidatedPadimR18         = "padim-r18-256"
)

// ErrM1G0NotSatisfied blocks engine construction until the required real
// contract fixture layout is available.
var ErrM1G0NotSatisfied = errors.New("M1-G0 gate is not satisfied: real anomaly contract fixtures are required")

// ErrInvalidConfig rejects a configuration that does not match anomaly-v1.
var ErrInvalidConfig = errors.New("invalid anomaly configuration")

// ErrInvalidContract rejects inconsistent RMP runtime contracts.
var ErrInvalidContract = errors.New("invalid anomaly runtime contract")

// ErrInvalidChecksum rejects an RMP whose packaged files do not match
// checksums.txt.
var ErrInvalidChecksum = errors.New("anomaly RMP checksum validation failed")

// ErrRuntimeContextMismatch blocks inference for a hard-gated context.
var ErrRuntimeContextMismatch = errors.New("anomaly runtime context mismatch")

// ErrEngineDestroyed is returned after the lifecycle owner has quiesced and
// destroyed the engine.
var ErrEngineDestroyed = errors.New("anomaly engine is destroyed")

// fixtureNames is the cross-language minimum RMP fixture layout. Deep artifact
// and checksum validation remain the responsibility of anomaly_export's
// validate_m1_gate CLI and LoadConfig.
var fixtureNames = []string{
	"model.onnx",
	"manifest.json",
	"labels.txt",
	"checksums.txt",
	"reference/threshold_report.json",
	"reference/model_card.json",
	"reference/golden_outputs.npz",
}

// CheckFixtureGate verifies that both required M1-G0 fixture directories are
// present. It intentionally checks file presence only; artifact semantics,
// checksums and numeric evidence are validated by anomaly_export's machine
// gate and by LoadConfig.
func CheckFixtureGate(fixturesDir string) error {
	if fixturesDir == "" {
		return fmt.Errorf("%w: fixtures directory is empty", ErrM1G0NotSatisfied)
	}
	requiredVariants := []struct {
		directory string
		name      string
	}{
		{directory: "efficientad-s", name: ValidatedEfficientAdSmall},
		{directory: "padim-r18", name: ValidatedPadimR18},
	}
	for _, variant := range requiredVariants {
		variantDir := filepath.Join(fixturesDir, variant.directory)
		info, err := os.Stat(variantDir)
		if err != nil {
			return fmt.Errorf("%w: missing fixture directory %s", ErrM1G0NotSatisfied, variantDir)
		}
		if !info.IsDir() {
			return fmt.Errorf("%w: fixture path is not a directory: %s", ErrM1G0NotSatisfied, variantDir)
		}
		for _, name := range fixtureNames {
			path := filepath.Join(variantDir, filepath.FromSlash(name))
			info, err := os.Stat(path)
			if err != nil {
				return fmt.Errorf("%w: missing fixture file %s", ErrM1G0NotSatisfied, path)
			}
			if info.IsDir() {
				return fmt.Errorf("%w: fixture file is a directory: %s", ErrM1G0NotSatisfied, path)
			}
			if info.Size() == 0 {
				return fmt.Errorf("%w: fixture file is empty: %s", ErrM1G0NotSatisfied, path)
			}
		}
	}
	return nil
}
