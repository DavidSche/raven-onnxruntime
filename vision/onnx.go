package vision

import (
	"fmt"
	"sync"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
)

type OnnxConfig struct {
	SessionOptions *ort.SessionOptions
	OnnxEngine     *ort.Engine

	// Required parameters
	OnnxRuntimeLibPath string // path to onnxruntime.dll (or .so, .dylib)
	// Optional parameters
	UseCuda    bool // (optional) whether to enable CUDA
	NumThreads int  // (optional) ONNX thread count, default determined by CPU core count

	// ApiVersion specifies the requested ONNX Runtime C API version.
	// Default is ort.DefaultApiVersion (currently 26).
	// If the specified version is unavailable, NewEngine will automatically downgrade to the highest version supported by the library.
	// Can also be overridden via the ORT_API_VERSION environment variable.
	ApiVersion ort.ApiVersion

	// EnableCpuMemArena controls the ONNX memory arena strategy.
	// false (default): disable memory arena, slightly slower inference, but memory is returned to the OS immediately after Destroy
	// true: enable memory arena, fastest inference, but memory is cached for reuse after Destroy
	EnableCpuMemArena bool
}

// engineState holds the global ONNX Engine singleton state.
// Uses Mutex instead of sync.Once so that DLL load failures can be retried rather than permanently stuck.
var engineState struct {
	mu   sync.Mutex
	eng  *ort.Engine
	path string // successfully loaded DLL path
}

// New initializes the ONNX environment.
//
// The global Engine is a singleton (only one DLL loaded per process).
// Key difference from sync.Once: DLL load failures can be retried, not permanently stuck.
func (cfg *OnnxConfig) New() error {
	if cfg.OnnxRuntimeLibPath == "" {
		return fmt.Errorf("OnnxRuntimeLibPath must not be empty")
	}

	engineState.mu.Lock()
	defer engineState.mu.Unlock()

	if engineState.eng != nil && engineState.path == cfg.OnnxRuntimeLibPath {
		if engineState.eng.IsAlive() {
			// already initialized with the same path, reuse directly
			cfg.OnnxEngine = engineState.eng
		} else {
			// stale singleton after Destroy(); drop it and recreate below
			engineState.eng = nil
			engineState.path = ""
		}
	}

	if cfg.OnnxEngine == nil {
		var engineOpts []ort.EngineOption
		if cfg.ApiVersion > 0 {
			engineOpts = append(engineOpts, ort.WithApiVersion(cfg.ApiVersion))
		}
		eng, err := ort.NewEngine(cfg.OnnxRuntimeLibPath, engineOpts...)
		if err != nil {
			return fmt.Errorf("failed to initialize ONNX Engine (dll=%s): %w", cfg.OnnxRuntimeLibPath, err)
		}
		engineState.eng = eng
		engineState.path = cfg.OnnxRuntimeLibPath
		cfg.OnnxEngine = eng
	}

	providers, err := cfg.OnnxEngine.AvailableProviders()
	if err != nil {
		return fmt.Errorf("failed to get ONNX providers: %w", err)
	}
	ortlog.Infow("onnx runtime providers detected",
		"providers", providers,
		"useCudaRequested", cfg.UseCuda)

	// create session options (set thread count)
	options, err := cfg.OnnxEngine.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("failed to create SessionOptions: %w", err)
	}
	if cfg.NumThreads > 0 {
		if err := options.SetIntraOpNumThreads(int32(cfg.NumThreads)); err != nil {
			options.Destroy()
			return fmt.Errorf("failed to set intra-op thread count: %w", err)
		}
	}
	if err := options.SetInterOpNumThreads(1); err != nil {
		options.Destroy()
		return fmt.Errorf("failed to set inter-op thread count: %w", err)
	}
	if err := options.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
		options.Destroy()
		return fmt.Errorf("failed to set execution mode: %w", err)
	}
	if err := options.SetMemPattern(true); err != nil {
		options.Destroy()
		return fmt.Errorf("failed to set memory pattern: %w", err)
	}
	if err := options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelAll); err != nil {
		options.Destroy()
		return fmt.Errorf("failed to set graph optimization level: %w", err)
	}

	// set memory arena strategy
	if err := options.SetCpuMemArena(cfg.EnableCpuMemArena); err != nil {
		options.Destroy()
		return fmt.Errorf("failed to set CPU memory arena: %w", err)
	}

	// enable CUDA (on failure, degrade to CPU without interrupting initialization)
	if cfg.UseCuda {
		if !containsString(providers, "CUDAExecutionProvider") {
			options.Destroy()
			return fmt.Errorf("CUDA requested but CUDAExecutionProvider not detected in ONNX Runtime")
		}
		if err := options.EnableCUDA(); err != nil {
			options.Destroy()
			return fmt.Errorf("failed to enable CUDA: %w", err)
		}
	}

	cfg.SessionOptions = options
	return nil
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

// Destroy releases SessionOptions and frees dynamically allocated C handle resources.
func (cfg *OnnxConfig) Destroy() {
	if cfg.SessionOptions != nil {
		cfg.SessionOptions.Destroy()
		cfg.SessionOptions = nil
	}
}
