package vision

import (
	"fmt"
	"slices"
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
	// Default is ort.DefaultApiVersion (currently 28).
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
func (cfg *OnnxConfig) New() (err error) {
	if cfg.OnnxRuntimeLibPath == "" {
		return fmt.Errorf("OnnxRuntimeLibPath must not be empty")
	}

	engineState.mu.Lock()
	defer engineState.mu.Unlock()

	// stale 检测先行：单例引擎被外部 Destroy（IsAlive=false）后，无论请求路径是否
	// 相同，都必须丢弃 stale 引用再走后续逻辑——否则 M1 会拿 stale 引擎的旧 path
	// 拒绝用户换路径重建（Destroy 后引擎 handle 已清零，仅残留 Go 指针与 path）。
	if engineState.eng != nil && !engineState.eng.IsAlive() {
		engineState.eng = nil
		engineState.path = ""
	}

	// M1: refuse to silently overwrite a previously initialized Engine with a
	// different library path; otherwise the old Engine would never be Destroyed.
	if engineState.eng != nil && engineState.path != cfg.OnnxRuntimeLibPath {
		return fmt.Errorf("ONNX Runtime engine already initialized with a different library path: %q (requested %q)", engineState.path, cfg.OnnxRuntimeLibPath)
	}

	if engineState.eng != nil && engineState.path == cfg.OnnxRuntimeLibPath {
		// already initialized with the same path, reuse directly
		cfg.OnnxEngine = engineState.eng
	}

	// createdEngine tracks whether this call created a brand-new Engine so that,
	// on a later failure, we can roll back the singleton state (M2) and let a
	// retry start from a clean slate instead of leaving a half-initialized Engine.
	createdEngine := false
	defer func() {
		if err != nil && createdEngine {
			engineState.eng = nil
			engineState.path = ""
			cfg.OnnxEngine = nil
		}
	}()

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
		createdEngine = true
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
		if !slices.Contains(providers, "CUDAExecutionProvider") {
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

// Destroy releases SessionOptions and frees dynamically allocated C handle resources.
func (cfg *OnnxConfig) Destroy() {
	if cfg.SessionOptions != nil {
		cfg.SessionOptions.Destroy()
		cfg.SessionOptions = nil
	}
}
