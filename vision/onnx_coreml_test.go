package vision

// ============================================================================
// CoreML EP 接入回归（vision 层 UseCoreML 配置链）
//
// 覆盖：
//   - C1（error-path，跨平台可跑）：UseCoreML=true 且运行时无 CoreMLExecutionProvider
//     → New() 显式报错（fail-fast 设计，与 CUDA 链同语义），且单例状态回滚干净
//   - C2（gated，Apple 环境才跑）：UseCoreML=true 且 CoreML EP 可用 → New() 成功
//     （Windows/CI 上无 CoreML EP，自动 skip）
//   - C3（gated）：UseCoreML=false 显式关闭 → 不报错（回归既有 CPU 路径不受污染）
//
// 依赖真实 ONNX Runtime 动态库（与 onnx_engine_state_test.go 同源 fixture；
// Windows 主力开发机为 GPU 变体 DLL——无 CoreML EP，C1 常规执行、C2 跳过）。
// ============================================================================

import (
	"slices"
	"strings"
	"testing"

	"github.com/DavidSche/raven-onnxruntime/ort"
)

// hasCoreMLProvider 探测当前运行时是否编入 CoreML EP。
func hasCoreMLProvider(t *testing.T, libPath string) bool {
	t.Helper()
	// 用一个独立的 OnnxConfig 走 New() 会污染单例；这里直接借 ort.Engine 探测。
	// vision 包内无法直接构造 ort.Engine（小写构造器），复用 New() 的探测路径：
	// 先保存单例，探测后恢复。
	restore := saveEngineState()
	defer restore()

	cfg := &OnnxConfig{OnnxRuntimeLibPath: libPath, UseCuda: false, UseCoreML: false}
	if err := cfg.New(); err != nil {
		t.Skipf("cannot init engine for provider probe: %v", err)
	}
	provs, err := cfg.OnnxEngine.AvailableProviders()
	if err != nil {
		t.Skipf("cannot list providers: %v", err)
	}
	return slices.Contains(provs, "CoreMLExecutionProvider")
}

// TestOnnxConfig_CoreMLRequestedButUnavailable —— C1 fail-fast 契约。
// 请求 CoreML 但运行时无该 EP：New() 必须显式报错（不静默降级），且
// engineState 回滚干净（createdEngine 分支），允许后续重试。
func TestOnnxConfig_CoreMLRequestedButUnavailable(t *testing.T) {
	libPath := onnxTestLibPath(t)
	if hasCoreMLProvider(t, libPath) {
		t.Skip("CoreML EP available on this runtime; unavailable-path covered by C2 gated test")
	}
	restore := saveEngineState()
	defer restore()

	cfg := &OnnxConfig{
		OnnxRuntimeLibPath: libPath,
		UseCoreML:          true,
	}
	err := cfg.New()
	if err == nil {
		t.Fatal("UseCoreML=true without CoreMLExecutionProvider should fail fast, got nil error")
	}
	if !strings.Contains(err.Error(), "CoreML requested but CoreMLExecutionProvider not detected") {
		t.Fatalf("unexpected error message: %v", err)
	}
	// fail-fast 不留半初始化状态：cfg.SessionOptions 不应被赋值
	if cfg.SessionOptions != nil {
		t.Error("SessionOptions should remain nil after fail-fast")
	}
	// 单例状态回滚干净：stale 引擎已被丢弃（下次 New 可重试）
	engineState.mu.Lock()
	alive := engineState.eng != nil && engineState.eng.IsAlive()
	engineState.mu.Unlock()
	if alive && engineState.path == libPath {
		// 允许复用型单例存活，但 SessionOptions 链路必须未被污染——
		// 这里只断言 New() 返回错误后的重试可用性
		retry := &OnnxConfig{OnnxRuntimeLibPath: libPath}
		if err := retry.New(); err != nil {
			t.Errorf("retry New() after fail-fast should succeed: %v", err)
		}
		retry.Destroy()
	}
}

// TestOnnxConfig_CoreMLAvailable —— C2 gated：仅 Apple 平台 + CoreML EP 构建执行。
func TestOnnxConfig_CoreMLAvailable(t *testing.T) {
	libPath := onnxTestLibPath(t)
	if !hasCoreMLProvider(t, libPath) {
		t.Skip("CoreML EP not compiled into this ONNX Runtime build (Windows GPU/CPU builds)")
	}
	restore := saveEngineState()
	defer restore()

	cfg := &OnnxConfig{
		OnnxRuntimeLibPath: libPath,
		UseCoreML:          true,
		CoreMLOpts:         &ort.CoreMLProviderOptions{MLComputeUnits: "cpuAndGPU"},
	}
	if err := cfg.New(); err != nil {
		t.Fatalf("UseCoreML=true with CoreML EP should succeed: %v", err)
	}
	defer cfg.Destroy()
	if cfg.SessionOptions == nil {
		t.Fatal("SessionOptions should be set after successful New()")
	}
}

// TestOnnxConfig_CoreMLDisabled —— C3：显式关闭走纯 CPU 路径（既有语义回归）。
func TestOnnxConfig_CoreMLDisabled(t *testing.T) {
	libPath := onnxTestLibPath(t)
	restore := saveEngineState()
	defer restore()

	cfg := &OnnxConfig{
		OnnxRuntimeLibPath: libPath,
		UseCoreML:          false,
		UseCuda:            false,
	}
	if err := cfg.New(); err != nil {
		t.Fatalf("UseCoreML=false should not error: %v", err)
	}
	defer cfg.Destroy()
	if cfg.SessionOptions == nil {
		t.Fatal("SessionOptions should be set")
	}
}
