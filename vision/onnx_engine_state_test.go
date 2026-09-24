package vision

// engineState 单例语义测试（生命周期审计，见 AGENTS.md「并发字段锁归属」）。
//
// 覆盖 onnx.go New() 的三条单例路径：
//   - M1：不同 library path 拒绝（避免静默覆盖导致旧 Engine 永不销毁）
//   - stale 重建：单例引擎被 Destroy（IsAlive=false）后，New() 应丢弃并重建
//   - 失败回滚：创建新引擎失败时，engineState 回滚为干净状态，允许重试
//
// 依赖真实 ONNX Runtime DLL（lib/onnxruntime.dll，与 ort 包测试同源）；DLL 缺失
// 时跳过，不无条件 panic。测试操作进程级单例，结束时恢复初始状态以免污染同包
// 其他测试。

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// onnxTestLibPath 定位仓库 lib/onnxruntime.dll（相对本包两级的仓库根）。
func onnxTestLibPath(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("..", "..", "lib", "onnxruntime.dll"))
	if err != nil {
		t.Skipf("cannot resolve DLL path: %v", err)
	}
	if _, err := os.Stat(abs); os.IsNotExist(err) {
		t.Skipf("onnxruntime.dll not present: %s", abs)
	}
	return abs
}

// saveEngineState 快照单例，返回恢复函数。
func saveEngineState() func() {
	eng, path := engineState.eng, engineState.path
	return func() {
		engineState.mu.Lock()
		engineState.eng, engineState.path = eng, path
		engineState.mu.Unlock()
	}
}

// TestEngineState_EmptyLibPath 空路径必须报错，且不触碰单例。
func TestEngineState_EmptyLibPath(t *testing.T) {
	restore := saveEngineState()
	defer restore()

	cfg := &OnnxConfig{}
	err := cfg.New()
	if err == nil {
		t.Fatal("New() with empty OnnxRuntimeLibPath must fail")
	}
	if engineState.eng != nil || engineState.path != "" {
		t.Fatalf("empty-path failure mutated singleton: eng=%v path=%q", engineState.eng, engineState.path)
	}
}

// TestEngineState_DifferentPathRejected M1：不同 library path 必须拒绝，
// 防止静默覆盖导致旧 Engine 永不销毁。
func TestEngineState_DifferentPathRejected(t *testing.T) {
	restore := saveEngineState()
	defer restore()

	lib := onnxTestLibPath(t)

	cfg1 := &OnnxConfig{OnnxRuntimeLibPath: lib}
	if err := cfg1.New(); err != nil {
		t.Fatalf("first New() failed: %v", err)
	}
	defer cfg1.Destroy()

	cfg2 := &OnnxConfig{OnnxRuntimeLibPath: filepath.Join(t.TempDir(), "other.dll")}
	err := cfg2.New()
	if err == nil {
		t.Fatal("New() with a different library path must be rejected (M1)")
	}
	if !strings.Contains(err.Error(), "different library path") {
		t.Errorf("M1 rejection err = %q, want contains 'different library path'", err)
	}
	engineState.mu.Lock()
	path := engineState.path
	engineState.mu.Unlock()
	if path != lib {
		t.Fatalf("singleton path changed to %q, want %q", path, lib)
	}
}

// TestEngineState_StaleRebuild 单例引擎被 Destroy（IsAlive=false）后，
// New() 应丢弃 stale 引用并重建新引擎。
func TestEngineState_StaleRebuild(t *testing.T) {
	restore := saveEngineState()
	defer restore()

	lib := onnxTestLibPath(t)

	cfg1 := &OnnxConfig{OnnxRuntimeLibPath: lib}
	if err := cfg1.New(); err != nil {
		t.Fatalf("first New() failed: %v", err)
	}
	defer cfg1.Destroy()
	engineState.mu.Lock()
	oldEng := engineState.eng
	engineState.mu.Unlock()

	// 模拟单例引擎被外部 Destroy：handle 清零，IsAlive 变 false。
	oldEng.Destroy()
	if oldEng.IsAlive() {
		t.Fatal("engine should not be alive after Destroy")
	}

	cfg2 := &OnnxConfig{OnnxRuntimeLibPath: lib}
	if err := cfg2.New(); err != nil {
		t.Fatalf("New() after stale must rebuild: %v", err)
	}
	defer cfg2.Destroy()

	if engineState.eng == nil || !engineState.eng.IsAlive() {
		t.Fatal("stale path must create a fresh alive engine")
	}
	if engineState.eng == oldEng {
		t.Fatal("stale path must not reuse the destroyed engine pointer")
	}
	if engineState.path != lib {
		t.Fatalf("singleton path = %q, want %q", engineState.path, lib)
	}
}

// TestEngineState_StaleCleanupThenRetry stale 引擎被 Destroy 后：换路径创建失败时
// 单例必须保持干净（stale 预清理，非 M2 回滚——见下），且后续可用原路径重试成功。
//
// 覆盖边界（如实说明）：此测试钉住的是 stale 预清理 + 失败后可重试的回归行为。
// 真正的 M2 机制（createdEngine=true 后 AvailableProviders/NewSessionOptions 等
// 后续步骤失败触发回滚）需要 mock Engine 才能驱动，本测试未覆盖该路径。
func TestEngineState_StaleCleanupThenRetry(t *testing.T) {
	restore := saveEngineState()
	defer restore()

	// 先正常初始化，制造"已有单例"环境。
	lib := onnxTestLibPath(t)
	cfg0 := &OnnxConfig{OnnxRuntimeLibPath: lib}
	if err := cfg0.New(); err != nil {
		t.Fatalf("setup New() failed: %v", err)
	}
	defer cfg0.Destroy()

	// 让单例变 stale（Destroy 引擎），使下一次 New() 走"重建"路径，再注入失败。
	engineState.mu.Lock()
	stale := engineState.eng
	engineState.mu.Unlock()
	stale.Destroy()

	badPath := filepath.Join(t.TempDir(), "no-such-onnxruntime.dll")
	cfgBad := &OnnxConfig{OnnxRuntimeLibPath: badPath}
	err := cfgBad.New()
	if err == nil {
		t.Fatal("New() with invalid library path must fail")
	}
	t.Logf("expected failure: %v", err)

	// stale 预清理断言：失败后单例应保持干净（不再指向 stale 引擎），
	// 后续重试从干净状态开始。
	engineState.mu.Lock()
	eng, path := engineState.eng, engineState.path
	engineState.mu.Unlock()
	if eng != nil || path != "" {
		t.Fatalf("failed New() must leave singleton clean, got eng=%v path=%q", eng, path)
	}

	// 清理后可重试成功。
	cfgRetry := &OnnxConfig{OnnxRuntimeLibPath: lib}
	if err := cfgRetry.New(); err != nil {
		t.Fatalf("retry New() after cleanup failed: %v", err)
	}
	defer cfgRetry.Destroy()
	engineState.mu.Lock()
	retryEng := engineState.eng
	engineState.mu.Unlock()
	if retryEng == nil || !retryEng.IsAlive() {
		t.Fatal("retry after cleanup must produce a live engine")
	}
}

// TestEngineState_ConcurrentNew 并发 New() 同一路径：单例部分经 engineState.mu
// 串行，必须全部成功且共享同一引擎（CI -race 门控）。
func TestEngineState_ConcurrentNew(t *testing.T) {
	restore := saveEngineState()
	defer restore()

	lib := onnxTestLibPath(t)

	const n = 8
	cfgs := make([]*OnnxConfig, n)
	errs := make([]error, n)
	done := make(chan struct{})
	for i := 0; i < n; i++ {
		cfgs[i] = &OnnxConfig{OnnxRuntimeLibPath: lib}
		go func(i int) {
			defer func() { done <- struct{}{} }()
			errs[i] = cfgs[i].New()
		}(i)
	}
	for i := 0; i < n; i++ {
		<-done
	}
	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent New() #%d failed: %v", i, err)
		}
	}
	for i, cfg := range cfgs {
		defer cfg.Destroy()
		if cfg.OnnxEngine == nil {
			t.Fatalf("concurrent New() #%d left OnnxEngine nil", i)
		}
		if cfg.OnnxEngine != engineState.eng {
			t.Fatalf("concurrent New() #%d got a different engine than the singleton", i)
		}
	}
	if engineState.eng == nil || !engineState.eng.IsAlive() {
		t.Fatal("singleton must be alive after concurrent New()")
	}
}
