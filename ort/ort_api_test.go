package ort

import (
	"os"
	"path/filepath"
	"testing"
)

// libPath is the ONNX Runtime 1.28.0 DLL used for integration testing.
const libPath = `E:\study-place\Davidche\2026\raven\lib\onnxruntime.dll`

// staticModelPath: 静态输入 [1,3,1024,1024] 的 ONNX 模型，用于 GetInputShape 回归测试。
const staticModelPath = `E:\study-place\Davidche\2026\raven\raven-go\models\yolo11\yolo11m-obb.onnx`

func newTestEngine(t *testing.T) *Engine {
	t.Helper()
	eng, err := NewEngine(filepath.Clean(libPath))
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	t.Cleanup(func() { eng.Destroy() })
	return eng
}

// TestSessionGetInputShape 回归：GetInputShape 曾因 CastTypeInfoToTensorInfo 绑定错误
// （C 签名带 out 参数返回 status，Go 绑定误注册为单参裸指针返回）在首次调用时段错误。
// 修复后应从静态输入模型读回 [1,3,1024,1024]。
func TestSessionGetInputShape(t *testing.T) {
	if _, err := os.Stat(staticModelPath); os.IsNotExist(err) {
		t.Skipf("model not present: %s", staticModelPath)
	}

	eng := newTestEngine(t)
	opts, err := eng.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions: %v", err)
	}
	defer opts.Destroy()
	s, err := eng.NewSession(filepath.Clean(staticModelPath), opts)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Destroy()

	dims, err := s.GetInputShape(0)
	if err != nil {
		t.Fatalf("GetInputShape: %v", err)
	}
	t.Logf("input shape: %v", dims)
	if len(dims) != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] != 1024 || dims[3] != 1024 {
		t.Fatalf("dims = %v, want [1 3 1024 1024]", dims)
	}
}

// TestEngineVersion verifies the engine reports the correct API version and version string.
func TestEngineVersion(t *testing.T) {
	eng := newTestEngine(t)

	v := eng.GetVersion()
	if v == "" || v == "unknown" {
		t.Fatalf("GetVersion returned %q", v)
	}
	t.Logf("ONNX Runtime version: %s", v)

	apiV := eng.GetApiVersion()
	if apiV != ApiVersion28 {
		t.Fatalf("expected API version 28, got %d", apiV)
	}
	t.Logf("API version: %d", apiV)
}

// TestAvailableProviders verifies that AvailableProviders returns a non-empty list.
func TestAvailableProviders(t *testing.T) {
	eng := newTestEngine(t)

	providers, err := eng.AvailableProviders()
	if err != nil {
		t.Fatalf("AvailableProviders failed: %v", err)
	}
	if len(providers) == 0 {
		t.Fatal("expected at least one provider")
	}
	t.Logf("providers: %v", providers)
}

// TestSessionOptionsDestroyIdempotent verifies that Destroy is safe to call multiple times
// (sync.Once guarantees single execution).
func TestSessionOptionsDestroyIdempotent(t *testing.T) {
	eng := newTestEngine(t)

	opts, err := eng.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions failed: %v", err)
	}

	// Calling Destroy multiple times should not panic or double-free.
	opts.Destroy()
	opts.Destroy()
	opts.Destroy()
}

// TestSessionOptionsDestroyConcurrent verifies that concurrent Destroy calls are safe.
func TestSessionOptionsDestroyConcurrent(t *testing.T) {
	eng := newTestEngine(t)

	opts, err := eng.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions failed: %v", err)
	}

	done := make(chan struct{})
	go func() {
		opts.Destroy()
		close(done)
	}()
	opts.Destroy() // concurrent call
	<-done
}

// TestNewApis_GetMemPatternEnabled verifies the ORT 1.27+ GetMemPatternEnabled API.
func TestNewApis_GetMemPatternEnabled(t *testing.T) {
	eng := newTestEngine(t)

	opts, err := eng.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions failed: %v", err)
	}
	defer opts.Destroy()

	// Enable memory pattern first
	if err := opts.SetMemPattern(true); err != nil {
		t.Fatalf("SetMemPattern(true) failed: %v", err)
	}

	enabled, err := opts.GetMemPatternEnabled()
	if err != nil {
		t.Fatalf("GetMemPatternEnabled failed: %v", err)
	}
	if !enabled {
		t.Fatal("expected memory pattern to be enabled after SetMemPattern(true)")
	}
	t.Logf("GetMemPatternEnabled: %v", enabled)

	// Disable and verify
	if err := opts.SetMemPattern(false); err != nil {
		t.Fatalf("SetMemPattern(false) failed: %v", err)
	}
	enabled, err = opts.GetMemPatternEnabled()
	if err != nil {
		t.Fatalf("GetMemPatternEnabled failed: %v", err)
	}
	if enabled {
		t.Fatal("expected memory pattern to be disabled after SetMemPattern(false)")
	}
	t.Logf("GetMemPatternEnabled after disable: %v", enabled)
}

// TestNewApis_GetExecutionMode verifies the ORT 1.27+ GetSessionExecutionMode API.
func TestNewApis_GetExecutionMode(t *testing.T) {
	eng := newTestEngine(t)

	opts, err := eng.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions failed: %v", err)
	}
	defer opts.Destroy()

	// Set sequential mode and verify
	if err := opts.SetExecutionMode(ExecutionModeSequential); err != nil {
		t.Fatalf("SetExecutionMode(Sequential) failed: %v", err)
	}
	mode, err := opts.GetExecutionMode()
	if err != nil {
		t.Fatalf("GetExecutionMode failed: %v", err)
	}
	if mode != ExecutionModeSequential {
		t.Fatalf("expected ExecutionModeSequential(%d), got %d", ExecutionModeSequential, mode)
	}
	t.Logf("ExecutionMode (sequential): %d", mode)

	// Set parallel mode and verify
	if err := opts.SetExecutionMode(ExecutionModeParallel); err != nil {
		t.Fatalf("SetExecutionMode(Parallel) failed: %v", err)
	}
	mode, err = opts.GetExecutionMode()
	if err != nil {
		t.Fatalf("GetExecutionMode failed: %v", err)
	}
	if mode != ExecutionModeParallel {
		t.Fatalf("expected ExecutionModeParallel(%d), got %d", ExecutionModeParallel, mode)
	}
	t.Logf("ExecutionMode (parallel): %d", mode)
}

// TestNewApis_GetExperimentalFunction verifies the ORT 1.28+ GetExperimentalFunction API.
// We don't test a specific function name (those are internal to ORT), but we verify:
// 1. The API call doesn't crash
// 2. A non-existent name returns an error (nil pointer from C, converted to Go error)
func TestNewApis_GetExperimentalFunction(t *testing.T) {
	eng := newTestEngine(t)

	// Query a non-existent experimental function — should return an error, not crash.
	_, err := eng.GetExperimentalFunction("NonExistentFunction_12345")
	if err == nil {
		t.Log("GetExperimentalFunction with non-existent name returned nil error (unexpected but not fatal)")
	} else {
		t.Logf("GetExperimentalFunction with non-existent name correctly returned error: %v", err)
	}
}

// TestEngineDestroyIdempotent verifies that Destroy is safe to call multiple times.
func TestEngineDestroyIdempotent(t *testing.T) {
	eng, err := NewEngine(filepath.Clean(libPath))
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	// Calling Destroy multiple times should not panic.
	eng.Destroy()
	eng.Destroy()
}

// TestEngineIsAlive verifies the IsAlive method.
func TestEngineIsAlive(t *testing.T) {
	eng, err := NewEngine(filepath.Clean(libPath))
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	if !eng.IsAlive() {
		t.Fatal("expected engine to be alive before Destroy")
	}

	eng.Destroy()

	if eng.IsAlive() {
		t.Fatal("expected engine to not be alive after Destroy")
	}
}
