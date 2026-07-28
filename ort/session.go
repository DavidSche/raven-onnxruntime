package ort

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
)

type Session struct {
	handle              SessionHandle
	engine              *Engine
	InputNames          []string
	OutputNames         []string
	outputNameSlices    [][]byte
	outputNamePtrs      []unsafe.Pointer
	cachedInputNamePtrs sync.Map
	destroyOnce         sync.Once
}

type SessionOptions struct {
	handle      SessionOptionsHandle
	engine      *Engine
	destroyOnce sync.Once
}

func (e *Engine) NewSessionOptions() (*SessionOptions, error) {
	var h SessionOptionsHandle
	status := e.funcs.createSessionOptions(&h)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}
	return &SessionOptions{handle: h, engine: e}, nil
}

// SetIntraOpNumThreads sets the number of intra-op threads.
func (o *SessionOptions) SetIntraOpNumThreads(num int32) error {
	return o.engine.checkStatus(o.engine.funcs.setIntraOpNumThreads(o.handle, num))
}

// SetInterOpNumThreads sets the number of inter-op threads.
func (o *SessionOptions) SetInterOpNumThreads(num int32) error {
	return o.engine.checkStatus(o.engine.funcs.setInterOpNumThreads(o.handle, num))
}

// SetExecutionMode sets the session execution mode.
func (o *SessionOptions) SetExecutionMode(mode ExecutionMode) error {
	return o.engine.checkStatus(o.engine.funcs.setSessionExecutionMode(o.handle, mode))
}

// SetMemPattern controls ONNX memory pattern optimization.
func (o *SessionOptions) SetMemPattern(enabled bool) error {
	if enabled {
		return o.engine.checkStatus(o.engine.funcs.enableMemPattern(o.handle))
	}
	return o.engine.checkStatus(o.engine.funcs.disableMemPattern(o.handle))
}

// SetGraphOptimizationLevel sets the graph optimization level.
func (o *SessionOptions) SetGraphOptimizationLevel(level GraphOptimizationLevel) error {
	return o.engine.checkStatus(o.engine.funcs.setSessionGraphOptimizationLevel(o.handle, level))
}

// SetCpuMemArena controls the CPU memory arena strategy.
//
//	false: disable memory arena, slightly slower inference, but memory is returned to the OS immediately after Destroy
//	true: enable memory arena, fastest inference, but memory is cached for reuse after Destroy (default)
func (o *SessionOptions) SetCpuMemArena(useArena bool) error {
	if useArena {
		return o.engine.checkStatus(o.engine.funcs.enableCpuMemArena(o.handle))
	}
	return o.engine.checkStatus(o.engine.funcs.disableCpuMemArena(o.handle))
}

// GetMemPatternEnabled reports whether the memory pattern optimization is enabled
// in the session options. This is the query counterpart to SetMemPattern.
//
// Requires ONNX Runtime 1.27+ (ORT_API_VERSION 27). Returns an error if the
// loaded library does not support this API.
func (o *SessionOptions) GetMemPatternEnabled() (bool, error) {
	if o.engine.funcs.getMemPatternEnabled == nil {
		return false, fmt.Errorf("GetMemPatternEnabled requires ONNX Runtime 1.27+ (current API version: %d)", o.engine.version)
	}
	var enabled int32
	status := o.engine.funcs.getMemPatternEnabled(o.handle, &enabled)
	if err := o.engine.checkStatus(status); err != nil {
		return false, err
	}
	return enabled != 0, nil
}

// GetExecutionMode returns the current execution mode set on the session options.
// This is the query counterpart to SetExecutionMode.
//
// Requires ONNX Runtime 1.27+ (ORT_API_VERSION 27). Returns an error if the
// loaded library does not support this API.
func (o *SessionOptions) GetExecutionMode() (ExecutionMode, error) {
	if o.engine.funcs.getSessionExecutionModeFunc == nil {
		return 0, fmt.Errorf("GetExecutionMode requires ONNX Runtime 1.27+ (current API version: %d)", o.engine.version)
	}
	var mode ExecutionMode
	status := o.engine.funcs.getSessionExecutionModeFunc(o.handle, &mode)
	if err := o.engine.checkStatus(status); err != nil {
		return 0, err
	}
	return mode, nil
}

// EnableCUDA enables CUDA execution provider.
func (o *SessionOptions) EnableCUDA() error {
	var cudaOpts CUDAProviderOptionsV2Handle
	status := o.engine.funcs.createCUDAProviderOptions(&cudaOpts)
	if err := o.engine.checkStatus(status); err != nil {
		return fmt.Errorf("failed to create CUDA provider options: %w", err)
	}
	defer o.engine.funcs.releaseCUDAProviderOptions(cudaOpts)

	status = o.engine.funcs.appendExecutionProvider_CUDA_V2(o.handle, cudaOpts)
	return o.engine.checkStatus(status)
}

// Destroy releases the underlying ORT SessionOptions resources.
// This method is safe for concurrent use; it will only execute once.
func (o *SessionOptions) Destroy() {
	o.destroyOnce.Do(func() {
		if o.handle != 0 {
			o.engine.funcs.releaseSessionOptions(o.handle)
			o.handle = 0
		}
	})
}

// NewSession creates a new inference session.
//
// # Params:
//
//	modelPath: path to the ONNX model file
//	opts: session configuration options
func (e *Engine) NewSession(modelPath string, opts *SessionOptions) (*Session, error) {
	ortlog.Infow("creating ONNX Runtime session", "modelPath", modelPath)

	var optHandle SessionOptionsHandle
	if opts != nil {
		optHandle = opts.handle
	}

	pathPtr, err := stringToPathPtr(modelPath)
	if err != nil {
		ortlog.Errorw("failed to convert model path", "modelPath", modelPath, "error", err)
		return nil, err
	}

	var h SessionHandle
	status := e.funcs.createSession(e.envHandle, pathPtr, optHandle, &h)
	runtime.KeepAlive(pathPtr)
	if err := e.checkStatus(status); err != nil {
		ortlog.Errorw("failed to create session", "modelPath", modelPath, "error", err)
		return nil, err
	}

	s := &Session{
		handle: h,
		engine: e,
	}

	if err := s.initMetadata(); err != nil {
		s.Destroy()
		ortlog.Errorw("failed to initialize session metadata", "modelPath", modelPath, "error", err)
		return nil, err
	}

	ortlog.Infow("ONNX Runtime session created successfully",
		"modelPath", modelPath,
		"inputs", s.InputNames,
		"outputs", s.OutputNames)

	return s, nil
}

func (s *Session) initMetadata() error {
	// input
	inputCount, err := s.getInputCount()
	if err != nil {
		return err
	}
	s.InputNames = make([]string, inputCount)
	for i := 0; i < inputCount; i++ {
		name, err := s.getInputName(i)
		if err != nil {
			return err
		}
		s.InputNames[i] = name
	}

	// output
	outputCount, err := s.getOutputCount()
	if err != nil {
		return err
	}
	s.OutputNames = make([]string, outputCount)
	for i := 0; i < outputCount; i++ {
		name, err := s.getOutputName(i)
		if err != nil {
			return err
		}
		s.OutputNames[i] = name
	}

	s.outputNameSlices = make([][]byte, outputCount)
	s.outputNamePtrs = make([]unsafe.Pointer, outputCount)
	for i, name := range s.OutputNames {
		b := make([]byte, len(name)+1)
		copy(b, name)
		s.outputNameSlices[i] = b
		s.outputNamePtrs[i] = unsafe.Pointer(&b[0])
	}

	return nil
}

func (s *Session) getInputCount() (int, error) {
	var count uintptr
	status := s.engine.funcs.sessionGetInputCount(s.handle, &count)
	return int(count), s.engine.checkStatus(status)
}

func (s *Session) getOutputCount() (int, error) {
	var count uintptr
	status := s.engine.funcs.sessionGetOutputCount(s.handle, &count)
	return int(count), s.engine.checkStatus(status)
}

func (s *Session) getInputName(index int) (string, error) {
	var allocator AllocatorHandle
	status := s.engine.funcs.getAllocatorWithDefaultOptions(&allocator)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	var namePtr *byte
	status = s.engine.funcs.sessionGetInputName(s.handle, uintptr(index), allocator, &namePtr)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	name := cStringToString(namePtr)
	s.engine.funcs.allocatorFree(allocator, unsafe.Pointer(namePtr))

	return name, nil
}

func (s *Session) getOutputName(index int) (string, error) {
	var allocator AllocatorHandle
	status := s.engine.funcs.getAllocatorWithDefaultOptions(&allocator)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	var namePtr *byte
	status = s.engine.funcs.sessionGetOutputName(s.handle, uintptr(index), allocator, &namePtr)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	name := cStringToString(namePtr)
	s.engine.funcs.allocatorFree(allocator, unsafe.Pointer(namePtr))

	return name, nil
}

func (s *Session) Destroy() {
	s.destroyOnce.Do(func() {
		if s.handle != 0 {
			ortlog.Debugw("destroying ONNX Runtime session", "inputs", s.InputNames, "outputs", s.OutputNames)
			s.engine.funcs.releaseSession(s.handle)
			s.handle = 0
		}
	})
}

// ReleaseCapturedGraph releases a previously captured graph and its associated
// resources for the given graph annotation ID.
//
// When graph capture is enabled, the EP records information during initial runs
// (e.g., GPU commands) and replays them on subsequent runs. This function frees
// the captured resources for a specific graph annotation ID, reclaiming memory.
//
// Requires ONNX Runtime 1.27+ (ORT_API_VERSION 27). Returns an error if the
// loaded library does not support this API.
func (s *Session) ReleaseCapturedGraph(graphAnnotationID int) error {
	if s.engine.funcs.sessionReleaseCapturedGraph == nil {
		return fmt.Errorf("ReleaseCapturedGraph requires ONNX Runtime 1.27+ (current API version: %d)", s.engine.version)
	}
	status := s.engine.funcs.sessionReleaseCapturedGraph(s.handle, int32(graphAnnotationID))
	return s.engine.checkStatus(status)
}

// NOTE: KernelContext_GetSyncStream (OrtApi index 419, since v1.28) is intentionally
// not wrapped at the Go level. It operates on OrtKernelContext*, which is only
// available inside custom-op Compute callbacks — a use case raven-onnxruntime does
// not support (it is an inference-only binding). The function pointer is present in
// the ortApi struct for ABI completeness but is not registered in apiFuncs.

func (s *Session) GetInputShape(index int) ([]int64, error) {
	var typeInfo TypeInfoHandle
	status := s.engine.funcs.sessionGetInputTypeInfo(s.handle, uintptr(index), &typeInfo)
	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get input type info at index %d: %w", index, err)
	}
	defer s.engine.funcs.releaseTypeInfo(typeInfo)

	var onnxType OnnxType
	status = s.engine.funcs.getOnnxTypeFromTypeInfo(typeInfo, &onnxType)
	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get onnx type from type info: %w", err)
	}

	if onnxType != 1 {
		return nil, fmt.Errorf("input %d is not a tensor type (onnxType=%d)", index, onnxType)
	}

	tensorInfo := s.engine.funcs.castTypeInfoToTensorInfo(typeInfo)
	if tensorInfo == 0 {
		return nil, fmt.Errorf("input %d cast to tensor info returned nil", index)
	}
	// NOTE: Do NOT call releaseTensorTypeAndShapeInfo(tensorInfo) here.
	// CastTypeInfoToTensorInfo returns a borrowed reference owned by typeInfo.
	// The deferred releaseTypeInfo(typeInfo) above already handles cleanup.
	// Releasing tensorInfo separately would cause a double-free.

	var dimCount uintptr
	status = s.engine.funcs.getDimensionsCount(tensorInfo, &dimCount)
	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get dimensions count: %w", err)
	}

	dims := make([]int64, dimCount)
	if dimCount > 0 {
		status = s.engine.funcs.getDimensions(tensorInfo, &dims[0], dimCount)
		if err := s.engine.checkStatus(status); err != nil {
			return nil, fmt.Errorf("failed to get dimensions: %w", err)
		}
	}

	return dims, nil
}

// Run executes inference.
func (s *Session) Run(inputs map[string]*Value) (map[string]*Value, error) {
	inputCount := len(inputs)
	outputCount := len(s.OutputNames)

	if inputCount == 0 {
		return nil, fmt.Errorf("session run: inputs map cannot be empty or nil")
	}
	if outputCount == 0 {
		return nil, fmt.Errorf("session run: session has no outputs")
	}

	ortlog.Debugw("starting inference",
		"inputCount", inputCount,
		"outputCount", outputCount,
		"inputs", mapKeys(inputs),
		"outputs", s.OutputNames)

	// input - keep byte slices alive until the C call completes
	inputNameSlices := make([][]byte, inputCount)
	inputNamePtrs := make([]unsafe.Pointer, inputCount)
	inputHandles := make([]ValueHandle, inputCount)
	i := 0
	for name, val := range inputs {
		if val == nil {
			return nil, fmt.Errorf("session run: input %q is nil", name)
		}
		ptr, backing := s.getOrCreateInputNamePtr(name)
		inputNameSlices[i] = backing
		inputNamePtrs[i] = ptr
		inputHandles[i] = val.handle
		i++
	}

	outputHandles := make([]ValueHandle, outputCount)

	// execute inference via the underlying runtime
	status := s.engine.funcs.run(
		s.handle,
		0,
		&inputNamePtrs[0],
		&inputHandles[0],
		uintptr(inputCount),
		&s.outputNamePtrs[0],
		uintptr(outputCount),
		&outputHandles[0],
	)

	// keep slices alive until the C call completes
	runtime.KeepAlive(inputNameSlices)
	runtime.KeepAlive(s.outputNameSlices)
	// Keep input Values alive so their underlying data is not GC'd
	// while ORT reads it during the run call above.
	runtime.KeepAlive(inputs)

	if err := s.engine.checkStatus(status); err != nil {
		// Release any output handles that may have been partially created
		// before the run failed, to avoid leaking ORT Value handles.
		for _, h := range outputHandles {
			if h != 0 {
				s.engine.funcs.releaseValue(h)
			}
		}
		ortlog.Errorw("inference failed",
			"inputs", mapKeys(inputs),
			"outputs", s.OutputNames,
			"error", err)
		return nil, fmt.Errorf("session run failed: %w", err)
	}

	results := make(map[string]*Value, outputCount)
	for i := 0; i < outputCount; i++ {
		results[s.OutputNames[i]] = &Value{
			handle: outputHandles[i],
			engine: s.engine,
		}
	}

	ortlog.Debugw("inference completed", "outputCount", len(results))

	return results, nil
}

// NewTensor delegates to Engine.NewTensor, creating a tensor using the session's engine context.
func (s *Session) NewTensor(shape []int64, data interface{}) (*Value, error) {
	return s.engine.NewTensor(shape, data)
}

func mapKeys(m map[string]*Value) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func (s *Session) getOrCreateInputNamePtr(name string) (unsafe.Pointer, []byte) {
	if cached, ok := s.cachedInputNamePtrs.Load(name); ok {
		backing := cached.([]byte)
		return unsafe.Pointer(&backing[0]), backing
	}

	backing := make([]byte, len(name)+1)
	copy(backing, name)

	actual, _ := s.cachedInputNamePtrs.LoadOrStore(name, backing)
	resolved := actual.([]byte)
	return unsafe.Pointer(&resolved[0]), resolved
}
