package ort

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"unsafe"

	"github.com/DavidSche/raven-onnxruntime/ort/internal/sys"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/ebitengine/purego"
)

type ApiVersion uint32

const (
	ApiVersion17 ApiVersion = 17
	ApiVersion18 ApiVersion = 18
	ApiVersion19 ApiVersion = 19
	ApiVersion20 ApiVersion = 20
	ApiVersion21 ApiVersion = 21
	ApiVersion22 ApiVersion = 22
	ApiVersion23 ApiVersion = 23
	ApiVersion24 ApiVersion = 24
	ApiVersion25 ApiVersion = 25
	ApiVersion26 ApiVersion = 26
	ApiVersion27 ApiVersion = 27
	ApiVersion28 ApiVersion = 28

	DefaultApiVersion = ApiVersion28

	LogVerbose LoggingLevel = 0
	LogInfo    LoggingLevel = 1
	LogWarning LoggingLevel = 2
	LogError   LoggingLevel = 3
	LogFatal   LoggingLevel = 4

	DefaultEnvName = "GETCHARZP"
)

var supportedApiVersions = []ApiVersion{
	ApiVersion28, ApiVersion27, ApiVersion26, ApiVersion25, ApiVersion24,
	ApiVersion23, ApiVersion22, ApiVersion21,
	ApiVersion20, ApiVersion19, ApiVersion18, ApiVersion17,
}

type EngineOption func(*engineConfig)

type engineConfig struct {
	apiVersion ApiVersion
}

func WithApiVersion(v ApiVersion) EngineOption {
	return func(c *engineConfig) {
		c.apiVersion = v
	}
}

func resolveApiVersion(opts []EngineOption) ApiVersion {
	cfg := &engineConfig{apiVersion: DefaultApiVersion}
	for _, opt := range opts {
		opt(cfg)
	}
	if envVal := os.Getenv("ORT_API_VERSION"); envVal != "" {
		if v, err := strconv.ParseUint(envVal, 10, 32); err == nil && v > 0 {
			ortlog.Infow("ORT_API_VERSION env override", "envValue", v)
			cfg.apiVersion = ApiVersion(v)
		} else {
			ortlog.Warnw("invalid ORT_API_VERSION env value, ignoring", "envValue", envVal, "error", err)
		}
	}
	return cfg.apiVersion
}

// Engine represents the ONNX Runtime inference engine context.
type Engine struct {
	handle  uintptr
	version ApiVersion
	api     *ortApi
	funcs   *apiFuncs

	envHandle   EnvHandle
	memInfo     MemoryInfoHandle
	destroyOnce sync.Once
}

func (e *Engine) AvailableProviders() ([]string, error) {
	var providers **byte
	var count int32
	status := e.funcs.getAvailableProviders(&providers, &count)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}
	defer func() {
		_ = e.funcs.releaseAvailableProviders(providers, count)
	}()

	raw := unsafe.Slice(providers, int(count))
	names := make([]string, 0, count)
	for _, provider := range raw {
		names = append(names, cStringToString(provider))
	}
	return names, nil
}

// NewEngine initializes the ONNX Runtime engine.
//
// Parameters:
//
//	libPath: path to the ONNX Runtime dynamic library
//	opts: optional configuration, e.g. WithApiVersion(ort.ApiVersion23)
//
// Version negotiation strategy:
//  1. Prefer the version specified by WithApiVersion
//  2. Then read the environment variable ORT_API_VERSION
//  3. Default to DefaultApiVersion (currently 28)
//  4. If the requested version is unavailable, automatically downgrade to the highest version supported by the library
func NewEngine(libPath string, opts ...EngineOption) (*Engine, error) {
	requestedVersion := resolveApiVersion(opts)

	ortlog.Infow("initializing ONNX Runtime engine",
		"libPath", libPath,
		"requestedApiVersion", requestedVersion)

	handle, err := sys.LoadLibrary(libPath)
	if err != nil {
		ortlog.Errorw("failed to load library", "libPath", libPath, "error", err)
		return nil, fmt.Errorf("failed to load library: %w", err)
	}

	engine := &Engine{
		handle:  handle,
		version: requestedVersion,
		funcs:   &apiFuncs{},
	}
	defer func() {
		if err != nil && engine != nil {
			engine.Destroy()
		}
	}()

	if err = engine.initApi(); err != nil {
		ortlog.Errorw("failed to initialize API", "error", err)
		return nil, err
	}
	if err = engine.initEnv(DefaultEnvName); err != nil {
		ortlog.Errorw("failed to initialize environment", "error", err)
		return nil, err
	}
	if err = engine.initMemInfo(); err != nil {
		ortlog.Errorw("failed to initialize memory info", "error", err)
		return nil, err
	}

	ortlog.Infow("ONNX Runtime engine initialized successfully",
		"version", engine.GetVersion(),
		"apiVersion", engine.version)

	return engine, nil
}

func (e *Engine) initApi() (err error) {
	// purego.RegisterLibFunc/RegisterFunc panic if the symbol is not found
	// or the function pointer is invalid. Recover and convert to error.
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic during ONNX Runtime API initialization: %v", r)
		}
	}()

	var ortGetApiBase func() *ortApiBase
	purego.RegisterLibFunc(&ortGetApiBase, e.handle, "OrtGetApiBase")

	apiBase := ortGetApiBase()
	if apiBase == nil {
		return fmt.Errorf("OrtGetApiBase returned nil")
	}
	purego.RegisterFunc(&e.funcs.getVersionString, apiBase.GetVersionString)

	var getApi func(ApiVersion) *ortApi
	purego.RegisterFunc(&getApi, apiBase.GetAPI)

	e.api = getApi(e.version)
	if e.api != nil {
		ortlog.Infow("ONNX Runtime API resolved",
			"version", e.version)
	} else {
		ortlog.Warnw("requested API version not available, attempting auto-negotiation",
			"requestedVersion", e.version,
			"libraryVersion", e.funcs.getVersionString())

		for _, candidate := range supportedApiVersions {
			if candidate >= e.version {
				continue
			}
			candidateApi := getApi(candidate)
			if candidateApi != nil {
				ortlog.Infow("API version auto-negotiated",
					"requestedVersion", e.version,
					"negotiatedVersion", candidate)
				e.api = candidateApi
				e.version = candidate
				break
			}
		}

		if e.api == nil {
			return fmt.Errorf("ONNX Runtime library does not support API version %d or any compatible lower version (library version: %s)",
				e.version, e.funcs.getVersionString())
		}
	}

	// status, error
	purego.RegisterFunc(&e.funcs.createStatus, e.api.CreateStatus)
	purego.RegisterFunc(&e.funcs.getErrorCode, e.api.GetErrorCode)
	purego.RegisterFunc(&e.funcs.getErrorMessage, e.api.GetErrorMessage)
	purego.RegisterFunc(&e.funcs.releaseStatus, e.api.ReleaseStatus)

	// env
	purego.RegisterFunc(&e.funcs.createEnv, e.api.CreateEnv)
	purego.RegisterFunc(&e.funcs.releaseEnv, e.api.ReleaseEnv)

	// allocator
	purego.RegisterFunc(&e.funcs.getAllocatorWithDefaultOptions, e.api.GetAllocatorWithDefaultOptions)
	purego.RegisterFunc(&e.funcs.allocatorFree, e.api.AllocatorFree)

	// memory info
	purego.RegisterFunc(&e.funcs.createCpuMemoryInfo, e.api.CreateCpuMemoryInfo)
	purego.RegisterFunc(&e.funcs.releaseMemoryInfo, e.api.ReleaseMemoryInfo)

	// CUDA
	purego.RegisterFunc(&e.funcs.createCUDAProviderOptions, e.api.CreateCUDAProviderOptions)
	purego.RegisterFunc(&e.funcs.releaseCUDAProviderOptions, e.api.ReleaseCUDAProviderOptions)
	purego.RegisterFunc(&e.funcs.updateCUDAProviderOptions, e.api.UpdateCUDAProviderOptions)
	purego.RegisterFunc(&e.funcs.appendExecutionProvider_CUDA_V2, e.api.SessionOptionsAppendExecutionProvider_CUDA_V2)

	// session options
	purego.RegisterFunc(&e.funcs.createSessionOptions, e.api.CreateSessionOptions)
	purego.RegisterFunc(&e.funcs.setSessionExecutionMode, e.api.SetSessionExecutionMode)
	purego.RegisterFunc(&e.funcs.enableMemPattern, e.api.EnableMemPattern)
	purego.RegisterFunc(&e.funcs.disableMemPattern, e.api.DisableMemPattern)
	purego.RegisterFunc(&e.funcs.setSessionGraphOptimizationLevel, e.api.SetSessionGraphOptimizationLevel)
	purego.RegisterFunc(&e.funcs.setIntraOpNumThreads, e.api.SetIntraOpNumThreads)
	purego.RegisterFunc(&e.funcs.setInterOpNumThreads, e.api.SetInterOpNumThreads)
	purego.RegisterFunc(&e.funcs.sessionOptionsAppendExecutionProvider, e.api.SessionOptionsAppendExecutionProvider)
	purego.RegisterFunc(&e.funcs.releaseSessionOptions, e.api.ReleaseSessionOptions)
	purego.RegisterFunc(&e.funcs.enableCpuMemArena, e.api.EnableCpuMemArena)
	purego.RegisterFunc(&e.funcs.disableCpuMemArena, e.api.DisableCpuMemArena)

	// session
	purego.RegisterFunc(&e.funcs.createSession, e.api.CreateSession)
	purego.RegisterFunc(&e.funcs.createSessionFromArray, e.api.CreateSessionFromArray)
	purego.RegisterFunc(&e.funcs.sessionGetInputCount, e.api.SessionGetInputCount)
	purego.RegisterFunc(&e.funcs.sessionGetOutputCount, e.api.SessionGetOutputCount)
	purego.RegisterFunc(&e.funcs.sessionGetInputName, e.api.SessionGetInputName)
	purego.RegisterFunc(&e.funcs.sessionGetOutputName, e.api.SessionGetOutputName)
	purego.RegisterFunc(&e.funcs.sessionGetInputTypeInfo, e.api.SessionGetInputTypeInfo)
	purego.RegisterFunc(&e.funcs.sessionGetOutputTypeInfo, e.api.SessionGetOutputTypeInfo)
	purego.RegisterFunc(&e.funcs.run, e.api.Run)
	purego.RegisterFunc(&e.funcs.releaseSession, e.api.ReleaseSession)

	// tensor, value
	purego.RegisterFunc(&e.funcs.createTensorWithDataAsOrtValue, e.api.CreateTensorWithDataAsOrtValue)
	purego.RegisterFunc(&e.funcs.getValueType, e.api.GetValueType)
	purego.RegisterFunc(&e.funcs.getTensorMutableData, e.api.GetTensorMutableData)
	purego.RegisterFunc(&e.funcs.getTensorTypeAndShape, e.api.GetTensorTypeAndShape)
	purego.RegisterFunc(&e.funcs.getTensorElementType, e.api.GetTensorElementType)
	purego.RegisterFunc(&e.funcs.getDimensionsCount, e.api.GetDimensionsCount)
	purego.RegisterFunc(&e.funcs.getDimensions, e.api.GetDimensions)
	purego.RegisterFunc(&e.funcs.getTensorShapeElementCount, e.api.GetTensorShapeElementCount)
	purego.RegisterFunc(&e.funcs.releaseValue, e.api.ReleaseValue)
	purego.RegisterFunc(&e.funcs.releaseTensorTypeAndShapeInfo, e.api.ReleaseTensorTypeAndShapeInfo)
	purego.RegisterFunc(&e.funcs.releaseTypeInfo, e.api.ReleaseTypeInfo)
	purego.RegisterFunc(&e.funcs.castTypeInfoToTensorInfo, e.api.CastTypeInfoToTensorInfo)
	purego.RegisterFunc(&e.funcs.getOnnxTypeFromTypeInfo, e.api.GetOnnxTypeFromTypeInfo)

	// provider
	purego.RegisterFunc(&e.funcs.getAvailableProviders, e.api.GetAvailableProviders)
	purego.RegisterFunc(&e.funcs.releaseAvailableProviders, e.api.ReleaseAvailableProviders)

	// ORT 1.27+ APIs — only register if the negotiated API version supports them.
	// The native OrtApi struct grows by appending function pointers; reading fields
	// beyond the struct's actual size (e.g. index 415+ on an API 26 library) is
	// undefined behavior. The version guard below prevents that.
	if e.version >= ApiVersion27 {
		purego.RegisterFunc(&e.funcs.getMemPatternEnabled, e.api.GetMemPatternEnabled)
		purego.RegisterFunc(&e.funcs.getSessionExecutionModeFunc, e.api.GetSessionExecutionMode)
		purego.RegisterFunc(&e.funcs.sessionReleaseCapturedGraph, e.api.SessionReleaseCapturedGraph)
	}

	// ORT 1.28+ APIs
	if e.version >= ApiVersion28 {
		purego.RegisterFunc(&e.funcs.getExperimentalFunction, e.api.GetExperimentalFunction)
	}

	return nil
}

func (e *Engine) initEnv(name string) error {
	namePtr, err := stringToCString(name)
	if err != nil {
		return err
	}
	status := e.funcs.createEnv(LogError, namePtr, &e.envHandle)
	runtime.KeepAlive(namePtr)
	if err := e.checkStatus(status); err != nil {
		return fmt.Errorf("failed to create env: %w", err)
	}
	return nil
}

func (e *Engine) initMemInfo() error {
	var memInfo MemoryInfoHandle
	status := e.funcs.createCpuMemoryInfo(DeviceAllocator, DefaultMemType, &memInfo)
	if err := e.checkStatus(status); err != nil {
		return fmt.Errorf("failed to create cpu memory info: %v", err)
	}
	e.memInfo = memInfo
	return nil
}

// GetVersion returns the ONNX Runtime version string, e.g. "1.23.2".
func (e *Engine) GetVersion() string {
	if e.funcs.getVersionString == nil {
		return "unknown"
	}
	return e.funcs.getVersionString()
}

// GetApiVersion returns the negotiated ONNX Runtime C API version (e.g. 28).
func (e *Engine) GetApiVersion() ApiVersion {
	return e.version
}

// GetExperimentalFunction retrieves an experimental function pointer by name.
//
// Experimental functions are not part of the stable ABI and may be added or removed
// between releases without notice. The returned unsafe.Pointer is an opaque function
// pointer (OrtExperimentalFnPtr) that the caller must cast to the correct function
// pointer type before calling, typically via purego.RegisterFunc.
//
// Name constants and typedefs are defined in onnxruntime_experimental_c_api.h.
// Names follow the pattern "<target struct>_<function name>_SinceV<ORT API version>".
//
// Requires ONNX Runtime 1.28+ (ORT_API_VERSION 28). Returns an error if the
// loaded library does not support this API.
func (e *Engine) GetExperimentalFunction(name string) (unsafe.Pointer, error) {
	if e.funcs.getExperimentalFunction == nil {
		return nil, fmt.Errorf("GetExperimentalFunction requires ONNX Runtime 1.28+ (current API version: %d)", e.version)
	}
	namePtr, err := stringToCString(name)
	if err != nil {
		return nil, err
	}
	fn := e.funcs.getExperimentalFunction(namePtr)
	runtime.KeepAlive(namePtr)
	if fn == nil {
		return nil, fmt.Errorf("experimental function %q not found in this ORT build", name)
	}
	return fn, nil
}

// Destroy releases all resources held by the engine.
// This method is safe for concurrent use; it will only execute once.
func (e *Engine) Destroy() {
	e.destroyOnce.Do(func() {
		ortlog.Infow("destroying ONNX Runtime engine")
		if e.memInfo != 0 {
			if e.funcs != nil && e.funcs.releaseMemoryInfo != nil {
				e.funcs.releaseMemoryInfo(e.memInfo)
			}
			e.memInfo = 0
		}
		if e.envHandle != 0 {
			if e.funcs != nil && e.funcs.releaseEnv != nil {
				e.funcs.releaseEnv(e.envHandle)
			}
			e.envHandle = 0
		}
		if e.handle != 0 {
			if err := sys.FreeLibrary(e.handle); err != nil {
				ortlog.Warnw("failed to unload ONNX Runtime library", "error", err)
			}
			e.handle = 0
		}
		ortlog.Infow("ONNX Runtime engine destroyed")
	})
}

// IsAlive reports whether the engine still owns a live native handle.
func (e *Engine) IsAlive() bool {
	return e != nil && e.handle != 0
}

// checkStatus checks the ORT status and converts ONNX error codes to Go errors.
// See onnxruntime_c_api.h OrtErrorCode for error code definitions.
func (e *Engine) checkStatus(status StatusHandle) error {
	if status == 0 {
		return nil
	}
	defer e.funcs.releaseStatus(status)

	code := e.funcs.getErrorCode(status)
	msgPtr := e.funcs.getErrorMessage(status)
	msg := cStringToString((*byte)(msgPtr))

	codeDesc := ortErrorCodeDesc(code)
	ortlog.Errorw("ONNX Runtime error",
		"code", code,
		"codeDesc", codeDesc,
		"message", msg)
	return fmt.Errorf("onnxruntime [%s(%d)]: %s", codeDesc, code, msg)
}

// ortErrorCodeDesc converts an OrtErrorCode to a human-readable string.
func ortErrorCodeDesc(code ErrorCode) string {
	switch code {
	case 0:
		return "OK"
	case 1:
		return "FAIL"
	case 2:
		return "INVALID_ARGUMENT"
	case 3:
		return "NO_SUCHFILE"
	case 4:
		return "NO_MODEL"
	case 5:
		return "ENGINE_ERROR"
	case 6:
		return "RUNTIME_EXCEPTION"
	case 7:
		return "INVALID_PROTOBUF"
	case 8:
		return "MODEL_LOADED"
	case 9:
		return "NOT_IMPLEMENTED"
	case 10:
		return "INVALID_GRAPH"
	case 11:
		return "EP_FAIL"
	default:
		return "UNKNOWN"
	}
}
