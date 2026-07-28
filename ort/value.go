package ort

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/up-zero/gotool"
)

// Value represents an ONNX Runtime Tensor value.
// Call Destroy() to release the underlying resources when done.
type Value struct {
	handle       ValueHandle
	engine       *Engine
	shape        []int64
	elementCount int
	dataRef      any // keeps the data slice alive to prevent GC
	destroyOnce  sync.Once
}

// NewTensor creates a Tensor using the Engine.
//
// # Params:
//
//	shape: Tensor dimensions, e.g. []int64{1, 3, 640, 640}
//	data: data slice ([]float32, []uint8, etc.), caller is responsible for its lifetime
func (e *Engine) NewTensor(shape []int64, data any) (*Value, error) {
	dataType, typeSize, dataLen, dataPtr, err := parseInputData(data)
	if err != nil {
		return nil, err
	}

	var valHandle ValueHandle
	var shapePtr *int64
	if len(shape) > 0 {
		shapePtr = &shape[0]
	}

	status := e.funcs.createTensorWithDataAsOrtValue(
		e.memInfo,
		dataPtr,
		uintptr(dataLen)*typeSize,
		shapePtr,
		uintptr(len(shape)),
		dataType,
		&valHandle,
	)
	// Keep data and shape alive until the C call completes.
	// dataPtr and shapePtr are unsafe.Pointers that the compiler does not
	// associate with their backing slices, so without KeepAlive the GC
	// could reclaim them while ORT is still reading them.
	runtime.KeepAlive(data)
	runtime.KeepAlive(shape)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}

	return &Value{
		handle:  valHandle,
		engine:  e,
		dataRef: data,
	}, nil
}

// GetShape returns the Tensor dimensions. Results are cached; only the first call queries ORT.
func (v *Value) GetShape() ([]int64, error) {
	if len(v.shape) > 0 {
		return v.shape, nil
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return nil, err
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var dimCount uintptr
	status := v.engine.funcs.getDimensionsCount(info, &dimCount)
	if err := v.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get dimensions count: %w", err)
	}

	v.shape = make([]int64, dimCount)
	if dimCount > 0 {
		status = v.engine.funcs.getDimensions(info, &v.shape[0], dimCount)
		if err := v.engine.checkStatus(status); err != nil {
			return nil, fmt.Errorf("failed to get dimensions: %w", err)
		}
	}

	return v.shape, nil
}

// GetElementCount returns the total number of elements in the Tensor. Results are cached.
func (v *Value) GetElementCount() (int, error) {
	if v.elementCount > 0 {
		return v.elementCount, nil
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return 0, err
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var elementCount uintptr
	status := v.engine.funcs.getTensorShapeElementCount(info, &elementCount)
	if err := v.engine.checkStatus(status); err != nil {
		return 0, fmt.Errorf("failed to get tensor shape element count: %w", err)
	}

	v.elementCount = int(elementCount)
	return v.elementCount, nil
}

// GetTensorData retrieves the Tensor data as the specified Go type.
// The generic parameter T must match the actual Tensor element type, otherwise a type mismatch error is returned.
//
// IMPORTANT: This function returns a COPY of the ORT-managed tensor data.
// The returned slice is safe to use after the Value is Destroyed.
func GetTensorData[T gotool.Number](v *Value) ([]T, error) {
	elementCount, err := v.GetElementCount()
	if err != nil {
		return nil, err
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get tensor type and shape info: %w", err)
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var dataType TensorElementDataType
	status := v.engine.funcs.getTensorElementType(info, &dataType)
	if err := v.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get tensor element type: %w", err)
	}

	var ptr unsafe.Pointer
	status = v.engine.funcs.getTensorMutableData(v.handle, &ptr)
	if err := v.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get tensor mutable data: %w", err)
	}

	// Defensive: guard against nil pointer with non-zero elementCount
	if elementCount > 0 && ptr == nil {
		return nil, fmt.Errorf("getTensorMutableData returned nil pointer with elementCount=%d", elementCount)
	}
	if elementCount == 0 {
		return []T{}, nil
	}

	var rawData any
	switch dataType {
	case TensorElementDataTypeFloat:
		rawData = unsafe.Slice((*float32)(ptr), elementCount)
	case TensorElementDataTypeDouble:
		rawData = unsafe.Slice((*float64)(ptr), elementCount)
	case TensorElementDataTypeInt64:
		rawData = unsafe.Slice((*int64)(ptr), elementCount)
	case TensorElementDataTypeInt32:
		rawData = unsafe.Slice((*int32)(ptr), elementCount)
	case TensorElementDataTypeInt16:
		rawData = unsafe.Slice((*int16)(ptr), elementCount)
	case TensorElementDataTypeInt8:
		rawData = unsafe.Slice((*int8)(ptr), elementCount)
	case TensorElementDataTypeUint64:
		rawData = unsafe.Slice((*uint64)(ptr), elementCount)
	case TensorElementDataTypeUint32:
		rawData = unsafe.Slice((*uint32)(ptr), elementCount)
	case TensorElementDataTypeUint16:
		rawData = unsafe.Slice((*uint16)(ptr), elementCount)
	case TensorElementDataTypeUint8:
		rawData = unsafe.Slice((*uint8)(ptr), elementCount)
	case TensorElementDataTypeBool:
		rawData = unsafe.Slice((*bool)(ptr), elementCount)
	default:
		return nil, fmt.Errorf("unsupported tensor element type: %d", dataType)
	}

	if data, ok := rawData.([]T); ok {
		// Return a copy to prevent use-after-free: the raw slice points into
		// ORT-managed memory that is freed when the Value is Destroyed.
		// Without this copy, callers that Destroy the Value before using the
		// returned slice would read freed memory.
		result := make([]T, len(data))
		copy(result, data)
		return result, nil
	}

	var t T
	return nil, fmt.Errorf("tensor data type mismatch: actual ORT type %d does not match requested Go type %T", dataType, t)
}

// DestroyValues releases a map of output tensors.
func DestroyValues(values map[string]*Value) {
	for _, v := range values {
		if v != nil {
			v.Destroy()
		}
	}
}

// DestroyValueSlice releases a slice of values.
func DestroyValueSlice(values []*Value) {
	for _, v := range values {
		if v != nil {
			v.Destroy()
		}
	}
}

func (v *Value) getTypeAndShapeInfo() (TensorTypeAndShapeInfoHandle, error) {
	var info TensorTypeAndShapeInfoHandle
	status := v.engine.funcs.getTensorTypeAndShape(v.handle, &info)
	if err := v.engine.checkStatus(status); err != nil {
		return 0, err
	}
	return info, nil
}

// parseInputData parses the input data to extract ORT-required data type, element size, count, and pointer.
//
// # Params:
//
//	data: input data (must be a supported numeric slice type)
//
// # Returns:
//
//	dataType:  ORT element type enum
//	typeSize:  size of a single element in bytes
//	dataLen:   number of elements
//	dataPtr:   unsafe.Pointer to the underlying array
//	error:     returns error for unsupported types
func parseInputData(data any) (TensorElementDataType, uintptr, int, unsafe.Pointer, error) {
	switch d := data.(type) {
	case []float32:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeFloat, 4, len(d), ptr, nil
	case []float64:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeDouble, 8, len(d), ptr, nil
	case []int64:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeInt64, 8, len(d), ptr, nil
	case []int32:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeInt32, 4, len(d), ptr, nil
	case []int16:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeInt16, 2, len(d), ptr, nil
	case []int8:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeInt8, 1, len(d), ptr, nil
	case []uint64:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeUint64, 8, len(d), ptr, nil
	case []uint32:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeUint32, 4, len(d), ptr, nil
	case []uint16:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeUint16, 2, len(d), ptr, nil
	case []uint8:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeUint8, 1, len(d), ptr, nil
	case []bool:
		var ptr unsafe.Pointer
		if len(d) > 0 {
			ptr = unsafe.Pointer(&d[0])
		}
		return TensorElementDataTypeBool, 1, len(d), ptr, nil
	default:
		return TensorElementDataTypeUndefined, 0, 0, nil, fmt.Errorf("unsupported input type: %T", data)
	}
}

// Destroy releases the underlying ORT Value resources.
// This method is safe for concurrent use; it will only execute once.
// The Value must not be used after calling Destroy.
func (v *Value) Destroy() {
	v.destroyOnce.Do(func() {
		if v.handle != 0 {
			v.engine.funcs.releaseValue(v.handle)
			v.handle = 0
		}
		// clear cached data to prevent misuse after Destroy
		v.dataRef = nil
		v.shape = nil
		v.elementCount = 0
	})
}
