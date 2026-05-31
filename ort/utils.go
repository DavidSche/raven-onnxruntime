package ort

import (
	"fmt"
	"runtime"
	"unsafe"
)

// DefaultLibraryPath returns the default path to the ONNX Runtime dynamic library based on the runtime platform.
//
// Example return values:
//
//	Windows:      ./lib/onnxruntime.dll
//	Linux amd64:  ./lib/onnxruntime_amd64.so
//	Linux arm64:  ./lib/onnxruntime_arm64.so
//	macOS amd64:  ./lib/onnxruntime_amd64.dylib
//	macOS arm64:  ./lib/onnxruntime_arm64.dylib
func DefaultLibraryPath() string {
	baseDir := "./lib/"
	libName := "onnxruntime"

	// Windows uses .dll without architecture suffix
	if runtime.GOOS == "windows" {
		return baseDir + libName + ".dll"
	}

	var ext string
	switch runtime.GOOS {
	case "darwin":
		ext = "dylib"
	case "linux":
		ext = "so"
	default:
		return baseDir + libName + "_amd64.so" // fallback to linux amd64 for unknown platforms
	}

	return fmt.Sprintf("%s%s_%s.%s", baseDir, libName, runtime.GOARCH, ext)
}

// stringToCString converts a Go string to a null-terminated byte slice and returns a pointer to its first byte.
// The caller must ensure the byte slice is kept alive during the C function call (to prevent GC).
func stringToCString(s string) (*byte, error) {
	b := make([]byte, len(s)+1)
	copy(b, s)
	// b[len(s)] is already 0 from make, no need to explicitly set
	return &b[0], nil
}

// cStringToString converts a null-terminated C string pointer to a Go string.
// Uses unsafe.Add for pointer stepping, conforming to the Go memory model.
func cStringToString(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	// use unsafe.Add for byte stepping to avoid uintptr arithmetic risks during GC
	p := unsafe.Pointer(ptr)
	length := 0
	for *(*byte)(p) != 0 {
		p = unsafe.Add(p, 1)
		length++
	}
	return string(unsafe.Slice(ptr, length))
}
