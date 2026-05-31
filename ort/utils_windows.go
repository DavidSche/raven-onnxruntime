//go:build windows

package ort

import (
	"syscall"
	"unsafe"
)

// stringToPathPtr converts a string to a UTF-16 (wchar_t*) pointer
func stringToPathPtr(s string) (unsafe.Pointer, error) {
	ptr, err := syscall.UTF16PtrFromString(s)
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(ptr), nil
}
