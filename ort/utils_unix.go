//go:build linux || darwin

package ort

import "unsafe"

// stringToPathPtr converts a string to a UTF-8 (char*) pointer
func stringToPathPtr(s string) (unsafe.Pointer, error) {
	b := make([]byte, len(s)+1)
	copy(b, s)
	b[len(s)] = 0
	return unsafe.Pointer(&b[0]), nil
}
