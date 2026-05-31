//go:build windows

package sys

import (
	"fmt"
	"syscall"
)

// LoadLibrary loads a dynamic library file
func LoadLibrary(name string) (uintptr, error) {
	handle, err := syscall.LoadLibrary(name)
	if err != nil {
		return 0, fmt.Errorf("failed to load dll %s: %w", name, err)
	}
	return uintptr(handle), nil
}

func FreeLibrary(handle uintptr) error {
	return syscall.FreeLibrary(syscall.Handle(handle))
}
