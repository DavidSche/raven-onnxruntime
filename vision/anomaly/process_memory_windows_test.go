//go:build windows

package anomaly

import (
	"fmt"
	"syscall"
	"unsafe"
)

type processMemoryStats struct {
	workingSetBytes     uint64
	peakWorkingSetBytes uint64
	privateBytes        uint64
	peakPagefileUsage   uint64
}

type processMemoryCounters struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
	privateUsage               uintptr
}

var (
	procGetCurrentProcess    = syscall.NewLazyDLL("kernel32.dll").NewProc("GetCurrentProcess")
	procGetProcessMemoryInfo = syscall.NewLazyDLL("psapi.dll").NewProc("GetProcessMemoryInfo")
)

func readProcessMemoryStats() (processMemoryStats, error) {
	if err := procGetProcessMemoryInfo.Find(); err != nil {
		return processMemoryStats{}, fmt.Errorf("GetProcessMemoryInfo unavailable: %w", err)
	}
	handle, _, _ := procGetCurrentProcess.Call()

	var counters processMemoryCounters
	counters.cb = uint32(unsafe.Sizeof(counters))
	result, _, errno := procGetProcessMemoryInfo.Call(
		handle,
		uintptr(unsafe.Pointer(&counters)),
		uintptr(unsafe.Sizeof(counters)),
	)
	if result == 0 {
		return processMemoryStats{}, fmt.Errorf("GetProcessMemoryInfo failed: %w", errno)
	}
	return processMemoryStats{
		workingSetBytes:     uint64(counters.workingSetSize),
		peakWorkingSetBytes: uint64(counters.peakWorkingSetSize),
		privateBytes:        uint64(counters.privateUsage),
		peakPagefileUsage:   uint64(counters.peakPagefileUsage),
	}, nil
}
