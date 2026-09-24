//go:build !windows

package anomaly

import "errors"

type processMemoryStats struct {
	workingSetBytes     uint64
	peakWorkingSetBytes uint64
	privateBytes        uint64
	peakPagefileUsage   uint64
}

func readProcessMemoryStats() (processMemoryStats, error) {
	return processMemoryStats{}, errors.New("process RSS sampling is not implemented on this platform")
}
