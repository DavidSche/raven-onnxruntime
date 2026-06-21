package examples

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"
	"time"
)

// PrintBenchSummaryTable prints a formatted comparison table for multiple summaries.
func PrintBenchSummaryTable(title string, summaries []BenchSummary) {
	sep := strings.Repeat("=", 80)
	fmt.Println(sep)
	fmt.Printf("  %s\n", title)
	fmt.Println(sep)

	if len(summaries) == 0 {
		fmt.Println("  (no results)")
		return
	}

	// Header
	fmt.Printf("%-18s | %-5s | %8s | %8s | %8s | %8s | %8s | %8s | %7s | %7s | %7s | %6s\n",
		"Model", "Task", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)", "Min(ms)", "Max(ms)", "FPS", "Errors", "StdDev", "CV%")
	fmt.Println(strings.Repeat("-", 125))

	for _, s := range summaries {
		cvPercent := s.CV * 100
		fmt.Printf("%-18s | %-5s | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %7.1f | %7d | %7.2f | %5.1f\n",
			s.EngineName,
			s.Task,
			float64(s.AvgMs)/float64(time.Millisecond),
			float64(s.P50Ms)/float64(time.Millisecond),
			float64(s.P95Ms)/float64(time.Millisecond),
			float64(s.P99Ms)/float64(time.Millisecond),
			float64(s.MinMs)/float64(time.Millisecond),
			float64(s.MaxMs)/float64(time.Millisecond),
			s.FPS,
			s.ErrorCount,
			float64(s.StdDevMs)/float64(time.Millisecond),
			cvPercent,
		)
	}
	fmt.Println(strings.Repeat("-", 125))

	// Speedup comparison (relative to first engine)
	if len(summaries) > 1 {
		base := summaries[0]
		fmt.Printf("\nSpeedup relative to %s:\n", base.EngineName)
		for _, s := range summaries[1:] {
			if base.AvgMs == 0 || s.AvgMs == 0 {
				continue
			}
			ratio := float64(base.AvgMs) / float64(s.AvgMs)
			if ratio >= 1.0 {
				// base is slower (higher ms), so s is faster
				fmt.Printf("  %s is %.2fx faster than %s\n", s.EngineName, ratio, base.EngineName)
			} else {
				// base is faster (lower ms)
				fmt.Printf("  %s is %.2fx faster than %s\n", base.EngineName, 1.0/ratio, s.EngineName)
			}
		}
	}
	fmt.Println()
}

// PrintBenchDetailTable prints per-frame detail results.
func PrintBenchDetailTable(title string, summaries []BenchSummary, frameLabels []string) {
	sep := strings.Repeat("=", 80)
	fmt.Println(sep)
	fmt.Printf("  %s\n", title)
	fmt.Println(sep)

	if len(summaries) == 0 {
		fmt.Println("  (no results)")
		return
	}

	// Header
	header := fmt.Sprintf("%-12s ", "Frame")
	for _, s := range summaries {
		header += fmt.Sprintf("| %-14s ", s.EngineName)
	}
	fmt.Println(header)

	sepLine := strings.Repeat("-", 12)
	for range summaries {
		sepLine += "+" + strings.Repeat("-", 16)
	}
	fmt.Println(sepLine)

	maxLen := 0
	for _, s := range summaries {
		if len(s.Latencies) > maxLen {
			maxLen = len(s.Latencies)
		}
	}

	for i := 0; i < maxLen; i++ {
		label := fmt.Sprintf("%d", i)
		if i < len(frameLabels) {
			label = frameLabels[i]
		}
		row := fmt.Sprintf("%-12s ", label)
		for _, s := range summaries {
			if i < len(s.Latencies) {
				row += fmt.Sprintf("| %12.2fms ", float64(s.Latencies[i])/float64(time.Millisecond))
			} else {
				row += "|                "
			}
		}
		fmt.Println(row)
	}
	fmt.Println()
}

// PrintBenchDetectionsTable prints detection count comparison.
func PrintBenchDetectionsTable(title string, summaries []BenchSummary) {
	sep := strings.Repeat("=", 60)
	fmt.Println(sep)
	fmt.Printf("  %s\n", title)
	fmt.Println(sep)

	fmt.Printf("%-18s | %10s | %12s | %12s\n", "Model", "Total Dets", "Avg Dets/F", "Avg Latency")
	fmt.Println(strings.Repeat("-", 60))

	for _, s := range summaries {
		avgDets := 0
		if s.TotalFrames > 0 {
			avgDets = s.TotalDets / s.TotalFrames
		}
		fmt.Printf("%-18s | %10d | %12d | %12v\n",
			s.EngineName,
			s.TotalDets,
			avgDets,
			s.AvgMs.Round(time.Microsecond),
		)
	}
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println()
}

// WriteBenchCSV writes benchmark summaries to a CSV file.
func WriteBenchCSV(path string, summaries []BenchSummary) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{"Model", "Task", "TotalFrames", "TotalDets", "ErrorCount", "AvgMs", "P50Ms", "P95Ms", "P99Ms", "MinMs", "MaxMs", "StdDevMs", "CV", "FPS", "PeakMemoryMB"}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, s := range summaries {
		record := []string{
			s.EngineName,
			s.Task,
			fmt.Sprintf("%d", s.TotalFrames),
			fmt.Sprintf("%d", s.TotalDets),
			fmt.Sprintf("%d", s.ErrorCount),
			fmt.Sprintf("%.3f", float64(s.AvgMs)/float64(time.Millisecond)),
			fmt.Sprintf("%.3f", float64(s.P50Ms)/float64(time.Millisecond)),
			fmt.Sprintf("%.3f", float64(s.P95Ms)/float64(time.Millisecond)),
			fmt.Sprintf("%.3f", float64(s.P99Ms)/float64(time.Millisecond)),
			fmt.Sprintf("%.3f", float64(s.MinMs)/float64(time.Millisecond)),
			fmt.Sprintf("%.3f", float64(s.MaxMs)/float64(time.Millisecond)),
			fmt.Sprintf("%.3f", float64(s.StdDevMs)/float64(time.Millisecond)),
			fmt.Sprintf("%.4f", s.CV),
			fmt.Sprintf("%.2f", s.FPS),
			fmt.Sprintf("%.1f", s.PeakMemoryMB),
		}
		if err := w.Write(record); err != nil {
			return err
		}
	}
	return nil
}

// PrintBatchBenchTable prints batch inference comparison.
func PrintBatchBenchTable(title string, summaries []BenchSummary, batchSize int) {
	sep := strings.Repeat("=", 80)
	fmt.Println(sep)
	fmt.Printf("  %s (batch_size=%d)\n", title, batchSize)
	fmt.Println(sep)

	fmt.Printf("%-18s | %-5s | %12s | %12s | %12s | %7s\n",
		"Model", "Task", "Avg(ms)", "P95(ms)", "P99(ms)", "FPS")
	fmt.Println(strings.Repeat("-", 80))

	for _, s := range summaries {
		if s.TotalFrames == 0 {
			fmt.Printf("%-18s | %-5s | %12s | %12s | %12s | %7s\n",
				s.EngineName, s.Task, "N/A", "N/A", "N/A", "N/A")
			continue
		}
		fmt.Printf("%-18s | %-5s | %12.2f | %12.2f | %12.2f | %7.1f\n",
			s.EngineName,
			s.Task,
			float64(s.AvgMs)/float64(time.Millisecond),
			float64(s.P95Ms)/float64(time.Millisecond),
			float64(s.P99Ms)/float64(time.Millisecond),
			s.FPS,
		)
	}
	fmt.Println(strings.Repeat("-", 80))

	// Throughput comparison
	if len(summaries) > 1 {
		fmt.Printf("\nThroughput comparison (batch_size=%d):\n", batchSize)
		for i, s := range summaries {
			if s.TotalFrames == 0 {
				continue
			}
			throughput := float64(batchSize) * s.FPS
			fmt.Printf("  %s: %.1f images/sec\n", s.EngineName, throughput)
			_ = i
		}
	}
	fmt.Println()
}
