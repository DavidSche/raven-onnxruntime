package anomaly

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

const anomalyMemoryBenchmarkFrames = 64

type memoryPeakTracker struct {
	mutex       sync.Mutex
	peakRSS     uint64
	peakPrivate uint64
}

func (tracker *memoryPeakTracker) observe(stats processMemoryStats) {
	tracker.mutex.Lock()
	defer tracker.mutex.Unlock()
	tracker.peakRSS = maxUint64(tracker.peakRSS, stats.workingSetBytes)
	tracker.peakPrivate = maxUint64(tracker.peakPrivate, stats.privateBytes)
}

func (tracker *memoryPeakTracker) peaks() (uint64, uint64) {
	tracker.mutex.Lock()
	defer tracker.mutex.Unlock()
	return tracker.peakRSS, tracker.peakPrivate
}

func reportMemoryMiB(b *testing.B, name string, bytesValue uint64) {
	b.Helper()
	if bytesValue > 0 {
		b.ReportMetric(float64(bytesValue)/(1024*1024), name)
	}
}

func reportSignedMemoryMiB(b *testing.B, name string, bytesValue int64) {
	b.Helper()
	b.ReportMetric(float64(bytesValue)/(1024*1024), name)
}

func BenchmarkAnomalyMemoryLifecycle(b *testing.B) {
	variant := os.Getenv("RAVEN_ANOMALY_MEMORY_VARIANT")
	if variant == "" {
		variant = "efficientad-s"
	}
	b.Run(fmt.Sprintf("variant=%s/frames=%d/mode=sequential", variant, anomalyMemoryBenchmarkFrames), func(b *testing.B) {
		if _, err := readProcessMemoryStats(); err != nil {
			b.Skipf("process memory sampler unavailable: %v", err)
		}
		if b.N != 1 {
			b.Fatalf("memory lifecycle must run with -benchtime=1x, got b.N=%d", b.N)
		}

		cfg := gatedBenchmarkConfig(b, variant)
		var baselineMemory, loadedMemory, warmedMemory, steadyMemory, postDestroyMemory processMemoryStats
		var baselineHeap, steadyHeap runtime.MemStats

		runtime.GC()
		runtime.ReadMemStats(&baselineHeap)
		baselineMemory, err := readProcessMemoryStats()
		if err != nil {
			b.Fatalf("read baseline process memory: %v", err)
		}
		tracker := &memoryPeakTracker{}
		tracker.observe(baselineMemory)
		stopMonitor := make(chan struct{})
		monitorDone := make(chan struct{})
		go func() {
			defer close(monitorDone)
			ticker := time.NewTicker(2 * time.Millisecond)
			defer ticker.Stop()
			for {
				select {
				case <-stopMonitor:
					if stats, err := readProcessMemoryStats(); err == nil {
						tracker.observe(stats)
					}
					return
				case <-ticker.C:
					if stats, err := readProcessMemoryStats(); err == nil {
						tracker.observe(stats)
					}
				}
			}
		}()

		startedAt := time.Now()
		engine, err := NewEngine(cfg)
		initDuration := time.Since(startedAt)
		if err != nil {
			close(stopMonitor)
			<-monitorDone
			b.Fatalf("NewEngine() error = %v", err)
		}
		loadedMemory, err = readProcessMemoryStats()
		if err != nil {
			engine.Destroy()
			close(stopMonitor)
			<-monitorDone
			b.Fatalf("read loaded process memory: %v", err)
		}
		tracker.observe(loadedMemory)

		img := gradientImage(cfg.InputSize)
		ctx := gatedRuntimeContext(cfg)
		ctx.StreamID = "memory-benchmark"
		warmupResult, err := engine.PredictAnomaly(img, ctx)
		if err != nil {
			engine.Destroy()
			close(stopMonitor)
			<-monitorDone
			b.Fatalf("warmup PredictAnomaly error = %v", err)
		}
		if warmupResult == nil || !isFiniteFloat32(warmupResult.Score) {
			engine.Destroy()
			close(stopMonitor)
			<-monitorDone
			b.Fatalf("warmup PredictAnomaly returned invalid result")
		}
		warmedMemory, err = readProcessMemoryStats()
		if err != nil {
			engine.Destroy()
			close(stopMonitor)
			<-monitorDone
			b.Fatalf("read warmup process memory: %v", err)
		}
		tracker.observe(warmedMemory)

		for frame := 0; frame < anomalyMemoryBenchmarkFrames; frame++ {
			result, predictErr := engine.PredictAnomaly(img, ctx)
			if predictErr != nil {
				engine.Destroy()
				close(stopMonitor)
				<-monitorDone
				b.Fatalf("PredictAnomaly[%d] error = %v", frame, predictErr)
			}
			if result == nil || !isFiniteFloat32(result.Score) {
				engine.Destroy()
				close(stopMonitor)
				<-monitorDone
				b.Fatalf("PredictAnomaly[%d] returned invalid result", frame)
			}
		}

		steadyMemory, err = readProcessMemoryStats()
		if err != nil {
			engine.Destroy()
			close(stopMonitor)
			<-monitorDone
			b.Fatalf("read steady-state process memory: %v", err)
		}
		tracker.observe(steadyMemory)
		runtime.GC()
		runtime.ReadMemStats(&steadyHeap)

		engine.Destroy()
		postDestroyMemory, err = readProcessMemoryStats()
		close(stopMonitor)
		<-monitorDone
		if err != nil {
			b.Fatalf("read post-destroy process memory: %v", err)
		}

		peakRSS, peakPrivate := tracker.peaks()
		modelBytes := benchmarkModelBytes(b, cfg)
		b.ReportMetric(float64(anomalyMemoryBenchmarkFrames), "frames")
		b.ReportMetric(initDuration.Seconds()*1000, "init_ms")
		reportMemoryMiB(b, "model_mib", uint64(modelBytes))
		reportMemoryMiB(b, "baseline_rss_mib", baselineMemory.workingSetBytes)
		reportMemoryMiB(b, "baseline_private_mib", baselineMemory.privateBytes)
		reportMemoryMiB(b, "load_rss_mib", loadedMemory.workingSetBytes)
		reportMemoryMiB(b, "load_private_mib", loadedMemory.privateBytes)
		reportSignedMemoryMiB(b, "load_rss_delta_mib", int64(loadedMemory.workingSetBytes)-int64(baselineMemory.workingSetBytes))
		reportSignedMemoryMiB(b, "load_private_delta_mib", int64(loadedMemory.privateBytes)-int64(baselineMemory.privateBytes))
		reportMemoryMiB(b, "warmup_rss_mib", warmedMemory.workingSetBytes)
		reportMemoryMiB(b, "warmup_private_mib", warmedMemory.privateBytes)
		reportMemoryMiB(b, "steady_rss_mib", steadyMemory.workingSetBytes)
		reportMemoryMiB(b, "steady_private_mib", steadyMemory.privateBytes)
		reportSignedMemoryMiB(b, "steady_rss_delta_mib", int64(steadyMemory.workingSetBytes)-int64(warmedMemory.workingSetBytes))
		reportSignedMemoryMiB(b, "steady_private_delta_mib", int64(steadyMemory.privateBytes)-int64(warmedMemory.privateBytes))
		reportMemoryMiB(b, "sampled_peak_rss_mib", peakRSS)
		reportMemoryMiB(b, "sampled_peak_private_mib", peakPrivate)
		reportMemoryMiB(b, "os_peak_rss_mib", loadedMemory.peakWorkingSetBytes)
		reportMemoryMiB(b, "post_destroy_rss_mib", postDestroyMemory.workingSetBytes)
		reportMemoryMiB(b, "post_destroy_private_mib", postDestroyMemory.privateBytes)
		reportMemoryMiB(b, "steady_go_heap_alloc_mib", steadyHeap.Alloc)
		reportMemoryMiB(b, "steady_go_sys_mib", steadyHeap.Sys)
	})
}

func maxUint64(left, right uint64) uint64 {
	if left > right {
		return left
	}
	return right
}
