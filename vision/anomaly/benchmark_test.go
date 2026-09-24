package anomaly

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type benchmarkFailure struct {
	err error
}

func gatedBenchmarkLibraryPath(b *testing.B) string {
	b.Helper()
	if path := os.Getenv("RAVEN_ORT_LIB_PATH"); path != "" {
		return path
	}
	path := filepath.Join("..", "..", "lib", "onnxruntime.dll")
	if _, err := os.Stat(path); err != nil {
		b.Skipf("ONNX Runtime library is unavailable: %v", err)
	}
	return path
}

func gatedBenchmarkConfig(b *testing.B, variant string) Config {
	b.Helper()
	cfg, err := LoadConfig(filepath.Join("..", "..", "fixtures", "anomaly", variant))
	if err != nil {
		b.Fatalf("LoadConfig(%s) error = %v", variant, err)
	}
	cfg.OnnxRuntimeLibPath = gatedBenchmarkLibraryPath(b)
	cfg.ExecutionProviderPolicy = map[string]string{"cpu": "required"}
	return cfg
}

func benchmarkModelBytes(b *testing.B, cfg Config) int64 {
	b.Helper()
	info, err := os.Stat(cfg.ModelPath)
	if err != nil {
		b.Fatalf("stat benchmark model: %v", err)
	}
	return info.Size()
}

func benchmarkWarmup(b *testing.B, engine *Engine, cfg Config, streams int) {
	b.Helper()
	img := gradientImage(cfg.InputSize)
	for worker := 0; worker < streams; worker++ {
		ctx := gatedRuntimeContext(cfg)
		ctx.StreamID = fmt.Sprintf("benchmark-warmup-%02d", worker)
		result, err := engine.PredictAnomaly(img, ctx)
		if err != nil {
			b.Fatalf("warmup PredictAnomaly[%d] error = %v", worker, err)
		}
		if result == nil || !isFiniteFloat32(result.Score) {
			b.Fatalf("warmup PredictAnomaly[%d] returned invalid result", worker)
		}
	}
}

func benchmarkPercentile(sorted []time.Duration, probability float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	index := int(math.Ceil(probability*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func BenchmarkAnomalyPredict(b *testing.B) {
	for _, variant := range []string{"efficientad-s", "padim-r18"} {
		for _, streams := range []int{1, 2, 4, 8} {
			b.Run(fmt.Sprintf("%s/streams=%d", variant, streams), func(b *testing.B) {
				cfg := gatedBenchmarkConfig(b, variant)
				engine, err := NewEngine(cfg)
				if err != nil {
					b.Fatalf("NewEngine() error = %v", err)
				}
				defer engine.Destroy()

				img := gradientImage(cfg.InputSize)
				benchmarkWarmup(b, engine, cfg, streams)

				totalFrames := b.N * streams
				latencies := make([]time.Duration, 0, totalFrames)
				firstFailure := atomic.Pointer[benchmarkFailure]{}
				nextFrame := atomic.Int64{}
				latencyMutex := sync.Mutex{}

				var memBefore, memAfter runtime.MemStats
				runtime.GC()
				runtime.ReadMemStats(&memBefore)

				b.ResetTimer()
				var waitGroup sync.WaitGroup
				for worker := 0; worker < streams; worker++ {
					waitGroup.Add(1)
					go func(worker int) {
						defer waitGroup.Done()
						ctx := gatedRuntimeContext(cfg)
						ctx.StreamID = fmt.Sprintf("benchmark-stream-%02d", worker)
						for {
							if nextFrame.Add(1) > int64(totalFrames) {
								return
							}
							startedAt := time.Now()
							result, err := engine.PredictAnomaly(img, ctx)
							elapsed := time.Since(startedAt)
							if err != nil {
								firstFailure.CompareAndSwap(nil, &benchmarkFailure{err: err})
								return
							}
							if result == nil || !isFiniteFloat32(result.Score) {
								firstFailure.CompareAndSwap(nil, &benchmarkFailure{
									err: fmt.Errorf("PredictAnomaly returned invalid result for stream %d", worker),
								})
								return
							}

							latencyMutex.Lock()
							latencies = append(latencies, elapsed)
							latencyMutex.Unlock()
						}
					}(worker)
				}
				waitGroup.Wait()
				b.StopTimer()

				if failure := firstFailure.Load(); failure != nil {
					b.Fatalf("concurrent PredictAnomaly error = %v", failure.err)
				}
				if len(latencies) != totalFrames {
					b.Fatalf("completed frames = %d, want %d", len(latencies), totalFrames)
				}

				runtime.ReadMemStats(&memAfter)
				elapsed := b.Elapsed()
				sort.Slice(latencies, func(left, right int) bool {
					return latencies[left] < latencies[right]
				})

				modelBytes := benchmarkModelBytes(b, cfg)
				b.ReportMetric(float64(streams), "streams")
				b.ReportMetric(float64(totalFrames), "frames")
				b.ReportMetric(float64(elapsed.Nanoseconds())/1e6, "wall_ms")
				b.ReportMetric(float64(totalFrames)/elapsed.Seconds(), "fps")
				b.ReportMetric(float64(benchmarkPercentile(latencies, 0.50).Nanoseconds())/1e6, "p50_ms")
				b.ReportMetric(float64(benchmarkPercentile(latencies, 0.95).Nanoseconds())/1e6, "p95_ms")
				b.ReportMetric(float64(benchmarkPercentile(latencies, 0.99).Nanoseconds())/1e6, "p99_ms")
				b.ReportMetric(float64(memAfter.Mallocs-memBefore.Mallocs)/float64(totalFrames), "allocs/frame")
				b.ReportMetric(float64(memAfter.TotalAlloc-memBefore.TotalAlloc)/float64(totalFrames), "bytes/frame")
				b.ReportMetric(float64(memAfter.Alloc)/1024/1024, "go_heap_alloc_mib")
				b.ReportMetric(float64(modelBytes)/1024/1024, "model_mib")
			})
		}
	}
}
