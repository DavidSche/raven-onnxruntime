package anomaly

import (
	"errors"
	"fmt"
	"image"
	"math"
	"runtime"
	"sort"
	"sync"
	"time"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
)

// Batch profile probe modes. A probe is deliberately distinct from a
// production batch path: it never changes the frozen single-frame contract.
const (
	BatchProfileSequential  = "single_sequential"
	BatchProfileConcurrent  = "single_concurrent"
	BatchProfileTensorBatch = "tensor_batch"
)

// BatchProfileOptions configures one non-production measurement run.
type BatchProfileOptions struct {
	// BatchSizes are experimental tensor-batch candidates. Every value must
	// be explicitly allowed by the RMP input_limits.max_batch.
	BatchSizes []int
	// WarmupIterations run before timing so memory patterns are populated.
	WarmupIterations int
	// TimedIterations is the number of complete workload measurements.
	TimedIterations int
	// IncludeConcurrent controls the multi-stream single-frame baseline.
	IncludeConcurrent bool
}

// BatchProfileMetric is one measured execution mode.
type BatchProfileMetric struct {
	Mode                   string  `json:"mode"`
	BatchSize              int     `json:"batch_size"`
	FramesPerIteration     int     `json:"frames_per_iteration"`
	WarmupIterations       int     `json:"warmup_iterations"`
	TimedIterations        int     `json:"timed_iterations"`
	WallP50MS              float64 `json:"wall_p50_ms"`
	WallP95MS              float64 `json:"wall_p95_ms"`
	WallP99MS              float64 `json:"wall_p99_ms"`
	FrameP50MS             float64 `json:"frame_p50_ms"`
	FrameP95MS             float64 `json:"frame_p95_ms"`
	FrameP99MS             float64 `json:"frame_p99_ms"`
	PreprocessP50MS        float64 `json:"preprocess_p50_ms"`
	ORTRunP50MS            float64 `json:"ort_run_p50_ms"`
	PostprocessP50MS       float64 `json:"postprocess_p50_ms"`
	FramesPerSecond        float64 `json:"frames_per_second"`
	AllocationsPerFrame    float64 `json:"allocations_per_frame"`
	AllocatedBytesPerFrame float64 `json:"allocated_bytes_per_frame"`
	GoHeapAllocMiB         float64 `json:"go_heap_alloc_mib"`
}

// BatchProfileDecision applies the fixed P2 review gate to one candidate.
// Eligibility is only a recommendation to build a separate P2 batch path; it
// does not qualify or enable production batch inference.
type BatchProfileDecision struct {
	BaselineMode            string   `json:"baseline_mode"`
	CandidateBatchSize      int      `json:"candidate_batch_size"`
	BaselineWallP95MS       float64  `json:"baseline_wall_p95_ms"`
	CandidateWallP95MS      float64  `json:"candidate_wall_p95_ms"`
	LatencyP95DeltaMS       float64  `json:"latency_p95_delta_ms"`
	LatencyWithinBaseline   bool     `json:"latency_p95_within_baseline"`
	BaselineFPS             float64  `json:"baseline_fps"`
	CandidateFPS            float64  `json:"candidate_fps"`
	ThroughputRatio         float64  `json:"throughput_ratio"`
	MinimumThroughputRatio  float64  `json:"minimum_throughput_ratio"`
	ThroughputGateMet       bool     `json:"throughput_gate_met"`
	EligibleForP2PathReview bool     `json:"eligible_for_p2_path_review"`
	Reasons                 []string `json:"reasons"`
}

// BatchProfileReport is the machine-readable probe evidence.
type BatchProfileReport struct {
	ModelPath         string                 `json:"model_path"`
	ModelSpec         string                 `json:"model_spec"`
	ValidatedVariant  string                 `json:"validated_variant"`
	RuntimeContract   string                 `json:"runtime_contract"`
	MaxRuntimeBatch   int                    `json:"max_runtime_batch"`
	MaxInputBatch     int                    `json:"max_input_batch"`
	DeploymentProfile string                 `json:"deployment_profile"`
	InputSize         image.Point            `json:"input_size"`
	ExecutionProvider map[string]string      `json:"execution_provider_policy"`
	Items             int                    `json:"items"`
	Metrics           []BatchProfileMetric   `json:"metrics"`
	ProbeOnly         bool                   `json:"probe_only"`
	Decisions         []BatchProfileDecision `json:"decisions"`
}

// ProfileBatch measures a single-frame baseline and explicitly eligible tensor
// batch candidates on one already-created engine.
//
// This is a P2 evaluation tool, not a production batch API. It does not
// implement BatchAnomalyPredictor, does not alter anomaly-v1, and does not let
// an RMP whose max_batch is 1 execute a larger tensor batch. Results are
// host-specific and must not become cross-device SLA thresholds.
func (e *Engine) ProfileBatch(items []AnomalyBatchItem, options BatchProfileOptions) (*BatchProfileReport, error) {
	if e == nil || e.session == nil {
		return nil, fmt.Errorf("%w: engine is not initialized", ErrInvalidConfig)
	}
	if e.destroyed.Load() {
		return nil, ErrEngineDestroyed
	}

	rmpConfig, err := LoadConfig(e.config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to revalidate source RMP: %w", err)
	}
	if rmpConfig.InputLimits.MaxBatch != e.config.InputLimits.MaxBatch {
		return nil, fmt.Errorf("%w: engine input_limits.max_batch %d differs from source RMP %d; reload the RMP before profiling", ErrInvalidContract, e.config.InputLimits.MaxBatch, rmpConfig.InputLimits.MaxBatch)
	}
	opts, err := normalizeBatchProfileOptions(options, len(items), e.config)
	if err != nil {
		return nil, err
	}
	prepared, err := prepareBatchProfileItems(items, e.config)
	if err != nil {
		return nil, err
	}

	report := &BatchProfileReport{
		ModelPath:         e.config.ModelPath,
		ModelSpec:         e.config.ModelSpec,
		ValidatedVariant:  e.config.ValidatedVariant,
		RuntimeContract:   RuntimeContract,
		MaxRuntimeBatch:   e.config.RuntimeMaxBatch,
		MaxInputBatch:     e.config.InputLimits.MaxBatch,
		DeploymentProfile: e.config.DeploymentProfile,
		InputSize:         e.config.InputSize,
		ExecutionProvider: copyExecutionProviderPolicy(e.config.ExecutionProviderPolicy),
		Items:             len(items),
		ProbeOnly:         true,
	}

	if err := warmupBatchProfile(e, prepared, opts); err != nil {
		return nil, err
	}

	metric, err := measureSequentialBatchProfile(e, prepared, opts)
	if err != nil {
		return nil, err
	}
	report.Metrics = append(report.Metrics, *metric)

	if opts.IncludeConcurrent {
		metric, err = measureConcurrentBatchProfile(e, prepared, opts)
		if err != nil {
			return nil, err
		}
		report.Metrics = append(report.Metrics, *metric)
	}

	for _, batchSize := range opts.BatchSizes {
		metric, err = measureTensorBatchProfile(e, prepared, opts, batchSize)
		if err != nil {
			return nil, err
		}
		report.Metrics = append(report.Metrics, *metric)
	}
	sort.Slice(report.Metrics, func(left, right int) bool {
		if report.Metrics[left].Mode != report.Metrics[right].Mode {
			return report.Metrics[left].Mode < report.Metrics[right].Mode
		}
		return report.Metrics[left].BatchSize < report.Metrics[right].BatchSize
	})
	report.Decisions = evaluateBatchProfileDecisions(report.Metrics)
	return report, nil
}

func evaluateBatchProfileDecisions(metrics []BatchProfileMetric) []BatchProfileDecision {
	var baseline *BatchProfileMetric
	for index := range metrics {
		if metrics[index].Mode == BatchProfileSequential {
			baseline = &metrics[index]
			break
		}
	}
	if baseline == nil {
		return []BatchProfileDecision{}
	}

	decisions := make([]BatchProfileDecision, 0)
	for _, metric := range metrics {
		if metric.Mode != BatchProfileTensorBatch {
			continue
		}
		latencyWithin := metric.WallP95MS <= baseline.WallP95MS
		throughputRatio := 0.0
		if baseline.FramesPerSecond > 0 {
			throughputRatio = metric.FramesPerSecond / baseline.FramesPerSecond
		}
		throughputMet := throughputRatio >= 2
		reasons := make([]string, 0, 2)
		if latencyWithin {
			reasons = append(reasons, "p95_latency_within_serial_baseline")
		} else {
			reasons = append(reasons, "p95_latency_exceeds_serial_baseline")
		}
		if throughputMet {
			reasons = append(reasons, "throughput_ratio_meets_2x")
		} else {
			reasons = append(reasons, "throughput_ratio_below_2x")
		}
		decisions = append(decisions, BatchProfileDecision{
			BaselineMode:            BatchProfileSequential,
			CandidateBatchSize:      metric.BatchSize,
			BaselineWallP95MS:       baseline.WallP95MS,
			CandidateWallP95MS:      metric.WallP95MS,
			LatencyP95DeltaMS:       metric.WallP95MS - baseline.WallP95MS,
			LatencyWithinBaseline:   latencyWithin,
			BaselineFPS:             baseline.FramesPerSecond,
			CandidateFPS:            metric.FramesPerSecond,
			ThroughputRatio:         throughputRatio,
			MinimumThroughputRatio:  2,
			ThroughputGateMet:       throughputMet,
			EligibleForP2PathReview: latencyWithin && throughputMet,
			Reasons:                 reasons,
		})
	}
	return decisions
}

func normalizeBatchProfileOptions(options BatchProfileOptions, itemCount int, cfg Config) (BatchProfileOptions, error) {
	if itemCount <= 0 {
		return BatchProfileOptions{}, fmt.Errorf("%w: at least one batch item is required", ErrInvalidConfig)
	}
	if options.TimedIterations <= 0 {
		return BatchProfileOptions{}, fmt.Errorf("%w: timed iterations must be positive", ErrInvalidConfig)
	}
	if options.WarmupIterations < 0 {
		return BatchProfileOptions{}, fmt.Errorf("%w: warmup iterations cannot be negative", ErrInvalidConfig)
	}
	if options.TimedIterations > 1024 || options.WarmupIterations > 64 || itemCount > 128 {
		return BatchProfileOptions{}, fmt.Errorf("%w: batch profile workload exceeds bounded evaluation limits", ErrInvalidConfig)
	}
	if cfg.RuntimeBatchMode != "single" || cfg.RuntimeMaxBatch != 1 {
		return BatchProfileOptions{}, fmt.Errorf("%w: batch profile requires an anomaly-v1 single-frame config", ErrInvalidConfig)
	}
	if cfg.InputLimits.MaxBatch < 1 {
		return BatchProfileOptions{}, fmt.Errorf("%w: input limits do not permit a batch", ErrInvalidConfig)
	}

	batches := make([]int, 0, len(options.BatchSizes))
	seen := make(map[int]bool, len(options.BatchSizes))
	for _, size := range options.BatchSizes {
		if size < 1 {
			return BatchProfileOptions{}, fmt.Errorf("%w: batch size %d must be positive", ErrInvalidConfig, size)
		}
		if size > cfg.InputLimits.MaxBatch {
			return BatchProfileOptions{}, fmt.Errorf(
				"%w: requested tensor batch %d exceeds RMP input_limits.max_batch %d; export a new explicitly batch-eligible RMP for P2 evaluation",
				ErrInvalidConfig, size, cfg.InputLimits.MaxBatch,
			)
		}
		if seen[size] {
			return BatchProfileOptions{}, fmt.Errorf("%w: duplicate batch size %d", ErrInvalidConfig, size)
		}
		seen[size] = true
		batches = append(batches, size)
	}
	sort.Ints(batches)
	return BatchProfileOptions{
		BatchSizes:        batches,
		WarmupIterations:  options.WarmupIterations,
		TimedIterations:   options.TimedIterations,
		IncludeConcurrent: options.IncludeConcurrent,
	}, nil
}

func prepareBatchProfileItems(items []AnomalyBatchItem, cfg Config) ([]preparedBatchItem, error) {
	prepared := make([]preparedBatchItem, 0, len(items))
	for index, item := range items {
		observeOnly, err := validateRuntimeContext(item.Image, item.Ctx, cfg)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", index, err)
		}
		tensorData, original, err := buildPreprocessTensorData(item.Image, cfg)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", index, err)
		}
		prepared = append(prepared, preparedBatchItem{
			image:       item.Image,
			tensorData:  tensorData,
			original:    original,
			ctx:         item.Ctx,
			observeOnly: observeOnly,
		})
	}
	if err := validateBatchProfileDeployment(prepared); err != nil {
		return nil, err
	}
	return prepared, nil
}

type preparedBatchItem struct {
	image       image.Image
	tensorData  []float32
	original    preprocessResult
	ctx         AnomalyRuntimeContext
	observeOnly bool
}

func (p preparedBatchItem) tensorDataImage() image.Image { return p.image }
func validateBatchProfileDeployment(prepared []preparedBatchItem) error {
	first := prepared[0].ctx
	for index, item := range prepared[1:] {
		ctx := item.ctx
		if ctx.DeploymentBindingID != first.DeploymentBindingID ||
			ctx.CameraID != first.CameraID ||
			ctx.CaptureResolution != first.CaptureResolution ||
			ctx.DecodedResolution != first.DecodedResolution ||
			ctx.ROIResolution != first.ROIResolution ||
			ctx.ROIPolicyHash != first.ROIPolicyHash ||
			ctx.ROIPolicyVersion != first.ROIPolicyVersion ||
			ctx.LightingProfile != first.LightingProfile ||
			ctx.Environment != first.Environment {
			return fmt.Errorf(
				"%w: batch items 0 and %d use different deployment/camera/context bindings; tensor batches must not mix camera or ROI contexts",
				ErrInvalidConfig, index+1,
			)
		}
	}
	return nil
}

func warmupBatchProfile(e *Engine, prepared []preparedBatchItem, opts BatchProfileOptions) error {
	for iteration := 0; iteration < opts.WarmupIterations; iteration++ {
		for _, item := range prepared {
			if _, err := e.PredictAnomaly(item.tensorDataImage(), item.ctx); err != nil {
				return fmt.Errorf("single-frame warmup: %w", err)
			}
		}
		for _, batchSize := range opts.BatchSizes {
			if _, err := runBatchedAnomaly(e, prepared, batchSize); err != nil {
				return fmt.Errorf("tensor batch %d warmup: %w", batchSize, err)
			}
		}
	}
	return nil
}

func measureSequentialBatchProfile(e *Engine, prepared []preparedBatchItem, opts BatchProfileOptions) (*BatchProfileMetric, error) {
	walls := make([]time.Duration, 0, opts.TimedIterations)
	frames := make([]time.Duration, 0, opts.TimedIterations*len(prepared))
	mem := collectProfileMemory(e, prepared, opts, func() error {
		for iteration := 0; iteration < opts.TimedIterations; iteration++ {
			startedAt := time.Now()
			for _, item := range prepared {
				frameStart := time.Now()
				result, err := e.PredictAnomaly(item.tensorDataImage(), item.ctx)
				frames = append(frames, time.Since(frameStart))
				if err != nil {
					return err
				}
				if result == nil || !isFiniteFloat32(result.Score) {
					return errors.New("sequential profile returned an invalid anomaly result")
				}
			}
			walls = append(walls, time.Since(startedAt))
		}
		return nil
	})
	if mem.err != nil {
		return nil, mem.err
	}
	return &BatchProfileMetric{
		Mode:                   BatchProfileSequential,
		BatchSize:              1,
		FramesPerIteration:     len(prepared),
		WarmupIterations:       opts.WarmupIterations,
		TimedIterations:        opts.TimedIterations,
		WallP50MS:              durationPercentileMS(walls, 0.50),
		WallP95MS:              durationPercentileMS(walls, 0.95),
		WallP99MS:              durationPercentileMS(walls, 0.99),
		FrameP50MS:             durationPercentileMS(frames, 0.50),
		FrameP95MS:             durationPercentileMS(frames, 0.95),
		FrameP99MS:             durationPercentileMS(frames, 0.99),
		FramesPerSecond:        framesPerSecond(walls, len(prepared)),
		AllocationsPerFrame:    memoryAllocationsPerFrame(mem, len(prepared)*opts.TimedIterations),
		AllocatedBytesPerFrame: memoryBytesPerFrame(mem, len(prepared)*opts.TimedIterations),
		GoHeapAllocMiB:         mem.heapAllocMiB,
	}, nil
}

func measureConcurrentBatchProfile(e *Engine, prepared []preparedBatchItem, opts BatchProfileOptions) (*BatchProfileMetric, error) {
	walls := make([]time.Duration, 0, opts.TimedIterations)
	frames := make([]time.Duration, 0, opts.TimedIterations*len(prepared))
	var measurementMutex sync.Mutex
	mem := collectProfileMemory(e, prepared, opts, func() error {
		var waitGroup sync.WaitGroup
		for iteration := 0; iteration < opts.TimedIterations; iteration++ {
			startedAt := time.Now()
			firstError := make(chan error, 1)
			for index, item := range prepared {
				waitGroup.Add(1)
				go func(index int, item preparedBatchItem) {
					defer waitGroup.Done()
					frameStart := time.Now()
					result, err := e.PredictAnomaly(item.tensorDataImage(), item.ctx)
					elapsed := time.Since(frameStart)
					measurementMutex.Lock()
					frames = append(frames, elapsed)
					measurementMutex.Unlock()
					if err != nil {
						select {
						case firstError <- err:
						default:
						}
						return
					}
					if result == nil || !isFiniteFloat32(result.Score) {
						select {
						case firstError <- fmt.Errorf("concurrent profile item %d returned an invalid anomaly result", index):
						default:
						}
					}
				}(index, item)
			}
			waitGroup.Wait()
			select {
			case err := <-firstError:
				return err
			default:
			}
			walls = append(walls, time.Since(startedAt))
		}
		return nil
	})
	if mem.err != nil {
		return nil, mem.err
	}
	return &BatchProfileMetric{
		Mode:                   BatchProfileConcurrent,
		BatchSize:              1,
		FramesPerIteration:     len(prepared),
		WarmupIterations:       opts.WarmupIterations,
		TimedIterations:        opts.TimedIterations,
		WallP50MS:              durationPercentileMS(walls, 0.50),
		WallP95MS:              durationPercentileMS(walls, 0.95),
		WallP99MS:              durationPercentileMS(walls, 0.99),
		FrameP50MS:             durationPercentileMS(frames, 0.50),
		FrameP95MS:             durationPercentileMS(frames, 0.95),
		FrameP99MS:             durationPercentileMS(frames, 0.99),
		FramesPerSecond:        framesPerSecond(walls, len(prepared)),
		AllocationsPerFrame:    memoryAllocationsPerFrame(mem, len(prepared)*opts.TimedIterations),
		AllocatedBytesPerFrame: memoryBytesPerFrame(mem, len(prepared)*opts.TimedIterations),
		GoHeapAllocMiB:         mem.heapAllocMiB,
	}, nil
}

func measureTensorBatchProfile(e *Engine, prepared []preparedBatchItem, opts BatchProfileOptions, batchSize int) (*BatchProfileMetric, error) {
	walls := make([]time.Duration, 0, opts.TimedIterations)
	preprocessDurations := make([]time.Duration, 0, opts.TimedIterations)
	runDurations := make([]time.Duration, 0, opts.TimedIterations)
	postprocessDurations := make([]time.Duration, 0, opts.TimedIterations)
	mem := collectProfileMemory(e, prepared, opts, func() error {
		for iteration := 0; iteration < opts.TimedIterations; iteration++ {
			startedAt := time.Now()
			preprocessStart := time.Now()
			preparedInputs, err := combinePreparedTensors(prepared, batchSize, e.config)
			if err != nil {
				return err
			}
			preprocessElapsed := time.Since(preprocessStart)

			var runElapsed, postprocessElapsed time.Duration
			for _, input := range preparedInputs {
				results, elapsed, runErr := runPreparedBatch(e, input)
				runElapsed += elapsed
				runDurations = append(runDurations, elapsed)
				if runErr != nil {
					return runErr
				}
				postprocessStart := time.Now()
				for index, result := range results {
					if result == nil || !isFiniteFloat32(result.Score) {
						return fmt.Errorf("tensor batch %d returned invalid result %d", input.batchSize, index)
					}
				}
				postprocessElapsed += time.Since(postprocessStart)
			}
			postprocessDurations = append(postprocessDurations, postprocessElapsed)
			preprocessDurations = append(preprocessDurations, preprocessElapsed)
			walls = append(walls, time.Since(startedAt))
		}
		return nil
	})
	if mem.err != nil {
		return nil, mem.err
	}
	return &BatchProfileMetric{
		Mode:                   BatchProfileTensorBatch,
		BatchSize:              batchSize,
		FramesPerIteration:     len(prepared),
		WarmupIterations:       opts.WarmupIterations,
		TimedIterations:        opts.TimedIterations,
		WallP50MS:              durationPercentileMS(walls, 0.50),
		WallP95MS:              durationPercentileMS(walls, 0.95),
		WallP99MS:              durationPercentileMS(walls, 0.99),
		FrameP50MS:             durationPercentileMS(walls, 0.50) / float64(len(prepared)),
		FrameP95MS:             durationPercentileMS(walls, 0.95) / float64(len(prepared)),
		FrameP99MS:             durationPercentileMS(walls, 0.99) / float64(len(prepared)),
		PreprocessP50MS:        durationPercentileMS(preprocessDurations, 0.50),
		ORTRunP50MS:            durationPercentileMS(runDurations, 0.50),
		PostprocessP50MS:       durationPercentileMS(postprocessDurations, 0.50),
		FramesPerSecond:        framesPerSecond(walls, len(prepared)),
		AllocationsPerFrame:    memoryAllocationsPerFrame(mem, len(prepared)*opts.TimedIterations),
		AllocatedBytesPerFrame: memoryBytesPerFrame(mem, len(prepared)*opts.TimedIterations),
		GoHeapAllocMiB:         mem.heapAllocMiB,
	}, nil
}

func collectProfileMemory(e *Engine, prepared []preparedBatchItem, opts BatchProfileOptions, measure func() error) profileMemory {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	err := measure()
	runtime.ReadMemStats(&after)
	return profileMemory{
		allocations:  after.Mallocs - before.Mallocs,
		bytes:        after.TotalAlloc - before.TotalAlloc,
		heapAllocMiB: float64(after.Alloc) / 1024 / 1024,
		err:          err,
	}
}

type profileMemory struct {
	allocations  uint64
	bytes        uint64
	heapAllocMiB float64
	err          error
}

func memoryAllocationsPerFrame(mem profileMemory, frames int) float64 {
	if frames == 0 {
		return 0
	}
	return float64(mem.allocations) / float64(frames)
}

func memoryBytesPerFrame(mem profileMemory, frames int) float64 {
	if frames == 0 {
		return 0
	}
	return float64(mem.bytes) / float64(frames)
}

func framesPerSecond(walls []time.Duration, frames int) float64 {
	if frames <= 0 || len(walls) == 0 {
		return 0
	}
	var total time.Duration
	for _, wall := range walls {
		total += wall
	}
	seconds := total.Seconds()
	if seconds <= 0 {
		return 0
	}
	return float64(frames*len(walls)) / seconds
}

func durationPercentileMS(values []time.Duration, probability float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(values))
	copy(sorted, values)
	sort.Slice(sorted, func(left, right int) bool { return sorted[left] < sorted[right] })
	index := int(math.Ceil(probability*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return float64(sorted[index].Nanoseconds()) / 1e6
}

func copyExecutionProviderPolicy(policy map[string]string) map[string]string {
	if policy == nil {
		return nil
	}
	output := make(map[string]string, len(policy))
	for key, value := range policy {
		output[key] = value
	}
	return output
}

func combinePreparedTensors(prepared []preparedBatchItem, batchSize int, cfg Config) ([]preparedTensorBatch, error) {
	if batchSize < 1 {
		return nil, fmt.Errorf("%w: tensor batch size must be positive", ErrInvalidConfig)
	}
	planeSize := cfg.InputSize.X * cfg.InputSize.Y
	elements := 3 * planeSize
	batches := make([]preparedTensorBatch, 0, (len(prepared)+batchSize-1)/batchSize)
	for start := 0; start < len(prepared); start += batchSize {
		end := min(start+batchSize, len(prepared))
		size := end - start
		data := make([]float32, 0, size*elements)
		batchOriginals := make([]preprocessResult, 0, size)
		for _, item := range prepared[start:end] {
			if len(item.tensorData) != elements {
				return nil, fmt.Errorf("%w: prepared tensor has %d elements, want %d", ErrInvalidConfig, len(item.tensorData), elements)
			}
			data = append(data, item.tensorData...)
			batchOriginals = append(batchOriginals, item.original)
		}
		batches = append(batches, preparedTensorBatch{
			batchSize: size,
			data:      data,
			originals: batchOriginals,
		})
	}
	return batches, nil
}

type preparedTensorBatch struct {
	batchSize int
	data      []float32
	originals []preprocessResult
}

func runBatchedAnomaly(e *Engine, prepared []preparedBatchItem, batchSize int) ([]*AnomalyResult, error) {
	inputs, err := combinePreparedTensors(prepared, batchSize, e.config)
	if err != nil {
		return nil, err
	}
	results := make([]*AnomalyResult, 0, len(prepared))
	preparedOffset := 0
	for _, input := range inputs {
		resultsForBatch, err := runPreparedBatchResults(e, input)
		if err != nil {
			return nil, err
		}
		for index, result := range resultsForBatch {
			result.ObserveOnly = prepared[preparedOffset+index].observeOnly
		}
		preparedOffset += input.batchSize
		results = append(results, resultsForBatch...)
	}
	if len(results) != len(prepared) {
		return nil, fmt.Errorf("%w: tensor batch returned %d results, want %d", ErrInvalidContract, len(results), len(prepared))
	}
	return results, nil
}
func runPreparedBatch(e *Engine, batch preparedTensorBatch) ([]*AnomalyResult, time.Duration, error) {
	startedAt := time.Now()
	results, err := runPreparedBatchResults(e, batch)
	return results, time.Since(startedAt), err
}

func runPreparedBatchResults(e *Engine, batch preparedTensorBatch) ([]*AnomalyResult, error) {
	shape := []int64{int64(batch.batchSize), 3, int64(e.config.InputSize.Y), int64(e.config.InputSize.X)}
	inputTensor, err := e.session.NewTensor(shape, batch.data)
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor batch input: %w", err)
	}
	defer inputTensor.Destroy()

	outputs, err := e.session.Run(map[string]*ort.Value{e.session.InputNames[0]: inputTensor})
	if err != nil {
		return nil, fmt.Errorf("tensor batch inference failed: %w", err)
	}
	defer ort.DestroyValues(outputs)

	return splitBatchOutputs(outputs, e.config, batch.batchSize, batch.originals)
}

func splitBatchOutputs(outputValues map[string]*ort.Value, cfg Config, batchSize int, originals []preprocessResult) ([]*AnomalyResult, error) {
	if len(originals) != batchSize {
		return nil, fmt.Errorf("%w: tensor batch has %d originals, want %d", ErrInvalidContract, len(originals), batchSize)
	}
	if len(outputValues) != cfg.OutputContract.OutputCount {
		return nil, fmt.Errorf("%w: tensor batch got %d outputs, want exactly %d", ErrInvalidContract, len(outputValues), cfg.OutputContract.OutputCount)
	}
	var scoreOutput, mapOutput *ort.Value
	for name, value := range outputValues {
		switch name {
		case "pred_score":
			scoreOutput = value
		case "anomaly_map":
			mapOutput = value
		default:
			return nil, fmt.Errorf("%w: unexpected tensor batch output %q", ErrInvalidContract, name)
		}
	}
	if scoreOutput == nil || mapOutput == nil {
		return nil, fmt.Errorf("%w: tensor batch requires pred_score and anomaly_map", ErrInvalidContract)
	}

	scoreShape, err := scoreOutput.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor batch score shape: %w", err)
	}
	if len(scoreShape) != 1 || scoreShape[0] != int64(batchSize) {
		return nil, fmt.Errorf("%w: tensor batch score shape %v, want [%d]", ErrInvalidContract, scoreShape, batchSize)
	}
	scoreData, err := ort.GetTensorData[float32](scoreOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor batch scores: %w", err)
	}
	if len(scoreData) != batchSize {
		return nil, fmt.Errorf("%w: tensor batch score has %d values, want %d", ErrInvalidContract, len(scoreData), batchSize)
	}

	mapShape, err := mapOutput.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor batch map shape: %w", err)
	}
	if len(mapShape) != 4 || mapShape[0] != int64(batchSize) || mapShape[1] != 1 {
		return nil, fmt.Errorf("%w: tensor batch map shape %v, want [%d,1,H,W]", ErrInvalidContract, mapShape, batchSize)
	}
	mapHeight, mapWidth := int(mapShape[2]), int(mapShape[3])
	if mapHeight < 1 || mapWidth < 1 || mapHeight > math.MaxInt/mapWidth {
		return nil, fmt.Errorf("%w: tensor batch map dimensions are invalid", ErrInvalidContract)
	}
	plane := mapHeight * mapWidth
	mapData, err := ort.GetTensorData[float32](mapOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor batch maps: %w", err)
	}
	if int64(len(mapData)) != int64(batchSize)*int64(plane) {
		return nil, fmt.Errorf("%w: tensor batch map has %d values, want %d", ErrInvalidContract, len(mapData), batchSize*plane)
	}

	results := make([]*AnomalyResult, 0, batchSize)
	for index := 0; index < batchSize; index++ {
		sourceMap := make([]float32, plane)
		copy(sourceMap, mapData[index*plane:(index+1)*plane])
		result, err := postprocessData(scoreData[index], sourceMap, mapWidth, mapHeight, cfg, originals[index])
		if err != nil {
			return nil, err
		}
		results = append(results, result)
	}
	return results, nil
}

// test
