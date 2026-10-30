# vision/anomaly

`vision/anomaly` implements the frozen `anomaly-v1` single-frame ONNX Runtime
contract for EfficientAD-S and PaDiM-R18 RMP packages. M1-G0 has passed with
real MVTec bottle contract fixtures.

## Production Flow

```text
RMP directory
  -> LoadConfig: strict manifest / threshold report / model card / checksums
  -> NewEngine: EP policy, session, exact input/output graph contract
  -> PredictAnomaly: context gate -> preprocess -> session.Run -> postprocess
  -> AnomalyResult
```

The engine is single-frame only. It never implements `BatchAnomalyPredictor`
and never groups concurrent callers into an ONNX batch.

## Loading A Model

```go
cfg, err := anomaly.LoadConfig("fixtures/anomaly/padim-r18")
if err != nil {
    return err
}
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := anomaly.NewEngine(cfg)
if err != nil {
    return err
}
defer engine.Destroy()
```

`LoadConfig` accepts either the RMP directory or a file inside it. It rejects
unknown JSON fields, inconsistent threshold/model identities, invalid checksums,
non-`anomaly-v1` contracts, and production profiles that are not covered by
`efficientad-s-256` or `padim-r18-256`.

`NewEngine` derives and revalidates the ONNX input shape. The ONNX graph is the
authority for H/W; a manifest mismatch is a load error. Exactly `pred_score` and
`anomaly_map` are accepted; positional fallback and forbidden outputs are not.

## Runtime Context

Hard-gated context fields return `ErrRuntimeContextMismatch` before inference.
Observe-gated mismatches still run and set `AnomalyResult.ObserveOnly`; callers
must not turn observe-only results into production alerts.

The effective ROI policy is always a hard gate and must match the calibrated
canonical full-frame policy. Capture/decoded resolution, camera, lighting, and
environment obey the RMP context policy. Required monotonic frame and decode
timestamps are also validated.

## Preprocess And Region Policy

Preprocessing is out-of-graph: torchvision-compatible stretch resize to 256x256,
RGB to NCHW float32, then `divide_255` or ImageNet normalization as declared by
the RMP.

Postprocessing thresholds the raw model map, labels 8-connected components,
filters regions below `min_region_area`, sorts by peak score with deterministic
area/coverage tie-breakers, and returns at most `max_regions` regions. Region
boxes are mapped back to the passed ROI image. `pred_score` must equal
`max(anomaly_map)` within the score contract assertion.

`output_anomaly_map=true` is opt-in. When `map_downsample` is greater than one,
the returned map is pooled, while `MapSourceWidth/Height` continue to describe
the raw ONNX map used by region policy.

## Execution Provider Policy

The RMP `execution_provider_policy` takes precedence over the legacy
`UseCuda`/`UseCoreML` booleans. CPU is required as the final fallback. A required
GPU EP failure blocks loading; a preferred EP failure rebuilds CPU options and is
reported through `anomaly_ep_fallback_total`. Provider selection does not imply
full-graph execution coverage.

## Concurrency And Lifecycle

`PredictAnomaly` is safe for concurrent calls. Request-local buffers and result
ownership belong to the caller; the engine retains no mutable request state.
Counters use atomics.

`Destroy` is post-quiescence. Stop admission, wait for all in-flight calls, then
call it exactly once. It is not a synchronization primitive and must not be
issued concurrently with inference.

## Tests

```powershell
go test -count=1 ./vision/anomaly
go vet ./vision/anomaly
golangci-lint run vision/anomaly
```

The DLL-gated tests use `../../lib/onnxruntime.dll` or `RAVEN_ORT_LIB_PATH` and
exercise both real production fixtures, single-frame inference, concurrent
prediction, and post-quiescence destroy behavior.

`go test -race` requires CGO. Enable CGO and a C compiler for local race checks.

## Golden Numerical Closure

`TestAnomalyGoldenNumericalFixture` reconstructs the fixture NCHW tensor and
bypasses image preprocessing to validate ONNX Runtime and Go postprocess
closure. It checks fixture schema, opset, labels, generation seed and model
artifact identity before comparing score/map values for the complete 32-image
Golden Numerical Set (16 normal and 16 anomalous samples per model).

## Benchmark

```powershell
$env:RAVEN_ORT_LIB_PATH='..\..\lib\onnxruntime.dll'
go test ./vision/anomaly -run '^$' -bench '^BenchmarkAnomalyPredict' -benchtime=32x -count=1
```

`BenchmarkAnomalyPredict` uses the real `PredictAnomaly` path, real production
RMP fixtures, CPU EP, and one shared engine without model-key serialization.
Each benchmark operation is one synchronized batch across the labeled stream
count (1/2/4/8). The harness reports wall time, FPS, per-frame P50/P95/P99, Go
allocations/bytes per frame, current Go heap allocation, and model size. It is
an inference microbenchmark, not a capture/decode/queue/RSS or RTSP end-to-end
benchmark. Values are host-specific and must not become cross-device SLA
thresholds.

## Memory Lifecycle Benchmark

Run each variant in a separate process so the OS process peak is attributable to
that model lifecycle:

```powershell
$env:RAVEN_ORT_LIB_PATH='..\..\lib\onnxruntime.dll'
go test ./vision/anomaly -run '^$' -bench '^BenchmarkAnomalyMemoryLifecycle$' -benchtime=1x -count=1 -v

$env:RAVEN_ANOMALY_MEMORY_VARIANT='padim-r18'
go test ./vision/anomaly -run '^$' -bench '^BenchmarkAnomalyMemoryLifecycle$' -benchtime=1x -count=1 -v
Remove-Item Env:RAVEN_ANOMALY_MEMORY_VARIANT
```

The benchmark loads the real RMP, runs one warmup frame, then 64 sequential real
inference frames. It samples Windows process working set/RSS and private bytes
every 2 ms, records load/steady deltas, sampled peaks, OS peak working set,
post-destroy values, and Go heap/sys. These are process-visible values and
include the Go test process and ONNX Runtime; they are not isolated model
allocations. Results are host-specific evidence and must not become cross-device
SLA thresholds.

## Batch Profile Probe

`Engine.ProfileBatch` is an optional P2 measurement tool. It never implements
`BatchAnomalyPredictor` and never changes the production single-frame contract.
It revalidates the source RMP before running, refuses a candidate batch above
`input_limits.max_batch`, rejects mixed deployment/camera/ROI contexts, and
compares sequential single-frame, concurrent single-frame, and explicit tensor
batch candidates.

Current production fixtures declare `max_batch=1`, so only the B=1 baselines can
run against them. To evaluate B>1, export or construct a separate development
experimental RMP with a new explicit `max_batch`; do not edit a production RMP
in place. Probe results are host-specific and are not production batch
qualification.

