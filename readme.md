# raven-onnxruntime

English | **[中文](./readme_zh.md)**

A pure Go ONNX Runtime visual model inference library, using [purego](https://github.com/ebitengine/purego) for zero-CGO calls to the ONNX Runtime shared library. Three core design goals: **High Performance**, **High Reliability**, and **Out-of-the-Box**.

## Core Features

### 🚀 High Performance

- **Batch Inference**: `PredictBatch(imgs []image.Image)` leverages GPU parallelism — a single ONNX `Run` call processes the entire batch, avoiding per-frame overhead
- **Hot-Path Zero Allocation**: `Session.Run()` reuses pre-allocated output name buffers and caches input name pointers via `sync.Map`, eliminating GC pressure in high-frequency inference scenarios (e.g. video streams)
- **Fast Preprocessing**: `fillCHW()` fast path directly reads underlying pixel bytes from `*image.RGBA` / `*image.NRGBA`, bypassing the `image.At().RGBA()` color conversion overhead; slow path only for unsupported image types
- **Complete SessionOptions**: Full ONNX Runtime configuration — `GraphOptimizationLevel(ALL)`, `MemPattern`, `ExecutionMode(Sequential)`, `InterOpNumThreads`, `CpuMemArena` — to maximize inference throughput
- **Segmented Timing**: Periodic per-stage timing logs (preprocess / run / postprocess) every 60 frames for easy bottleneck identification

### 🛡️ High Reliability

- **Engine Race Safety**: `NewTensor` is an `Engine` method (no global variable), eliminating data races in multi-engine scenarios
- **Input Validation**: `Session.Run()` returns an error for empty inputs instead of panicking; all output value access checks for existence before use
- **Coordinate Clamping**: Bounding box coordinates are clamped to image bounds in all model engines, preventing out-of-range drawing
- **Dynamic Library Lifecycle**: `Engine.Destroy()` properly unloads the shared library via `FreeLibrary`/`Dlclose`; `Value.Destroy()` clears cached metadata to prevent stale data access
- **Retriable Singleton**: `sync.Mutex` + state checking replaces `sync.Once`, allowing DLL load failures to be retried; stale singletons after `Destroy()` are auto-detected and recreated
- **Error Handling**: All `CopyProperties` calls check and return errors; SAM2 embedding keys are built dynamically instead of hardcoded; readable ONNX error code mapping (`ortErrorCodeDesc()`)
- **API Version Auto-Negotiation**: Functional Options + automatic downgrade — if the requested API version is unavailable, the highest compatible version is used with a warning log

### 📦 Out-of-the-Box

- **Zero CGO**: Pure Go, no C toolchain required — just the ONNX Runtime shared library
- **Dynamic I/O Names**: Automatically retrieves input/output names from the ONNX session, compatible with different model export toolchains (no hardcoded `"images"` / `"output0"`)
- **Generic NMS**: Unified `nms[T nmsCandidate]()` function — detection, pose, OBB models only need to implement `GetBox()` and `GetScore()`
- **Multi-Format Output**: Pose post-processing dynamically calculates `numObjects` / `attributes` from shape; supports both `[1, anchors, attributes]` and `[1, channels, anchors]` formats
- **Auto Input Size Detection**: RF-DETR engines parse the ONNX protobuf to detect native input resolution, with a built-in resolution table for all variants
- **Dynamic Batch Detection**: RF-DETR engines auto-detect whether the model supports dynamic batch dimensions; static-batch models gracefully fall back to sequential inference
- **Pluggable Logging**: Inject any logger implementing the `ortlog.Logger` interface via `ortlog.SetLogger()`

## Supported Models

| Model | Tasks | Package | Batch | CUDA |
|-------|-------|---------|-------|------|
| YOLO26 | Detection | `vision/yolo26` | ✅ PredictBatch | ✅ |
| YOLO26 | Segmentation | `vision/yolo26` | ✅ | ✅ |
| YOLO26 | Pose Estimation | `vision/yolo26` | ✅ PredictBatch | ✅ |
| YOLO26 | OBB | `vision/yolo26` | ✅ | ✅ |
| YOLO26 | Classification | `vision/yolo26` | ✅ | ✅ |
| YOLOv11 | Detection | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | Segmentation | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | Pose Estimation | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | OBB | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | Classification | `vision/yolov11` | ✅ | ✅ |
| RF-DETR | Detection | `vision/rfdetr` | ✅ PredictBatch | ✅ |
| RF-DETR | Segmentation | `vision/rfdetr` | ✅ PredictBatch | ✅ |
| LTDETR | Detection | `vision/ltdetr` | ✅ PredictBatch | ✅ |
| EdgeCrafter | Detection | `vision/edgecrafter` | ✅ PredictBatch | ✅ |
| EdgeCrafter | Segmentation | `vision/edgecrafter` | ✅ PredictBatch | ✅ |
| EdgeCrafter | Pose Estimation | `vision/edgecrafter` | ✅ PredictBatch | ✅ |
| SAM2 | Image Segmentation | `vision/sam2` | — | ✅ |
| SAM3 / SAM3H / SAM3.1 | Image Segmentation | `vision/sam3` | — | ✅ |
| Depth-Anything-3 | Depth Estimation | `vision/depth_anything3` | — | ✅ |

> Note: `vision/groundingdino` and `vision/groundedsam2` ship with a lightweight fallback tokenizer for local testing. It is character-based and not a production BERT tokenizer. For real open-vocabulary use, pre-tokenize captions with Python and load the cached tokenization files instead.

## Project Structure

```
raven-onnxruntime/
├── ort/                        # ONNX Runtime low-level bindings
│   ├── api.go                  # ONNX C API structs and purego function bindings
│   ├── onnxruntime.go          # Engine init, API registration, error code mapping
│   ├── session.go              # Session / SessionOptions management
│   ├── value.go                # Tensor creation and data access
│   ├── utils.go                # Common utility functions
│   ├── utils_unix.go           # Unix platform adapter
│   ├── utils_windows.go        # Windows platform adapter
│   ├── internal/sys/           # Dynamic library loading (cross-platform)
│   │   ├── dll_unix.go
│   │   └── dll_windows.go
│   └── ortlog/                 # Structured logging package
│       └── ortlog.go
├── vision/                     # Computer vision model wrappers
│   ├── onnx.go                 # OnnxConfig global Engine singleton and Session options
│   ├── draw.go                 # Detection box / label rendering
│   ├── fonts/                  # Rendering fonts
│   ├── yolo26/                 # YOLO26 models (det/seg/pose/obb/cls)
│   ├── yolov11/                # YOLOv11 models (det/seg/pose/obb/cls)
│   ├── rfdetr/                 # RF-DETR models (det/seg)
│   ├── ltdetr/                 # LTDETR models (det)
│   ├── edgecrafter/            # EdgeCrafter models (det/seg/pose)
│   ├── sam2/                   # SAM2 image segmentation
│   ├── sam3/                   # SAM3H / SAM3.1 image segmentation
│   └── depth_anything3/        # Depth-Anything-3 depth estimation
├── include/                    # ONNX Runtime C API headers
├── examples/                   # Usage examples
├── assets/                     # Project assets
├── go.mod
└── go.sum
```

## Quick Start

### Installation

```bash
go get github.com/DavidSche/raven-onnxruntime
```

### Prerequisites

- Go 1.24+
- ONNX Runtime shared library (`onnxruntime.dll` / `libonnxruntime.so` / `libonnxruntime.dylib`)
- Canonical model directory: repository root `models/` (override with `RAVEN_MODELS_DIR` if needed)
- ONNX Runtime path can be overridden with `RAVEN_ORT_LIB_PATH`

### YOLO26 Detection

```go
package main

import (
    "image"
    "os"

    "github.com/DavidSche/raven-onnxruntime/vision/yolo26"
)

func main() {
    cfg := yolo26.DefaultDetConfig()
    cfg.ModelPath = "models/yolo26/yolo26m.onnx"
    cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"
    cfg.UseCuda = true

    engine, err := yolo26.NewDetEngine(cfg)
    if err != nil {
        panic(err)
    }
    defer engine.Destroy()

    img := loadImage("test.jpg")
    results, err := engine.Predict(img)
    // Or batch inference:
    // batchResults, err := engine.PredictBatch([]image.Image{img1, img2, img3})
}
```

### YOLO26 Pose

```go
cfg := yolo26.DefaultConfig()
cfg.ModelPath = "models/yolo26/yolo26m-pose.onnx"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"
cfg.NumKeyPoints = 17

engine, err := yolo26.NewPoseEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// results[i].KeyPoints contains 17 keypoint coordinates and confidence
```

### RF-DETR Detection

```go
cfg := rfdetr.DefaultDetConfig()
cfg.ModelPath = "models/rf-detr/rf-detr-base-coco.onnx"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := rfdetr.NewDetEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// Or batch inference (auto-detects dynamic batch support):
// batchResults, err := engine.PredictBatch([]image.Image{img1, img2})
```

### LTDETR Detection

```go
cfg := ltdetr.DefaultDetConfig()
cfg.ModelPath = "models/ltdetr/dinov3_vits16-ltdetr-coco.onnx"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := ltdetr.NewDetEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// Or batch inference:
// batchResults, err := engine.PredictBatch([]image.Image{img1, img2})
```

### EdgeCrafter Detection

```go
cfg := edgecrafter.DefaultDetConfig()
cfg.ModelPath = "models/ecdet/ecdet_s.onnx"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := edgecrafter.NewDetEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// Or batch inference:
// batchResults, err := engine.PredictBatch([]image.Image{img1, img2})
```

### EdgeCrafter Segmentation

```go
cfg := edgecrafter.DefaultSegConfig()
cfg.ModelPath = "models/ecdet/ecseg_s.onnx"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := edgecrafter.NewSegEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// results[i].Mask contains the instance segmentation mask
```

### EdgeCrafter Pose Estimation

```go
cfg := edgecrafter.DefaultPoseConfig()
cfg.ModelPath = "models/ecdet/ecpose_s.onnx"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := edgecrafter.NewPoseEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// results[i].KeyPoints contains keypoint coordinates
```

### SAM2 Segmentation

```go
cfg := sam2.DefaultConfig()
cfg.ModelPath = "models/sam2"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := sam2.NewEngine(cfg)
defer engine.Destroy()

ctx, err := engine.EncodeImage(img)
defer ctx.Destroy()

points := []sam2.Point{{X: 320, Y: 240, Label: 1}}
mask, score, err := ctx.Decode(points)
```

### SAM3 Segmentation

```go
cfg := sam3.DefaultConfig()
cfg.ModelPath = "models/sam3"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := sam3.NewEngine(cfg)
defer engine.Destroy()

ctx, err := engine.EncodeImage(img)
defer ctx.Destroy()

points := []sam3.Point{{X: 320, Y: 240, Label: sam3.LabelForeground}}
mask, score, err := ctx.Decode(points)
```

### Depth-Anything-3 Depth Estimation

```go
cfg := depth_anything3.DefaultConfig()
cfg.ModelPath = "models/da3-small/da3-small_518x518.onnx"
cfg.OnnxRuntimeLibPath = "lib/onnxruntime.dll"

engine, err := depth_anything3.NewEngine(cfg)
defer engine.Destroy()

result, err := engine.Predict(img)
// result.Depth contains the depth map (H, W)
// result.Confidence contains the depth confidence map (H, W)

// Visualize: pseudo-color depth map
colorImg := depth_anything3.DepthToColormap(result)
// Visualize: depth overlay on original image
overlayImg := depth_anything3.DrawDepthOverlay(img, result, 0.5)
// Visualize: confidence-filtered depth map
confImg := depth_anything3.DrawDepthWithConfidence(result, 0.3)
```

## Logging Configuration

By default, the standard library `log` is used for output. You can inject a custom Logger via `ortlog.SetLogger()`:

```go
import "github.com/DavidSche/raven-onnxruntime/ort/ortlog"

ortlog.SetLogger(myZapLogger) // Implement the ortlog.Logger interface
```

## Dependencies

- [ebitengine/purego](https://github.com/ebitengine/purego) — Zero-CGO C shared library calls
- [up-zero/gotool](https://github.com/up-zero/gotool) — Image processing and utility functions
- [golang.org/x/image](https://pkg.go.dev/golang.org/x/image) — Image format support


## Acknowledgments

This project was initially derived from [go-vision](https://github.com/GetcharZp/go-vision) and [onnxruntime_purego](https://github.com/GetcharZp/onnxruntime_purego), reliability, and safety. Special thanks to GetcharZp!

## License & Notices

- This repository is released under the Apache 2.0 License; see [`LICENSE`](./LICENSE).
- Third-party notices are collected in [`NOTICE`](./NOTICE) and [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
- The bundled `NotoSansSC-Regular.ttf` font is licensed under the SIL Open Font License 1.1.
- If you redistribute the ONNX Runtime binary with this project, include the upstream ONNX Runtime license and notices as well.
