# raven-onnxruntime

**[English](./readme.md)** | 中文

纯 Go 语言的 ONNX Runtime 视觉模型推理共享库，通过 [purego](https://github.com/ebitengine/purego) 零 CGO 调用 ONNX Runtime 共享库。三大核心设计目标：**高性能**、**高可靠性**、**开箱即用**。

## 核心特性

### 🚀 高性能

- **批量推理**：`PredictBatch(imgs []image.Image)` 充分利用 GPU 并行能力 — 一次 ONNX `Run` 调用处理整个批次，避免逐帧开销
- **热路径零分配**：`Session.Run()` 复用预分配的输出名称缓冲区，通过 `sync.Map` 缓存输入名称指针，消除高频推理场景（如视频流）下的 GC 压力
- **快速预处理**：`fillCHW()` 快速路径直接读取 `*image.RGBA` / `*image.NRGBA` 底层像素字节，绕过 `image.At().RGBA()` 颜色转换开销；慢速路径仅用于不支持的图像类型
- **完整 SessionOptions**：完整配置 ONNX Runtime — `GraphOptimizationLevel(ALL)`、`MemPattern`、`ExecutionMode(Sequential)`、`InterOpNumThreads`、`CpuMemArena` — 最大化推理吞吐量
- **分段耗时埋点**：每 60 帧输出一次分阶段耗时日志（预处理 / 推理 / 后处理），方便快速定位瓶颈

### 🛡️ 高可靠性

- **Engine 竞争安全**：`NewTensor` 是 `Engine` 方法（无全局变量），消除多引擎场景下的数据竞争
- **输入校验**：`Session.Run()` 对空输入返回错误而非 panic；所有输出值访问前均检查存在性
- **坐标边界约束**：所有模型引擎的边界框坐标均 clamp 到图像边界内，防止越界绘制
- **动态库生命周期**：`Engine.Destroy()` 通过 `FreeLibrary`/`Dlclose` 正确卸载共享库；`Value.Destroy()` 清空缓存元数据防止过期数据访问
- **可重试单例**：`sync.Mutex` + 状态判断替代 `sync.Once`，DLL 加载失败可重试；`Destroy()` 后的过期单例自动检测并重建
- **错误处理**：所有 `CopyProperties` 调用均检查并返回错误；SAM2 embedding key 动态构建而非硬编码；ONNX 错误码可读映射（`ortErrorCodeDesc()`）
- **API 版本自动协商**：Functional Options + 自动降级 — 请求的 API 版本不可用时，自动使用最高兼容版本并输出 Warn 日志

### 📦 开箱即用

- **零 CGO**：纯 Go 实现，无需 C 工具链 — 只需 ONNX Runtime 共享库即可运行
- **动态 I/O 名称**：自动从 ONNX session 获取输入/输出名称，兼容不同模型导出工具链（不再硬编码 `"images"` / `"output0"`）
- **泛型 NMS**：统一 `nms[T nmsCandidate]()` 函数 — 检测、姿态、OBB 模型只需实现 `GetBox()` 和 `GetScore()`
- **多格式输出**：姿态后处理根据 shape 动态计算 `numObjects` / `attributes`；同时支持 `[1, anchors, attributes]` 和 `[1, channels, anchors]` 两种格式
- **自动输入尺寸检测**：RF-DETR 引擎解析 ONNX protobuf 检测原生输入分辨率，内置全变体分辨率表
- **动态 Batch 检测**：RF-DETR 引擎自动检测模型是否支持动态 batch 维度，静态 batch 模型优雅退化为逐帧推理
- **可插拔日志**：通过 `ortlog.SetLogger()` 注入任何实现 `ortlog.Logger` 接口的日志器

## 支持的模型

| 模型 | 任务 | 包路径 | 批量推理 | CUDA |
|------|------|--------|----------|------|
| YOLO26 | 检测 | `vision/yolo26` | ✅ PredictBatch | ✅ |
| YOLO26 | 分割 | `vision/yolo26` | ✅ | ✅ |
| YOLO26 | 姿态估计 | `vision/yolo26` | ✅ PredictBatch | ✅ |
| YOLO26 | OBB | `vision/yolo26` | ✅ | ✅ |
| YOLO26 | 分类 | `vision/yolo26` | ✅ | ✅ |
| YOLOv11 | 检测 | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | 分割 | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | 姿态估计 | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | OBB | `vision/yolov11` | ✅ | ✅ |
| YOLOv11 | 分类 | `vision/yolov11` | ✅ | ✅ |
| RF-DETR | 检测 | `vision/rfdetr` | ✅ PredictBatch | ✅ |
| RF-DETR | 分割 | `vision/rfdetr` | ✅ PredictBatch | ✅ |
| SAM2 | 图像分割 | `vision/sam2` | — | ✅ |
| SAM3 / SAM3H / SAM3.1 | 图像分割 | `vision/sam3` | — | ✅ |

## 项目结构

```
raven-onnxruntime/
├── ort/                        # ONNX Runtime 底层封装
│   ├── api.go                  # ONNX C API 结构体与 purego 函数绑定
│   ├── onnxruntime.go          # Engine 初始化、API 注册、错误码映射
│   ├── session.go              # Session / SessionOptions 管理
│   ├── value.go                # Tensor 创建与数据读取
│   ├── utils.go                # 通用工具函数
│   ├── utils_unix.go           # Unix 平台适配
│   ├── utils_windows.go        # Windows 平台适配
│   ├── internal/sys/           # 动态库加载（跨平台）
│   │   ├── dll_unix.go
│   │   └── dll_windows.go
│   └── ortlog/                 # 结构化日志包
│       └── ortlog.go
├── vision/                     # 计算机视觉模型封装
│   ├── onnx.go                 # OnnxConfig 全局 Engine 单例与 Session 选项配置
│   ├── draw.go                 # 检测框/标签绘制
│   ├── fonts/                  # 绘制字体
│   ├── yolo26/                 # YOLO26 模型（det/seg/pose/obb/cls）
│   ├── yolov11/                # YOLOv11 模型（det/seg/pose/obb/cls）
│   ├── rfdetr/                 # RF-DETR 模型（det/seg）
│   ├── sam2/                   # SAM2 图像分割
│   └── sam3/                   # SAM3H / SAM3.1 图像分割
├── include/                    # ONNX Runtime C API 头文件
├── examples/                   # 使用示例
├── assets/                     # 项目资源
├── go.mod
└── go.sum
```

## 快速开始

### 安装

```bash
go get github.com/DavidSche/raven-onnxruntime
```

### 前置条件

- Go 1.24+
- ONNX Runtime 共享库（`onnxruntime.dll` / `libonnxruntime.so` / `libonnxruntime.dylib`）

### YOLO26 检测

```go
package main

import (
    "image"
    "os"

    "github.com/DavidSche/raven-onnxruntime/vision/yolo26"
)

func main() {
    cfg := yolo26.DefaultDetConfig()
    cfg.ModelPath = "yolo26n-det.onnx"
    cfg.OnnxRuntimeLibPath = "./lib/onnxruntime.dll"
    cfg.UseCuda = true

    engine, err := yolo26.NewDetEngine(cfg)
    if err != nil {
        panic(err)
    }
    defer engine.Destroy()

    img := loadImage("test.jpg")
    results, err := engine.Predict(img)
    // 或批量推理：
    // batchResults, err := engine.PredictBatch([]image.Image{img1, img2, img3})
}
```

### YOLO26 姿态

```go
cfg := yolo26.DefaultConfig()
cfg.ModelPath = "yolo26n-pose.onnx"
cfg.OnnxRuntimeLibPath = "./lib/onnxruntime.dll"
cfg.NumKeyPoints = 17

engine, err := yolo26.NewPoseEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// results[i].KeyPoints 包含 17 个关键点坐标与置信度
```

### RF-DETR 检测

```go
cfg := rfdetr.DefaultDetConfig()
cfg.ModelPath = "rf-detr-base-coco.onnx"
cfg.OnnxRuntimeLibPath = "./lib/onnxruntime.dll"

engine, err := rfdetr.NewDetEngine(cfg)
defer engine.Destroy()

results, err := engine.Predict(img)
// 或批量推理（自动检测动态 batch 支持）：
// batchResults, err := engine.PredictBatch([]image.Image{img1, img2})
```

### SAM2 分割

```go
cfg := sam2.DefaultConfig()
cfg.EncodeModelPath = "sam2_encoder.onnx"
cfg.DecodeModelPath = "sam2_decoder.onnx"
cfg.OnnxRuntimeLibPath = "./lib/onnxruntime.dll"

engine, err := sam2.NewEngine(cfg)
defer engine.Destroy()

ctx, err := engine.EncodeImage(img)
defer ctx.Destroy()

points := []sam2.Point{{X: 320, Y: 240, Label: 1}}
mask, score, err := ctx.Decode(points)
```

### SAM3 分割

```go
cfg := sam3.DefaultConfig()
cfg.VisionModelPath = "sam3_vision_encoder.onnx"
cfg.TextModelPath = "sam3_text_encoder.onnx"
cfg.DecoderModelPath = "sam3_decoder.onnx"
cfg.OnnxRuntimeLibPath = "./lib/onnxruntime.dll"

engine, err := sam3.NewEngine(cfg)
defer engine.Destroy()

ctx, err := engine.EncodeImage(img)
defer ctx.Destroy()

points := []sam3.Point{{X: 320, Y: 240, Label: sam3.LabelForeground}}
mask, score, err := ctx.Decode(points)
```

## 日志配置

默认使用标准库 `log` 输出。可通过 `ortlog.SetLogger()` 注入自定义 Logger：

```go
import "github.com/DavidSche/raven-onnxruntime/ort/ortlog"

ortlog.SetLogger(myZapLogger) // 实现 ortlog.Logger 接口
```

## 依赖

- [ebitengine/purego](https://github.com/ebitengine/purego) — 零 CGO 调用 C 共享库
- [up-zero/gotool](https://github.com/up-zero/gotool) — 图像处理与工具函数
- [golang.org/x/image](https://pkg.go.dev/golang.org/x/image) — 图像格式支持


## 致谢

本项目代码最初来源于 [go-vision](https://github.com/GetcharZp/go-vision) 和 [onnxruntime_purego](https://github.com/GetcharZp/onnxruntime_purego) 两个独立仓库，在此对 GetcharZp 表示感谢！

## 许可与声明

- 本仓库采用 Apache 2.0 许可，见 [`LICENSE`](./LICENSE)。
- 第三方声明已汇总到 [`NOTICE`](./NOTICE) 和 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。
- 随仓库附带的 `NotoSansSC-Regular.ttf` 字体采用 SIL Open Font License 1.1。
- 如果你在分发本项目时同时分发 ONNX Runtime 二进制文件，请一并附带其上游许可与声明。
