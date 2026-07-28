package examples

import (
	"fmt"
	"image"
	"math"
	"sort"
	"time"

	"github.com/DavidSche/raven-onnxruntime/vision/dfine"
	"github.com/DavidSche/raven-onnxruntime/vision/edgecrafter"
	"github.com/DavidSche/raven-onnxruntime/vision/ltdetr"
	"github.com/DavidSche/raven-onnxruntime/vision/rfdetr"
	"github.com/DavidSche/raven-onnxruntime/vision/yolo26"
	"github.com/DavidSche/raven-onnxruntime/vision/yolov11"
)

// BenchEngine is a unified interface for benchmarking any vision model engine.
type BenchEngine interface {
	Name() string
	Task() string
	Predict(img image.Image) (int, error)
	PredictBatch(imgs []image.Image) (int, error)
	SupportsBatch() bool
	Destroy()
}

// --- YOLO26 Adapters ---

type yolo26DetAdapter struct {
	engine *yolo26.DetEngine
	name   string
}

func (a *yolo26DetAdapter) Name() string        { return a.name }
func (a *yolo26DetAdapter) Task() string        { return "det" }
func (a *yolo26DetAdapter) SupportsBatch() bool { return true }
func (a *yolo26DetAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolo26DetAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolo26DetAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

type yolo26SegAdapter struct {
	engine *yolo26.SegEngine
	name   string
}

func (a *yolo26SegAdapter) Name() string        { return a.name }
func (a *yolo26SegAdapter) Task() string        { return "seg" }
func (a *yolo26SegAdapter) SupportsBatch() bool { return false }
func (a *yolo26SegAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolo26SegAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolo26SegAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

type yolo26PoseAdapter struct {
	engine *yolo26.PoseEngine
	name   string
}

func (a *yolo26PoseAdapter) Name() string        { return a.name }
func (a *yolo26PoseAdapter) Task() string        { return "pose" }
func (a *yolo26PoseAdapter) SupportsBatch() bool { return true }
func (a *yolo26PoseAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolo26PoseAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolo26PoseAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

type yolo26OBBAdapter struct {
	engine *yolo26.OBBEngine
	name   string
}

func (a *yolo26OBBAdapter) Name() string        { return a.name }
func (a *yolo26OBBAdapter) Task() string        { return "obb" }
func (a *yolo26OBBAdapter) SupportsBatch() bool { return false }
func (a *yolo26OBBAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolo26OBBAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolo26OBBAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

type yolo26ClsAdapter struct {
	engine *yolo26.ClsEngine
	name   string
}

func (a *yolo26ClsAdapter) Name() string        { return a.name }
func (a *yolo26ClsAdapter) Task() string        { return "cls" }
func (a *yolo26ClsAdapter) SupportsBatch() bool { return false }
func (a *yolo26ClsAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolo26ClsAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img, 5)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolo26ClsAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

// --- YOLOv11 Adapters ---

type yolov11DetAdapter struct {
	engine *yolov11.DetEngine
	name   string
}

func (a *yolov11DetAdapter) Name() string        { return a.name }
func (a *yolov11DetAdapter) Task() string        { return "det" }
func (a *yolov11DetAdapter) SupportsBatch() bool { return false }
func (a *yolov11DetAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolov11DetAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolov11DetAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

type yolov11SegAdapter struct {
	engine *yolov11.SegEngine
	name   string
}

func (a *yolov11SegAdapter) Name() string        { return a.name }
func (a *yolov11SegAdapter) Task() string        { return "seg" }
func (a *yolov11SegAdapter) SupportsBatch() bool { return false }
func (a *yolov11SegAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolov11SegAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolov11SegAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

type yolov11PoseAdapter struct {
	engine *yolov11.PoseEngine
	name   string
}

func (a *yolov11PoseAdapter) Name() string        { return a.name }
func (a *yolov11PoseAdapter) Task() string        { return "pose" }
func (a *yolov11PoseAdapter) SupportsBatch() bool { return false }
func (a *yolov11PoseAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolov11PoseAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolov11PoseAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

type yolov11OBBAdapter struct {
	engine *yolov11.OBBEngine
	name   string
}

func (a *yolov11OBBAdapter) Name() string        { return a.name }
func (a *yolov11OBBAdapter) Task() string        { return "obb" }
func (a *yolov11OBBAdapter) SupportsBatch() bool { return false }
func (a *yolov11OBBAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolov11OBBAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolov11OBBAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

type yolov11ClsAdapter struct {
	engine *yolov11.ClsEngine
	name   string
}

func (a *yolov11ClsAdapter) Name() string        { return a.name }
func (a *yolov11ClsAdapter) Task() string        { return "cls" }
func (a *yolov11ClsAdapter) SupportsBatch() bool { return false }
func (a *yolov11ClsAdapter) Destroy()            { a.engine.Destroy() }
func (a *yolov11ClsAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img, 5)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *yolov11ClsAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

// --- RF-DETR Adapters ---

type rfdetrDetAdapter struct {
	engine *rfdetr.DetEngine
	name   string
}

func (a *rfdetrDetAdapter) Name() string        { return a.name }
func (a *rfdetrDetAdapter) Task() string        { return "det" }
func (a *rfdetrDetAdapter) SupportsBatch() bool { return true }
func (a *rfdetrDetAdapter) Destroy()            { a.engine.Destroy() }
func (a *rfdetrDetAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *rfdetrDetAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

type rfdetrSegAdapter struct {
	engine *rfdetr.SegEngine
	name   string
}

func (a *rfdetrSegAdapter) Name() string        { return a.name }
func (a *rfdetrSegAdapter) Task() string        { return "seg" }
func (a *rfdetrSegAdapter) SupportsBatch() bool { return true }
func (a *rfdetrSegAdapter) Destroy()            { a.engine.Destroy() }
func (a *rfdetrSegAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *rfdetrSegAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

// --- LTDETR Adapters ---

type ltdetrDetAdapter struct {
	engine *ltdetr.DetEngine
	name   string
}

func (a *ltdetrDetAdapter) Name() string        { return a.name }
func (a *ltdetrDetAdapter) Task() string        { return "det" }
func (a *ltdetrDetAdapter) SupportsBatch() bool { return true }
func (a *ltdetrDetAdapter) Destroy()            { a.engine.Destroy() }
func (a *ltdetrDetAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *ltdetrDetAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

// --- EdgeCrafter Adapters ---

type edgecrafterDetAdapter struct {
	engine *edgecrafter.DetEngine
	name   string
}

func (a *edgecrafterDetAdapter) Name() string        { return a.name }
func (a *edgecrafterDetAdapter) Task() string        { return "det" }
func (a *edgecrafterDetAdapter) SupportsBatch() bool { return true }
func (a *edgecrafterDetAdapter) Destroy()            { a.engine.Destroy() }
func (a *edgecrafterDetAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *edgecrafterDetAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

type edgecrafterSegAdapter struct {
	engine *edgecrafter.SegEngine
	name   string
}

func (a *edgecrafterSegAdapter) Name() string        { return a.name }
func (a *edgecrafterSegAdapter) Task() string        { return "seg" }
func (a *edgecrafterSegAdapter) SupportsBatch() bool { return true }
func (a *edgecrafterSegAdapter) Destroy()            { a.engine.Destroy() }
func (a *edgecrafterSegAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *edgecrafterSegAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

type edgecrafterPoseAdapter struct {
	engine *edgecrafter.PoseEngine
	name   string
}

func (a *edgecrafterPoseAdapter) Name() string        { return a.name }
func (a *edgecrafterPoseAdapter) Task() string        { return "pose" }
func (a *edgecrafterPoseAdapter) SupportsBatch() bool { return true }
func (a *edgecrafterPoseAdapter) Destroy()            { a.engine.Destroy() }
func (a *edgecrafterPoseAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *edgecrafterPoseAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

// --- Engine Factory ---

// EngineSpec defines a model engine to be benchmarked.
type EngineSpec struct {
	Name      string
	Task      string
	ModelPath string
	LibPath   string
	UseCuda   bool
	DynBatch  bool
}

// CreateEngine creates a BenchEngine from an EngineSpec.
func CreateEngine(spec EngineSpec) (BenchEngine, error) {
	switch spec.Name {
	case "yolo26":
		return createYOLO26Engine(spec)
	case "yolov11":
		return createYOLOv11Engine(spec)
	case "rfdetr":
		return createRFDETRengine(spec)
	case "ltdetr":
		return createLTDETRengine(spec)
	case "dfine":
		return createDFINEengine(spec)
	case "edgecrafter":
		return createEdgeCrafterEngine(spec)
	default:
		return nil, fmt.Errorf("unknown model: %s", spec.Name)
	}
}

func createYOLO26Engine(spec EngineSpec) (BenchEngine, error) {
	switch spec.Task {
	case "det":
		cfg := yolo26.DefaultDetConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolo26.NewDetEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolo26DetAdapter{engine: e, name: "YOLO26-det"}, nil
	case "seg":
		cfg := yolo26.DefaultSegConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolo26.NewSegEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolo26SegAdapter{engine: e, name: "YOLO26-seg"}, nil
	case "pose":
		cfg := yolo26.DefaultPoseConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolo26.NewPoseEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolo26PoseAdapter{engine: e, name: "YOLO26-pose"}, nil
	case "obb":
		cfg := yolo26.DefaultOBBConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolo26.NewOBBEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolo26OBBAdapter{engine: e, name: "YOLO26-obb"}, nil
	case "cls":
		cfg := yolo26.DefaultClsConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolo26.NewClsEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolo26ClsAdapter{engine: e, name: "YOLO26-cls"}, nil
	default:
		return nil, fmt.Errorf("unsupported YOLO26 task: %s", spec.Task)
	}
}

func createYOLOv11Engine(spec EngineSpec) (BenchEngine, error) {
	switch spec.Task {
	case "det":
		cfg := yolov11.DefaultDetConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolov11.NewDetEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolov11DetAdapter{engine: e, name: "YOLOv11-det"}, nil
	case "seg":
		cfg := yolov11.DefaultSegConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolov11.NewSegEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolov11SegAdapter{engine: e, name: "YOLOv11-seg"}, nil
	case "pose":
		cfg := yolov11.DefaultPoseConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolov11.NewPoseEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolov11PoseAdapter{engine: e, name: "YOLOv11-pose"}, nil
	case "obb":
		cfg := yolov11.DefaultOBBConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolov11.NewOBBEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolov11OBBAdapter{engine: e, name: "YOLOv11-obb"}, nil
	case "cls":
		cfg := yolov11.DefaultClsConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		e, err := yolov11.NewClsEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &yolov11ClsAdapter{engine: e, name: "YOLOv11-cls"}, nil
	default:
		return nil, fmt.Errorf("unsupported YOLOv11 task: %s", spec.Task)
	}
}

func createRFDETRengine(spec EngineSpec) (BenchEngine, error) {
	switch spec.Task {
	case "det":
		cfg := rfdetr.DefaultDetConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := rfdetr.NewDetEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &rfdetrDetAdapter{engine: e, name: "RF-DETR-det"}, nil
	case "seg":
		cfg := rfdetr.DefaultSegConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := rfdetr.NewSegEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &rfdetrSegAdapter{engine: e, name: "RF-DETR-seg"}, nil
	default:
		return nil, fmt.Errorf("unsupported RF-DETR task: %s", spec.Task)
	}
}

func createLTDETRengine(spec EngineSpec) (BenchEngine, error) {
	switch spec.Task {
	case "det":
		cfg := ltdetr.DefaultDetConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := ltdetr.NewDetEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &ltdetrDetAdapter{engine: e, name: "LTDETR-det"}, nil
	default:
		return nil, fmt.Errorf("unsupported LTDETR task: %s", spec.Task)
	}
}

// --- D-FINE Adapters ---

type dfineDetAdapter struct {
	engine *dfine.DetEngine
	name   string
}

func (a *dfineDetAdapter) Name() string        { return a.name }
func (a *dfineDetAdapter) Task() string        { return "det" }
func (a *dfineDetAdapter) SupportsBatch() bool { return true }
func (a *dfineDetAdapter) Destroy()            { a.engine.Destroy() }
func (a *dfineDetAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *dfineDetAdapter) PredictBatch(imgs []image.Image) (int, error) {
	res, err := a.engine.PredictBatch(imgs)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, r := range res {
		total += len(r)
	}
	return total, nil
}

type dfineSegAdapter struct {
	engine *dfine.SegEngine
	name   string
}

func (a *dfineSegAdapter) Name() string        { return a.name }
func (a *dfineSegAdapter) Task() string        { return "seg" }
func (a *dfineSegAdapter) SupportsBatch() bool { return false }
func (a *dfineSegAdapter) Destroy()            { a.engine.Destroy() }
func (a *dfineSegAdapter) Predict(img image.Image) (int, error) {
	res, err := a.engine.Predict(img)
	if err != nil {
		return 0, err
	}
	return len(res), nil
}
func (a *dfineSegAdapter) PredictBatch(imgs []image.Image) (int, error) {
	return 0, nil
}

func createDFINEengine(spec EngineSpec) (BenchEngine, error) {
	switch spec.Task {
	case "det":
		cfg := dfine.DefaultDetConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := dfine.NewDetEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &dfineDetAdapter{engine: e, name: "D-FINE-det"}, nil
	case "seg":
		cfg := dfine.DefaultSegConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := dfine.NewSegEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &dfineSegAdapter{engine: e, name: "D-FINE-seg"}, nil
	default:
		return nil, fmt.Errorf("unsupported D-FINE task: %s", spec.Task)
	}
}

// --- EdgeCrafter Adapters ---

func createEdgeCrafterEngine(spec EngineSpec) (BenchEngine, error) {
	switch spec.Task {
	case "det":
		cfg := edgecrafter.DefaultDetConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := edgecrafter.NewDetEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &edgecrafterDetAdapter{engine: e, name: "EdgeCrafter-det"}, nil
	case "seg":
		cfg := edgecrafter.DefaultSegConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := edgecrafter.NewSegEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &edgecrafterSegAdapter{engine: e, name: "EdgeCrafter-seg"}, nil
	case "pose":
		cfg := edgecrafter.DefaultPoseConfig()
		cfg.ModelPath = spec.ModelPath
		cfg.OnnxRuntimeLibPath = spec.LibPath
		cfg.UseCuda = spec.UseCuda
		cfg.DynamicBatch = spec.DynBatch
		e, err := edgecrafter.NewPoseEngine(cfg)
		if err != nil {
			return nil, err
		}
		return &edgecrafterPoseAdapter{engine: e, name: "EdgeCrafter-pose"}, nil
	default:
		return nil, fmt.Errorf("unsupported EdgeCrafter task: %s", spec.Task)
	}
}

// percentile calculates the p-th percentile from a sorted slice of durations.
func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := float64(len(sorted)-1) * p / 100.0
	lower := int(idx)
	upper := lower + 1
	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	frac := idx - float64(lower)
	return sorted[lower] + time.Duration(frac*float64(sorted[upper]-sorted[lower]))
}

// BenchSummary holds aggregated benchmark results for a single engine.
type BenchSummary struct {
	EngineName  string
	Task        string
	TotalFrames int
	TotalDets   int
	ErrorCount  int
	TotalMs     time.Duration
	AvgMs       time.Duration
	MinMs       time.Duration
	MaxMs       time.Duration
	P50Ms       time.Duration
	P95Ms       time.Duration
	P99Ms       time.Duration
	StdDevMs    time.Duration // standard deviation of latencies
	CV          float64       // coefficient of variation (StdDev/Avg), measures stability
	FPS         float64
	Latencies   []time.Duration

	// Segmented timing (optional, populated when engine supports it)
	PreprocessMs  time.Duration // average preprocessing time
	InferenceMs   time.Duration // average pure inference time
	PostprocessMs time.Duration // average postprocessing time

	// Memory (optional, populated when monitoring is enabled)
	PeakMemoryMB float64 // peak Go runtime memory in MB
}

// ComputeSummary computes a BenchSummary from per-frame latencies and detection counts.
func ComputeSummary(engineName, task string, latencies []time.Duration, detCounts []int, errorCount int) BenchSummary {
	n := len(latencies)
	if n == 0 {
		return BenchSummary{EngineName: engineName, Task: task, ErrorCount: errorCount}
	}

	sorted := make([]time.Duration, n)
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var totalMs time.Duration
	var totalDets int
	for i := range latencies {
		totalMs += latencies[i]
		if i < len(detCounts) {
			totalDets += detCounts[i]
		}
	}

	avgMs := totalMs / time.Duration(n)

	// Compute standard deviation and coefficient of variation
	stdDevMs := time.Duration(0)
	cv := 0.0
	if n > 1 {
		stdDevFloat := 0.0
		for _, lat := range latencies {
			diff := float64(lat-avgMs) / float64(time.Millisecond)
			stdDevFloat += diff * diff
		}
		stdDevFloat = math.Sqrt(stdDevFloat / float64(n))
		stdDevMs = time.Duration(stdDevFloat * float64(time.Millisecond))
		if avgMs > 0 {
			cv = stdDevFloat / (float64(avgMs) / float64(time.Millisecond))
		}
	}

	return BenchSummary{
		EngineName:  engineName,
		Task:        task,
		TotalFrames: n,
		TotalDets:   totalDets,
		ErrorCount:  errorCount,
		TotalMs:     totalMs,
		AvgMs:       avgMs,
		MinMs:       sorted[0],
		MaxMs:       sorted[n-1],
		P50Ms:       percentile(sorted, 50),
		P95Ms:       percentile(sorted, 95),
		P99Ms:       percentile(sorted, 99),
		StdDevMs:    stdDevMs,
		CV:          cv,
		FPS:         1.0 / avgMs.Seconds(),
		Latencies:   sorted,
	}
}
