package examples

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/DavidSche/raven-onnxruntime/vision/dfine"
	"github.com/DavidSche/raven-onnxruntime/vision/edgecrafter"
	"github.com/DavidSche/raven-onnxruntime/vision/ltdetr"
	"github.com/DavidSche/raven-onnxruntime/vision/rfdetr"
	"github.com/DavidSche/raven-onnxruntime/vision/yolo26"
	"github.com/DavidSche/raven-onnxruntime/vision/yolov11"
)

// ============================================================
// Accuracy Engine Interface
// ============================================================

// AccuracyEngine wraps a model engine for accuracy evaluation.
type AccuracyEngine interface {
	Name() string
	Task() string
	PredictDetections(img image.Image) ([]Detection, error)
	Destroy()
}

// ============================================================
// YOLO26 Accuracy Engines
// ============================================================

type yolo26DetAccEngine struct {
	engine *yolo26.DetEngine
	name   string
}

func NewYOLO26DetAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := yolo26.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001, // low threshold for mAP evaluation
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
	}
	eng, err := yolo26.NewDetEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create YOLO26 DetEngine: %w", err)
	}
	// Derive name from model filename (e.g. "yolo26n.onnx" → "yolo26n")
	base := filepath.Base(modelPath)
	name := strings.TrimSuffix(base, filepath.Ext(base))
	return &yolo26DetAccEngine{engine: eng, name: name}, nil
}

func (e *yolo26DetAccEngine) Name() string { return e.name }
func (e *yolo26DetAccEngine) Task() string { return "det" }
func (e *yolo26DetAccEngine) Destroy()     { e.engine.Destroy() }

func (e *yolo26DetAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type yolo26SegAccEngine struct {
	engine *yolo26.SegEngine
	name   string
}

func NewYOLO26SegAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := yolo26.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
		NumMaskCoeffs:      32,
	}
	eng, err := yolo26.NewSegEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create YOLO26 SegEngine: %w", err)
	}
	return &yolo26SegAccEngine{engine: eng, name: strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))}, nil
}

func (e *yolo26SegAccEngine) Name() string { return e.name }
func (e *yolo26SegAccEngine) Task() string { return "seg" }
func (e *yolo26SegAccEngine) Destroy()     { e.engine.Destroy() }

func (e *yolo26SegAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type yolo26PoseAccEngine struct {
	engine *yolo26.PoseEngine
	name   string
}

func NewYOLO26PoseAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := yolo26.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         1,
		NumKeyPoints:       17,
	}
	eng, err := yolo26.NewPoseEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create YOLO26 PoseEngine: %w", err)
	}
	return &yolo26PoseAccEngine{engine: eng, name: strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))}, nil
}

func (e *yolo26PoseAccEngine) Name() string { return e.name }
func (e *yolo26PoseAccEngine) Task() string { return "pose" }
func (e *yolo26PoseAccEngine) Destroy()     { e.engine.Destroy() }

func (e *yolo26PoseAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

// ============================================================
// YOLOv11 Accuracy Engines
// ============================================================

type yolov11DetAccEngine struct {
	engine *yolov11.DetEngine
	name   string
}

func NewYOLOv11DetAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := yolov11.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
	}
	eng, err := yolov11.NewDetEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create YOLOv11 DetEngine: %w", err)
	}
	return &yolov11DetAccEngine{engine: eng, name: strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))}, nil
}

func (e *yolov11DetAccEngine) Name() string { return e.name }
func (e *yolov11DetAccEngine) Task() string { return "det" }
func (e *yolov11DetAccEngine) Destroy()     { e.engine.Destroy() }

func (e *yolov11DetAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type yolov11SegAccEngine struct {
	engine *yolov11.SegEngine
	name   string
}

func NewYOLOv11SegAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := yolov11.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
		NumMaskCoeffs:      32,
	}
	eng, err := yolov11.NewSegEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create YOLOv11 SegEngine: %w", err)
	}
	return &yolov11SegAccEngine{engine: eng, name: strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))}, nil
}

func (e *yolov11SegAccEngine) Name() string { return e.name }
func (e *yolov11SegAccEngine) Task() string { return "seg" }
func (e *yolov11SegAccEngine) Destroy()     { e.engine.Destroy() }

func (e *yolov11SegAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type yolov11PoseAccEngine struct {
	engine *yolov11.PoseEngine
	name   string
}

func NewYOLOv11PoseAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := yolov11.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         1,
		NumKeyPoints:       17,
	}
	eng, err := yolov11.NewPoseEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create YOLOv11 PoseEngine: %w", err)
	}
	return &yolov11PoseAccEngine{engine: eng, name: strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))}, nil
}

func (e *yolov11PoseAccEngine) Name() string { return e.name }
func (e *yolov11PoseAccEngine) Task() string { return "pose" }
func (e *yolov11PoseAccEngine) Destroy()     { e.engine.Destroy() }

func (e *yolov11PoseAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

// ============================================================
// RF-DETR Accuracy Engines
// ============================================================

type rfdetrDetAccEngine struct {
	engine *rfdetr.DetEngine
	name   string
}

func NewRFDETRDetAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := rfdetr.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
	}
	eng, err := rfdetr.NewDetEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create RF-DETR DetEngine: %w", err)
	}
	return &rfdetrDetAccEngine{engine: eng, name: "RF-DETR-det"}, nil
}

func (e *rfdetrDetAccEngine) Name() string { return e.name }
func (e *rfdetrDetAccEngine) Task() string { return "det" }
func (e *rfdetrDetAccEngine) Destroy()     { e.engine.Destroy() }

func (e *rfdetrDetAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type rfdetrSegAccEngine struct {
	engine *rfdetr.SegEngine
	name   string
}

func NewRFDETRSegAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := rfdetr.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
	}
	eng, err := rfdetr.NewSegEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create RF-DETR SegEngine: %w", err)
	}
	return &rfdetrSegAccEngine{engine: eng, name: "RF-DETR-seg"}, nil
}

func (e *rfdetrSegAccEngine) Name() string { return e.name }
func (e *rfdetrSegAccEngine) Task() string { return "seg" }
func (e *rfdetrSegAccEngine) Destroy()     { e.engine.Destroy() }

func (e *rfdetrSegAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

// ============================================================
// LTDETR Accuracy Engine
// ============================================================

type ltdetrDetAccEngine struct {
	engine *ltdetr.DetEngine
	name   string
}

func NewLTDETRDetAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := ltdetr.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		InputSize:          640,
		NumClasses:         80,
	}
	eng, err := ltdetr.NewDetEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create LTDETR DetEngine: %w", err)
	}
	return &ltdetrDetAccEngine{engine: eng, name: "LTDETR-det"}, nil
}

func (e *ltdetrDetAccEngine) Name() string { return e.name }
func (e *ltdetrDetAccEngine) Task() string { return "det" }
func (e *ltdetrDetAccEngine) Destroy()     { e.engine.Destroy() }

func (e *ltdetrDetAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

// ============================================================
// D-FINE Accuracy Engines
// ============================================================

type dfineDetAccEngine struct {
	engine *dfine.DetEngine
	name   string
}

func NewDFINEDetAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := dfine.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		InputSize:          640,
		NumClasses:         80,
		MaxDetections:      300,
		PreprocessConfig:   vision.DefaultImageNetPreprocessConfig(),
	}
	eng, err := dfine.NewDetEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create D-FINE DetEngine: %w", err)
	}
	base := filepath.Base(modelPath)
	name := strings.TrimSuffix(base, filepath.Ext(base))
	return &dfineDetAccEngine{engine: eng, name: name}, nil
}

func (e *dfineDetAccEngine) Name() string { return e.name }
func (e *dfineDetAccEngine) Task() string { return "det" }
func (e *dfineDetAccEngine) Destroy()     { e.engine.Destroy() }

func (e *dfineDetAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type dfineSegAccEngine struct {
	engine *dfine.SegEngine
	name   string
}

func NewDFINESegAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := dfine.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		InputSize:          640,
		NumClasses:         80,
		MaxDetections:      300,
		NumMasks:           300,
		MaskHeight:         160,
		MaskWidth:          160,
		MaskThreshold:      0.5,
		PreprocessConfig:   vision.DefaultImageNetPreprocessConfig(),
	}
	eng, err := dfine.NewSegEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create D-FINE SegEngine: %w", err)
	}
	return &dfineSegAccEngine{engine: eng, name: strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))}, nil
}

func (e *dfineSegAccEngine) Name() string { return e.name }
func (e *dfineSegAccEngine) Task() string { return "seg" }
func (e *dfineSegAccEngine) Destroy()     { e.engine.Destroy() }

func (e *dfineSegAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

// ============================================================
// EdgeCrafter Accuracy Engines
// ============================================================

type ecDetAccEngine struct {
	engine *edgecrafter.DetEngine
	name   string
}

func NewECDetAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := edgecrafter.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
	}
	eng, err := edgecrafter.NewDetEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create EdgeCrafter DetEngine: %w", err)
	}
	return &ecDetAccEngine{engine: eng, name: "EdgeCrafter-det"}, nil
}

func (e *ecDetAccEngine) Name() string { return e.name }
func (e *ecDetAccEngine) Task() string { return "det" }
func (e *ecDetAccEngine) Destroy()     { e.engine.Destroy() }

func (e *ecDetAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type ecSegAccEngine struct {
	engine *edgecrafter.SegEngine
	name   string
}

func NewECSegAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := edgecrafter.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         80,
	}
	eng, err := edgecrafter.NewSegEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create EdgeCrafter SegEngine: %w", err)
	}
	return &ecSegAccEngine{engine: eng, name: "EdgeCrafter-seg"}, nil
}

func (e *ecSegAccEngine) Name() string { return e.name }
func (e *ecSegAccEngine) Task() string { return "seg" }
func (e *ecSegAccEngine) Destroy()     { e.engine.Destroy() }

func (e *ecSegAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

type ecPoseAccEngine struct {
	engine *edgecrafter.PoseEngine
	name   string
}

func NewECPoseAccEngine(modelPath, libPath string, useCuda bool) (AccuracyEngine, error) {
	cfg := edgecrafter.Config{
		OnnxRuntimeLibPath: libPath,
		ModelPath:          modelPath,
		UseCuda:            useCuda,
		ConfThreshold:      0.001,
		IOUThreshold:       0.65,
		InputSize:          640,
		NumClasses:         2,
		NumBodyPoints:      17,
	}
	eng, err := edgecrafter.NewPoseEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create EdgeCrafter PoseEngine: %w", err)
	}
	return &ecPoseAccEngine{engine: eng, name: "EdgeCrafter-pose"}, nil
}

func (e *ecPoseAccEngine) Name() string { return e.name }
func (e *ecPoseAccEngine) Task() string { return "pose" }
func (e *ecPoseAccEngine) Destroy()     { e.engine.Destroy() }

func (e *ecPoseAccEngine) PredictDetections(img image.Image) ([]Detection, error) {
	results, err := e.engine.Predict(img)
	if err != nil {
		return nil, err
	}
	dets := make([]Detection, len(results))
	for i, r := range results {
		dets[i] = Detection{ClassID: r.ClassID, Score: float64(r.Score), BBox: RectToBBox(r.Box)}
	}
	return dets, nil
}

// ============================================================
// Accuracy Evaluation Runner
// ============================================================

// AccuracySpec defines an accuracy evaluation specification.
type AccuracySpec struct {
	Name      string
	Task      string
	ModelPath string
	LibPath   string
	UseCuda   bool
}

// EvaluateAccuracy runs accuracy evaluation for a single engine against COCO ground truths.
func EvaluateAccuracy(engine AccuracyEngine, cocoDS *COCODataset, imageDir string, maxImages int) (MAPResult, error) {
	catToClass := CategoryIDToClassID()
	imageEvals := make([]ImageEval, 0)

	processed := 0
	for _, cocoImg := range cocoDS.Images {
		if maxImages > 0 && processed >= maxImages {
			break
		}

		imgPath := filepath.Join(imageDir, cocoImg.FileName)
		if _, err := os.Stat(imgPath); os.IsNotExist(err) {
			continue
		}

		img, err := loadImage(imgPath)
		if err != nil {
			fmt.Printf("  Warning: failed to load %s: %v\n", cocoImg.FileName, err)
			continue
		}

		// Get ground truths
		anns := cocoDS.ImageAnnotations(cocoImg.ID)
		gts := make([]GroundTruth, 0, len(anns))
		for _, ann := range anns {
			if ann.IsCrowd == 1 {
				continue // skip crowd annotations
			}
			classID, ok := catToClass[ann.CategoryID]
			if !ok {
				continue // category not in COCO 80
			}
			gts = append(gts, GroundTruth{
				ClassID: classID,
				BBox:    COCOBBoxToBBox(ann.BBox),
			})
		}

		// Run prediction
		preds, err := engine.PredictDetections(img)
		if err != nil {
			fmt.Printf("  Warning: prediction failed for %s: %v\n", cocoImg.FileName, err)
			continue
		}

		imageEvals = append(imageEvals, ImageEval{
			Predictions:  preds,
			GroundTruths: gts,
		})
		processed++

		if processed%50 == 0 {
			fmt.Printf("  Processed %d images...\n", processed)
		}
	}

	if len(imageEvals) == 0 {
		return MAPResult{}, fmt.Errorf("no images evaluated")
	}

	result := ComputeMAP(imageEvals, 80)
	return result, nil
}

// CompareAccuracy runs accuracy evaluation for multiple engines and prints comparison.
func CompareAccuracy(engines []AccuracyEngine, cocoDS *COCODataset, imageDir string, maxImages int) {
	var results []MAPResult

	for _, eng := range engines {
		fmt.Printf("\nEvaluating %s...\n", eng.Name())
		result, err := EvaluateAccuracy(eng, cocoDS, imageDir, maxImages)
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}
		results = append(results, result)
		PrintMAPReport(fmt.Sprintf("%s Accuracy Results", eng.Name()), result)
	}

	if len(results) > 1 {
		PrintAccuracyComparison(engines, results)
	}
}

// PrintAccuracyComparison prints a comparison table of accuracy results.
func PrintAccuracyComparison(engines []AccuracyEngine, results []MAPResult) {
	sep := strings.Repeat("=", 70)
	fmt.Println(sep)
	fmt.Println("  Accuracy Comparison")
	fmt.Println(sep)
	fmt.Printf("%-20s | %10s | %10s | %10s | %8s | %8s\n",
		"Model", "mAP", "mAP@0.50", "mAP@0.75", "Images", "GT")
	fmt.Println(strings.Repeat("-", 70))

	for i, r := range results {
		name := engines[i].Name()
		fmt.Printf("%-20s | %10.4f | %10.4f | %10.4f | %8d | %8d\n",
			name, r.MAP, r.MAP50, r.MAP75, r.NumImages, r.NumGT)
	}
	fmt.Println(sep)
	fmt.Println()
}

// ============================================================
// Factory Functions
// ============================================================

// CreateAccuracyEngine creates an AccuracyEngine from a spec.
func CreateAccuracyEngine(spec AccuracySpec) (AccuracyEngine, error) {
	switch {
	case strings.HasPrefix(strings.ToLower(spec.Name), "yolo26"):
		switch spec.Task {
		case "det":
			return NewYOLO26DetAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "seg":
			return NewYOLO26SegAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "pose":
			return NewYOLO26PoseAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		default:
			return nil, fmt.Errorf("unsupported YOLO26 task: %s", spec.Task)
		}
	case strings.HasPrefix(strings.ToLower(spec.Name), "yolov11"):
		switch spec.Task {
		case "det":
			return NewYOLOv11DetAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "seg":
			return NewYOLOv11SegAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "pose":
			return NewYOLOv11PoseAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		default:
			return nil, fmt.Errorf("unsupported YOLOv11 task: %s", spec.Task)
		}
	case strings.HasPrefix(strings.ToLower(spec.Name), "rfdetr") || strings.HasPrefix(strings.ToLower(spec.Name), "rf-detr"):
		switch spec.Task {
		case "det":
			return NewRFDETRDetAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "seg":
			return NewRFDETRSegAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		default:
			return nil, fmt.Errorf("unsupported RF-DETR task: %s", spec.Task)
		}
	case strings.HasPrefix(strings.ToLower(spec.Name), "ltdetr"):
		return NewLTDETRDetAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
	case strings.HasPrefix(strings.ToLower(spec.Name), "dfine") || strings.HasPrefix(strings.ToLower(spec.Name), "d-fine"):
		switch spec.Task {
		case "det":
			return NewDFINEDetAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "seg":
			return NewDFINESegAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		default:
			return nil, fmt.Errorf("unsupported D-FINE task: %s", spec.Task)
		}
	case strings.HasPrefix(strings.ToLower(spec.Name), "edgecrafter") || strings.HasPrefix(strings.ToLower(spec.Name), "ec"):
		switch spec.Task {
		case "det":
			return NewECDetAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "seg":
			return NewECSegAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		case "pose":
			return NewECPoseAccEngine(spec.ModelPath, spec.LibPath, spec.UseCuda)
		default:
			return nil, fmt.Errorf("unsupported EdgeCrafter task: %s", spec.Task)
		}
	default:
		return nil, fmt.Errorf("unknown engine: %s", spec.Name)
	}
}

// DetAccuracySpecs returns accuracy evaluation specs for detection models.
func DetAccuracySpecs(useCuda bool) []AccuracySpec {
	libPath := ExampleORTLibraryPath()
	return []AccuracySpec{
		{Name: "yolo26", Task: "det", ModelPath: ExampleModelPath("yolo26", "yolo26m.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "rfdetr", Task: "det", ModelPath: ExampleModelPath("rf-detr", "rf-detr-medium.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "ltdetr", Task: "det", ModelPath: ExampleModelPath("ltdetr", "dinov3_vits16-ltdetr-coco.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "dfine", Task: "det", ModelPath: ExampleModelPath("dfine", "dfine_n_coco.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "edgecrafter", Task: "det", ModelPath: ExampleModelPath("ecdet", "ecdet_s.onnx"), LibPath: libPath, UseCuda: useCuda},
	}
}

// SegAccuracySpecs returns accuracy evaluation specs for segmentation models.
func SegAccuracySpecs(useCuda bool) []AccuracySpec {
	libPath := ExampleORTLibraryPath()
	return []AccuracySpec{
		{Name: "yolo26", Task: "seg", ModelPath: ExampleModelPath("yolo26", "yolo26m-seg.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "rfdetr", Task: "seg", ModelPath: ExampleModelPath("rfdetr", "rfdetr-medium-seg.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "dfine", Task: "seg", ModelPath: ExampleModelPath("dfine_seg", "dfine_seg_s_1x3x640x640.onnx"), LibPath: libPath, UseCuda: useCuda},
		{Name: "edgecrafter", Task: "seg", ModelPath: ExampleModelPath("edgecrafter", "edgecrafter-seg.onnx"), LibPath: libPath, UseCuda: useCuda},
	}
}

// YOLO26ScaleAccuracySpecs returns accuracy specs for YOLO26 scale comparison.
func YOLO26ScaleAccuracySpecs(task string, useCuda bool) []AccuracySpec {
	libPath := ExampleORTLibraryPath()
	var specs []AccuracySpec
	scales := []struct{ suffix, label string }{
		{"n", "yolo26n"}, {"s", "yolo26s"}, {"m", "yolo26m"}, {"l", "yolo26l"},
	}
	for _, sc := range scales {
		filename := sc.label
		if task != "det" {
			filename += "-" + task
		}
		filename += ".onnx"
		specs = append(specs, AccuracySpec{
			Name: sc.label, Task: task,
			ModelPath: ExampleModelPath("yolo26", filename),
			LibPath:   libPath, UseCuda: useCuda,
		})
	}
	return specs
}
