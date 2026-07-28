package examples

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ============================================================
// COCO Annotation Format
// ============================================================

// COCOAnnotation represents a single COCO annotation entry.
type COCOAnnotation struct {
	ID         int        `json:"id"`
	ImageID    int        `json:"image_id"`
	CategoryID int        `json:"category_id"`
	BBox       [4]float64 `json:"bbox"` // [x, y, width, height]
	Area       float64    `json:"area"`
	IsCrowd    int        `json:"iscrowd"`
}

// COCOImage represents a single COCO image entry.
type COCOImage struct {
	ID       int    `json:"id"`
	FileName string `json:"file_name"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
}

// COCOCategory represents a COCO category.
type COCOCategory struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

// COCODataset represents the full COCO annotation structure.
type COCODataset struct {
	Images      []COCOImage      `json:"images"`
	Annotations []COCOAnnotation `json:"annotations"`
	Categories  []COCOCategory   `json:"categories"`
}

// LoadCOCODataset loads a COCO annotation JSON file.
func LoadCOCODataset(jsonPath string) (*COCODataset, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", jsonPath, err)
	}

	var ds COCODataset
	if err := json.Unmarshal(data, &ds); err != nil {
		return nil, fmt.Errorf("failed to parse %s: %w", jsonPath, err)
	}
	return &ds, nil
}

// ImageAnnotations returns all annotations for a given image ID.
func (ds *COCODataset) ImageAnnotations(imageID int) []COCOAnnotation {
	var anns []COCOAnnotation
	for _, a := range ds.Annotations {
		if a.ImageID == imageID {
			anns = append(anns, a)
		}
	}
	return anns
}

// CategoryName returns the category name for a given category ID.
func (ds *COCODataset) CategoryName(catID int) string {
	for _, c := range ds.Categories {
		if c.ID == catID {
			return c.Name
		}
	}
	return fmt.Sprintf("unknown_%d", catID)
}

// COCOClassIDToCategoryID maps YOLO class IDs (0-79) to COCO category IDs.
// COCO category IDs are non-contiguous (1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,...).
// YOLO models use contiguous 0-based class IDs mapped to COCO categories in order.
var COCOClassIDToCategoryID = []int{
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
	21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
	41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
	59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
	80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
}

// CategoryIDToClassID builds a reverse mapping from COCO category ID to YOLO class ID.
func CategoryIDToClassID() map[int]int {
	m := make(map[int]int, len(COCOClassIDToCategoryID))
	for classID, catID := range COCOClassIDToCategoryID {
		m[catID] = classID
	}
	return m
}

// COCO80ClassNames are the 80 COCO class names in YOLO class ID order.
var COCO80ClassNames = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
	"mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
	"toothbrush",
}

// ============================================================
// IoU Computation
// ============================================================

// BBox represents [x1, y1, x2, y2] bounding box.
type BBox [4]float64

// COCOBBoxToBBox converts COCO [x, y, w, h] to [x1, y1, x2, y2].
func COCOBBoxToBBox(bbox [4]float64) BBox {
	return BBox{bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]}
}

// RectToBBox converts image.Rectangle to BBox.
func RectToBBox(r image.Rectangle) BBox {
	return BBox{float64(r.Min.X), float64(r.Min.Y), float64(r.Max.X), float64(r.Max.Y)}
}

// IoU computes Intersection over Union between two bounding boxes.
func IoU(a, b BBox) float64 {
	x1 := math.Max(a[0], b[0])
	y1 := math.Max(a[1], b[1])
	x2 := math.Min(a[2], b[2])
	y2 := math.Min(a[3], b[3])

	interArea := math.Max(0, x2-x1) * math.Max(0, y2-y1)
	if interArea == 0 {
		return 0
	}

	areaA := (a[2] - a[0]) * (a[3] - a[1])
	areaB := (b[2] - b[0]) * (b[3] - b[1])
	unionArea := areaA + areaB - interArea

	return interArea / unionArea
}

// ============================================================
// mAP Computation
// ============================================================

// Detection holds a single prediction for mAP evaluation.
type Detection struct {
	ClassID int
	Score   float64
	BBox    BBox
}

// GroundTruth holds a single ground truth annotation for mAP evaluation.
type GroundTruth struct {
	ClassID int
	BBox    BBox
	Used    bool // track if matched during evaluation
}

// ImageEval holds predictions and ground truths for a single image.
type ImageEval struct {
	Predictions  []Detection
	GroundTruths []GroundTruth
}

// MAPResult holds mAP evaluation results.
type MAPResult struct {
	MAP        float64         // mAP @ IoU=0.50:0.95
	MAP50      float64         // mAP @ IoU=0.50
	MAP75      float64         // mAP @ IoU=0.75
	PerClassAP map[int]float64 // AP per class @ IoU=0.50
	Precision  []float64       // precision curve
	Recall     []float64       // recall curve
	NumImages  int
	NumGT      int // total ground truths
	NumPred    int // total predictions
}

// ComputeMAP computes mAP at multiple IoU thresholds.
// Uses the COCO-style evaluation: average AP over IoU thresholds [0.50, 0.55, ..., 0.95].
func ComputeMAP(imageEvals []ImageEval, numClasses int) MAPResult {
	iouThresholds := make([]float64, 10)
	for i := range iouThresholds {
		iouThresholds[i] = 0.50 + float64(i)*0.05
	}

	aps := make([]float64, len(iouThresholds))
	perClassAP := make(map[int]float64)

	totalGT := 0
	totalPred := 0
	for _, ie := range imageEvals {
		totalGT += len(ie.GroundTruths)
		totalPred += len(ie.Predictions)
	}

	for tIdx, iouThresh := range iouThresholds {
		ap := computeAPAtIoU(imageEvals, numClasses, iouThresh)
		aps[tIdx] = ap

		// Also compute per-class AP at IoU=0.50
		if iouThresh == 0.50 {
			for c := 0; c < numClasses; c++ {
				classAP := computeClassAPAtIoU(imageEvals, c, iouThresh)
				if classAP > 0 {
					perClassAP[c] = classAP
				}
			}
		}
	}

	// mAP is mean of APs across all IoU thresholds
	mapVal := 0.0
	for _, ap := range aps {
		mapVal += ap
	}
	mapVal /= float64(len(aps))

	return MAPResult{
		MAP:        mapVal,
		MAP50:      aps[0],
		MAP75:      aps[5],
		PerClassAP: perClassAP,
		NumImages:  len(imageEvals),
		NumGT:      totalGT,
		NumPred:    totalPred,
	}
}

// computeAPAtIoU computes mean AP across all classes at a given IoU threshold.
func computeAPAtIoU(imageEvals []ImageEval, numClasses int, iouThresh float64) float64 {
	totalAP := 0.0
	validClasses := 0

	for c := 0; c < numClasses; c++ {
		ap := computeClassAPAtIoU(imageEvals, c, iouThresh)
		// Only count classes that have ground truths
		hasGT := false
		for _, ie := range imageEvals {
			for _, gt := range ie.GroundTruths {
				if gt.ClassID == c {
					hasGT = true
					break
				}
			}
			if hasGT {
				break
			}
		}
		if hasGT {
			totalAP += ap
			validClasses++
		}
	}

	if validClasses == 0 {
		return 0
	}
	return totalAP / float64(validClasses)
}

// computeClassAPAtIoU computes AP for a single class at a given IoU threshold.
func computeClassAPAtIoU(imageEvals []ImageEval, classID int, iouThresh float64) float64 {
	// Collect all predictions and ground truths for this class
	type predWithGT struct {
		score   float64
		tp      bool
		imageID int
	}

	var allPreds []predWithGT
	numGT := 0

	// Reset ground truth "used" flags
	for i := range imageEvals {
		for j := range imageEvals[i].GroundTruths {
			imageEvals[i].GroundTruths[j].Used = false
			if imageEvals[i].GroundTruths[j].ClassID == classID {
				numGT++
			}
		}
	}

	if numGT == 0 {
		return 0
	}

	// Collect predictions for this class
	for imgIdx, ie := range imageEvals {
		for _, pred := range ie.Predictions {
			if pred.ClassID != classID {
				continue
			}
			// Find best matching ground truth
			bestIoU := 0.0
			bestGTIdx := -1
			for gtIdx, gt := range ie.GroundTruths {
				if gt.ClassID != classID || gt.Used {
					continue
				}
				iou := IoU(pred.BBox, gt.BBox)
				if iou > bestIoU {
					bestIoU = iou
					bestGTIdx = gtIdx
				}
			}

			tp := false
			if bestIoU >= iouThresh && bestGTIdx >= 0 {
				tp = true
				imageEvals[imgIdx].GroundTruths[bestGTIdx].Used = true
			}

			allPreds = append(allPreds, predWithGT{
				score:   pred.Score,
				tp:      tp,
				imageID: imgIdx,
			})
		}
	}

	if len(allPreds) == 0 {
		return 0
	}

	// Sort by score descending
	sort.Slice(allPreds, func(i, j int) bool {
		return allPreds[i].score > allPreds[j].score
	})

	// Compute precision-recall curve
	tpCount := 0
	fpCount := 0
	var precisions, recalls []float64

	for _, p := range allPreds {
		if p.tp {
			tpCount++
		} else {
			fpCount++
		}
		precision := float64(tpCount) / float64(tpCount+fpCount)
		recall := float64(tpCount) / float64(numGT)
		precisions = append(precisions, precision)
		recalls = append(recalls, recall)
	}

	// Compute AP using 11-point interpolation (PASCAL VOC style)
	ap := 0.0
	for t := 0.0; t <= 1.0; t += 0.1 {
		maxP := 0.0
		for i, r := range recalls {
			if r >= t && precisions[i] > maxP {
				maxP = precisions[i]
			}
		}
		ap += maxP
	}
	ap /= 11.0

	return ap
}

// ============================================================
// Accuracy Report
// ============================================================

// PrintMAPReport prints a formatted mAP evaluation report.
func PrintMAPReport(title string, result MAPResult) {
	sep := strings.Repeat("=", 70)
	fmt.Println(sep)
	fmt.Printf("  %s\n", title)
	fmt.Println(sep)
	fmt.Printf("  Images:     %d\n", result.NumImages)
	fmt.Printf("  Ground Truths: %d\n", result.NumGT)
	fmt.Printf("  Predictions:   %d\n", result.NumPred)
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("  mAP @0.50:0.95:  %.4f\n", result.MAP)
	fmt.Printf("  mAP @0.50:       %.4f\n", result.MAP50)
	fmt.Printf("  mAP @0.75:       %.4f\n", result.MAP75)
	fmt.Println(strings.Repeat("-", 70))

	if len(result.PerClassAP) > 0 {
		fmt.Println("  Per-class AP @0.50:")
		// Sort by class ID
		classIDs := make([]int, 0, len(result.PerClassAP))
		for id := range result.PerClassAP {
			classIDs = append(classIDs, id)
		}
		sort.Ints(classIDs)
		for _, id := range classIDs {
			name := "unknown"
			if id < len(COCO80ClassNames) {
				name = COCO80ClassNames[id]
			}
			fmt.Printf("    %-20s (class %2d): %.4f\n", name, id, result.PerClassAP[id])
		}
	}
	fmt.Println(sep)
	fmt.Println()
}

// ============================================================
// Dataset Helpers
// ============================================================

// COCOImageDir returns the directory where COCO images are stored.
// Checks RAVEN_COCO_DIR env var first, then defaults to models/coco/val2017.
func COCOImageDir() string {
	if v := os.Getenv("RAVEN_COCO_DIR"); v != "" {
		return v
	}
	return filepath.Join(ExampleModelsRoot(), "coco", "val2017")
}

// COCOAnnotationPath returns the path to the COCO annotation file.
// Checks RAVEN_COCO_ANN env var first, then defaults to models/coco/annotations/instances_val2017.json.
func COCOAnnotationPath() string {
	if v := os.Getenv("RAVEN_COCO_ANN"); v != "" {
		return v
	}
	return filepath.Join(ExampleModelsRoot(), "coco", "annotations", "instances_val2017.json")
}

// COCOSubsetAnnotationPath returns the path to a COCO subset annotation file.
func COCOSubsetAnnotationPath(name string) string {
	return filepath.Join(ExampleModelsRoot(), "coco", "annotations", fmt.Sprintf("instances_%s.json", name))
}

// FindCOCOImages checks if COCO dataset images are available.
func FindCOCOImages(dir string) ([]string, error) {
	var files []string
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
			files = append(files, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(files)
	return files, nil
}
