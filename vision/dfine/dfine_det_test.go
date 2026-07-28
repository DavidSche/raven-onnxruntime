package dfine

import (
	"image"
	"testing"
)

// ============================================================
// postprocess — model-coordinate mode (boxesInOrigCoords=false)
// D-FINE-seg single-input model: boxes in model input space,
// need scaling to original image coords via boxXyxyToOrigScale.
// ============================================================

func newTestDetEngine() *DetEngine {
	return &DetEngine{
		config: Config{
			ConfThreshold: 0.5,
			MaxDetections: 100,
			InputSize:     640,
		},
	}
}

func TestDetPostprocess_AllBelowThreshold(t *testing.T) {
	e := newTestDetEngine()
	boxes := []float32{
		10, 20, 100, 200,
		50, 60, 150, 250,
	}
	scores := []float32{0.1, 0.3}
	labels := []int64{1, 2}

	results := e.postprocess(boxes, scores, labels, 2, imageParams{origW: 640, origH: 640}, false)
	if len(results) != 0 {
		t.Errorf("expected 0 results (all below threshold), got %d", len(results))
	}
}

func TestDetPostprocess_MixedThreshold(t *testing.T) {
	e := newTestDetEngine()
	boxes := []float32{
		10, 20, 100, 200, // 0
		50, 60, 150, 250, // 1
		5, 10, 80, 120, // 2
	}
	scores := []float32{0.9, 0.3, 0.8}
	labels := []int64{1, 2, 3}

	results := e.postprocess(boxes, scores, labels, 3, imageParams{origW: 640, origH: 640}, false)
	if len(results) != 2 {
		t.Fatalf("expected 2 results (idx 0 and 2 above threshold), got %d", len(results))
	}

	// Sorted by score descending: idx 0 (0.9) first, idx 2 (0.8) second
	if results[0].Score != 0.9 || results[0].ClassID != 1 {
		t.Errorf("first: expected score=0.9 class=1, got score=%.2f class=%d", results[0].Score, results[0].ClassID)
	}
	if results[1].Score != 0.8 || results[1].ClassID != 3 {
		t.Errorf("second: expected score=0.8 class=3, got score=%.2f class=%d", results[1].Score, results[1].ClassID)
	}
}

func TestDetPostprocess_ScoreAtExactThreshold(t *testing.T) {
	e := newTestDetEngine()
	boxes := []float32{10, 20, 100, 200, 50, 60, 150, 250}
	scores := []float32{0.5, 0.49} // 0.5 passes (< is strict), 0.49 fails
	labels := []int64{1, 2}

	results := e.postprocess(boxes, scores, labels, 2, imageParams{origW: 640, origH: 640}, false)
	if len(results) != 1 {
		t.Fatalf("expected 1 result (0.5 passes, 0.49 fails), got %d", len(results))
	}
	if results[0].Score != 0.5 {
		t.Errorf("expected score 0.5, got %.2f", results[0].Score)
	}
}

func TestDetPostprocess_MaxDetections(t *testing.T) {
	e := &DetEngine{
		config: Config{
			ConfThreshold: 0.0, // include all
			MaxDetections: 3,
			InputSize:     640,
		},
	}

	boxes := make([]float32, 10*4)
	for i := 0; i < 10; i++ {
		boxes[i*4+0] = float32(i * 10)
		boxes[i*4+1] = float32(i * 10)
		boxes[i*4+2] = float32(i*10 + 50)
		boxes[i*4+3] = float32(i*10 + 50)
	}
	scores := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	labels := []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	results := e.postprocess(boxes, scores, labels, 10, imageParams{origW: 640, origH: 640}, false)
	if len(results) != 3 {
		t.Fatalf("expected 3 results (MaxDetections=3), got %d", len(results))
	}

	// Top 3 by score: 1.0, 0.9, 0.8
	if results[0].Score != 1.0 || results[1].Score != 0.9 || results[2].Score != 0.8 {
		t.Errorf("expected top 3 scores [1.0, 0.9, 0.8], got [%.1f, %.1f, %.1f]",
			results[0].Score, results[1].Score, results[2].Score)
	}
}

func TestDetPostprocess_NoLabels(t *testing.T) {
	e := newTestDetEngine()
	boxes := []float32{10, 20, 100, 200}
	scores := []float32{0.9}

	results := e.postprocess(boxes, scores, nil, 1, imageParams{origW: 640, origH: 640}, false)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ClassID != -1 {
		t.Errorf("expected ClassID -1 when labels=nil, got %d", results[0].ClassID)
	}
}

func TestDetPostprocess_ModelCoordBoxValues(t *testing.T) {
	e := newTestDetEngine()
	// Model coord mode (boxesInOrigCoords=false): boxes in model input space (640x640)
	boxes := []float32{100, 200, 300, 400}
	scores := []float32{0.9}
	labels := []int64{5}

	results := e.postprocess(boxes, scores, labels, 1, imageParams{origW: 1280, origH: 720}, false)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// scaleX = 1280/640 = 2.0, scaleY = 720/640 = 1.125
	// (100,200,300,400) → (200,225,600,450)
	expected := image.Rect(200, 225, 600, 450)
	if results[0].Box != expected {
		t.Errorf("model coord: expected box %v, got %v", expected, results[0].Box)
	}
}

// ============================================================
// postprocess — original-coordinate mode (boxesInOrigCoords=true)
// D-FINE 2-input model with orig_target_sizes: boxes already in
// original image coordinates, just clip to image bounds.
// ============================================================

func TestDetPostprocess_OrigCoordMode(t *testing.T) {
	e := newTestDetEngine()
	// boxesInOrigCoords=true: boxes already in original image pixel coords
	boxes := []float32{100, 200, 500, 400}
	scores := []float32{0.9}
	labels := []int64{1}

	results := e.postprocess(boxes, scores, labels, 1, imageParams{origW: 640, origH: 480}, true)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// Box coordinates used directly, clipped to [0,origW] × [0,origH]
	expected := image.Rect(100, 200, 500, 400)
	if results[0].Box != expected {
		t.Errorf("orig coord: expected box %v, got %v", expected, results[0].Box)
	}
}

func TestDetPostprocess_OrigCoordClipNegative(t *testing.T) {
	e := newTestDetEngine()
	boxes := []float32{-50, -30, 300, 200}
	scores := []float32{0.9}
	labels := []int64{1}

	results := e.postprocess(boxes, scores, labels, 1, imageParams{origW: 640, origH: 480}, true)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// Negative values clipped to 0
	expected := image.Rect(0, 0, 300, 200)
	if results[0].Box != expected {
		t.Errorf("clip negative: expected box %v, got %v", expected, results[0].Box)
	}
}

func TestDetPostprocess_OrigCoordClipOvershoot(t *testing.T) {
	e := newTestDetEngine()
	boxes := []float32{100, 50, 900, 800}
	scores := []float32{0.9}
	labels := []int64{1}

	results := e.postprocess(boxes, scores, labels, 1, imageParams{origW: 640, origH: 480}, true)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// Overshoot clipped to image bounds
	expected := image.Rect(100, 50, 640, 480)
	if results[0].Box != expected {
		t.Errorf("clip overshoot: expected box %v, got %v", expected, results[0].Box)
	}
}

func TestDetPostprocess_OrigCoordFullFrame(t *testing.T) {
	e := newTestDetEngine()
	boxes := []float32{-100, -100, 2000, 2000}
	scores := []float32{0.95}
	labels := []int64{0}

	results := e.postprocess(boxes, scores, labels, 1, imageParams{origW: 1920, origH: 1080}, true)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// Clipped to full image bounds
	expected := image.Rect(0, 0, 1920, 1080)
	if results[0].Box != expected {
		t.Errorf("full frame: expected box %v, got %v", expected, results[0].Box)
	}
}

// ============================================================
// postprocess — both modes compared for the same box
// ============================================================

func TestDetPostprocess_ModelCoordVsOrigCoord(t *testing.T) {
	e := newTestDetEngine()

	// Same box (100,200,300,400), image 640x480
	boxes := []float32{100, 200, 300, 400}
	scores := []float32{0.9}
	labels := []int64{1}

	// Model coord mode: scale from 640→640×480
	// scaleX = 640/640 = 1.0, scaleY = 480/640 = 0.75
	// Box: (100, 150, 300, 300)
	modelResults := e.postprocess(boxes, scores, labels, 1, imageParams{origW: 640, origH: 480}, false)
	if len(modelResults) != 1 {
		t.Fatalf("model coord: expected 1 result, got %d", len(modelResults))
	}
	modelBox := modelResults[0].Box

	// Orig coord mode: use coordinates directly (just clip)
	origResults := e.postprocess(boxes, scores, labels, 1, imageParams{origW: 640, origH: 480}, true)
	if len(origResults) != 1 {
		t.Fatalf("orig coord: expected 1 result, got %d", len(origResults))
	}
	origBox := origResults[0].Box

	// For a 640×640 model → 640×480 image, boxes should differ (y-coord scaled)
	if modelBox == origBox {
		t.Errorf("model coord box %v should differ from orig coord box %v for non-square image", modelBox, origBox)
	}

	// Verify the actual values
	// Model coord: (100,200,300,400) → scale y by 480/640=0.75 → (100,150,300,300)
	expectedModelBox := image.Rect(100, 150, 300, 300)
	if modelBox != expectedModelBox {
		t.Errorf("model coord: expected %v, got %v", expectedModelBox, modelBox)
	}

	// Orig coord: (100,200,300,400) → clip to [0,640]×[0,480] → (100,200,300,400)
	expectedOrigBox := image.Rect(100, 200, 300, 400)
	if origBox != expectedOrigBox {
		t.Errorf("orig coord: expected %v, got %v", expectedOrigBox, origBox)
	}
}

// ============================================================
// Stride boundary tests (simulating PredictBatch slicing)
// ============================================================

func TestDetPostprocess_StrideBoundaries(t *testing.T) {
	e := newTestDetEngine()

	// Simulate batch of 2 images, each with 3 detections
	boxes := []float32{
		// img0: 3 det * 4 coords
		10, 20, 100, 200, 5, 10, 50, 100, 200, 150, 400, 350,
		// img1: 3 det * 4 coords
		30, 40, 120, 180, 15, 25, 60, 90, 250, 180, 350, 300,
	}
	scores := []float32{
		// img0
		0.9, 0.8, 0.3,
		// img1
		0.85, 0.75, 0.95,
	}
	labels := []int64{
		// img0
		1, 2, 3,
		// img1
		1, 3, 2,
	}

	// Image 0: indices 0-2, box stride=12, score stride=3, label stride=3
	img0Results := e.postprocess(
		boxes[0:12], scores[0:3], labels[0:3],
		3, imageParams{origW: 640, origH: 640}, false,
	)
	if len(img0Results) != 2 {
		t.Fatalf("img0: expected 2 results (dets 0,1 above 0.5), got %d", len(img0Results))
	}

	// Image 1: indices 3-5
	img1Results := e.postprocess(
		boxes[12:24], scores[3:6], labels[3:6],
		3, imageParams{origW: 640, origH: 640}, false,
	)
	if len(img1Results) != 3 {
		t.Fatalf("img1: expected 3 results (all above 0.5), got %d", len(img1Results))
	}

	// img1 top: det5 (score=0.95, class=2)
	if img1Results[0].Score != 0.95 || img1Results[0].ClassID != 2 {
		t.Errorf("img1 top: expected score=0.95 class=2, got score=%.2f class=%d",
			img1Results[0].Score, img1Results[0].ClassID)
	}
}
