package dfine

import (
	"image"
	"math"
	"testing"
)

// ============================================================
// boxXyxyToOrigScale — coordinate conversion tests
// ============================================================

func TestBoxXyxyToOrigScale_SameSize(t *testing.T) {
	// 640x640 model → 640x640 original → identity
	box := boxXyxyToOrigScale(10, 20, 100, 200, 640, 640, 640)
	expected := image.Rect(10, 20, 100, 200)
	if box != expected {
		t.Errorf("same-size: got %v, want %v", box, expected)
	}
}

func TestBoxXyxyToOrigScale_LargerImage(t *testing.T) {
	// 640x640 model → 1280x720 original → scale up
	box := boxXyxyToOrigScale(100, 50, 300, 200, 640, 1280, 720)
	// scaleX = 1280/640 = 2.0, scaleY = 720/640 = 1.125
	expected := image.Rect(200, 56, 600, 225)
	if box != expected {
		t.Errorf("larger image: got %v, want %v", box, expected)
	}
}

func TestBoxXyxyToOrigScale_SmallerImage(t *testing.T) {
	// 640x640 model → 320x240 original → scale down
	box := boxXyxyToOrigScale(200, 200, 400, 400, 640, 320, 240)
	// scaleX = 320/640 = 0.5, scaleY = 240/640 = 0.375
	expected := image.Rect(100, 75, 200, 150)
	if box != expected {
		t.Errorf("smaller image: got %v, want %v", box, expected)
	}
}

func TestBoxXyxyToOrigScale_ClipToBounds(t *testing.T) {
	// Box extends beyond original image → clipped
	box := boxXyxyToOrigScale(-50, -30, 700, 700, 640, 640, 480)
	expected := image.Rect(0, 0, 640, 480)
	if box != expected {
		t.Errorf("clip: got %v, want %v", box, expected)
	}
}

func TestBoxXyxyToOrigScale_ZeroBox(t *testing.T) {
	// Box at (0,0) → should stay at (0,0)
	box := boxXyxyToOrigScale(0, 0, 0, 0, 640, 1920, 1080)
	expected := image.Rect(0, 0, 0, 0)
	if box != expected {
		t.Errorf("zero box: got %v, want %v", box, expected)
	}
}

func TestBoxXyxyToOrigScale_NonSquareAspect(t *testing.T) {
	// Wide image: 640→1920 (3x), 640→1080 (1.6875x)
	box := boxXyxyToOrigScale(213, 160, 426, 320, 640, 1920, 1080)
	// x: 213*3=639, 426*3=1278
	// y: 160*1.6875=270, 320*1.6875=540
	expected := image.Rect(639, 270, 1278, 540)
	if box != expected {
		t.Errorf("non-square: got %v, want %v", box, expected)
	}
}

// ============================================================
// resizeMask — mask resizing and binarization tests
// ============================================================

func TestResizeMask_Basic(t *testing.T) {
	// Mask plane: 4x4, Original image: 8x8, Box: (2,2)-(6,6)
	// One pixel in mask at (2,2) maps to (4,4) in original
	maskData := make([]float32, 4*4) // 4x4 mask
	maskData[2*4+2] = 0.9            // pixel (2,2) in mask coords

	mask := resizeMask(maskData, 4, 4, image.Rect(2, 2, 6, 6), 8, 8, 0.5)
	if mask == nil {
		t.Fatal("resizeMask returned nil")
	}

	// Scale: maskW/origW = 4/8 = 0.5, maskH/origH = 4/8 = 0.5
	// (x=4, y=4) in original → (mx=2, my=2) in mask → value 0.9 > 0.5 → white
	if mask.GrayAt(4, 4).Y != 255 {
		t.Errorf("expected white pixel at (4,4), got %d", mask.GrayAt(4, 4).Y)
	}

	// (x=3, y=3) should be black (not in mask hotspot)
	if mask.GrayAt(3, 3).Y != 0 {
		t.Errorf("expected black pixel at (3,3), got %d", mask.GrayAt(3, 3).Y)
	}

	// Pixel outside box should be black
	if mask.GrayAt(0, 0).Y != 0 {
		t.Errorf("expected black pixel at (0,0) outside box, got %d", mask.GrayAt(0, 0).Y)
	}
}

func TestResizeMask_Threshold(t *testing.T) {
	maskData := make([]float32, 2*2) // 2x2 mask
	maskData[0] = 0.3                // below 0.5 threshold
	maskData[1] = 0.5                // at threshold (code uses >, not >=, so should be BLACK)
	maskData[2] = 0.7                // above threshold
	maskData[3] = 0.9                // above threshold

	mask := resizeMask(maskData, 2, 2, image.Rect(0, 0, 4, 4), 4, 4, 0.5)
	if mask == nil {
		t.Fatal("resizeMask returned nil")
	}

	// Scale: 2/4 = 0.5 → mask (mx,my) = orig(x,y)*0.5
	// (0,0): mx=0,my=0 → 0.3 < 0.5 → black ✓
	if mask.GrayAt(0, 0).Y != 0 {
		t.Errorf("(0,0): expected black (0.3<threshold), got %d", mask.GrayAt(0, 0).Y)
	}
	// (2,0): mx=1,my=0 → 0.5 exactly at threshold → black (code uses > not >=)
	if mask.GrayAt(2, 0).Y != 0 {
		t.Errorf("(2,0): expected black (0.5==threshold, code uses >), got %d", mask.GrayAt(2, 0).Y)
	}
	// (0,2): mx=0,my=1 → 0.7 > 0.5 → white ✓
	if mask.GrayAt(0, 2).Y != 255 {
		t.Errorf("(0,2): expected white (0.7>threshold), got %d", mask.GrayAt(0, 2).Y)
	}
	// (2,2): mx=1,my=1 → 0.9 > 0.5 → white ✓
	if mask.GrayAt(2, 2).Y != 255 {
		t.Errorf("(2,2): expected white (0.9>threshold), got %d", mask.GrayAt(2, 2).Y)
	}
}

func TestResizeMask_EmptyMask(t *testing.T) {
	// All-zero mask data
	maskData := make([]float32, 4*4)

	mask := resizeMask(maskData, 4, 4, image.Rect(0, 0, 8, 8), 8, 8, 0.5)
	if mask == nil {
		t.Fatal("resizeMask returned nil")
	}

	// All pixels should be black
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if mask.GrayAt(x, y).Y != 0 {
				t.Errorf("empty mask: expected black at (%d,%d), got %d", x, y, mask.GrayAt(x, y).Y)
				return
			}
		}
	}
}

func TestResizeMask_MaskLargerThanBox(t *testing.T) {
	// Mask 10x10, original 5x5, but box is only (1,1)-(3,3)
	maskData := make([]float32, 10*10)
	maskData[5*10+5] = 0.9 // pixel in mask coords

	mask := resizeMask(maskData, 10, 10, image.Rect(1, 1, 3, 3), 5, 5, 0.5)
	if mask == nil {
		t.Fatal("resizeMask returned nil")
	}

	// Scale: 10/5 = 2 → mx=5,my=5 when x=2.5,y=2.5
	// Since we iterate only within box (1,1)-(3,3), only (2,2) maps to mask (4,4)
	// Actually: mx=int(2*2)=4, my=int(2*2)=4 → mask[4*10+4]=0.0 → black
	// but (2,2) maps to my=int(2*2)=4, mx=int(2*2)=4 which is 0.0
	// Hmm, (2.5, 2.5) is the center of (2,2) to (3,3) box... let me recalculate
	// mx = int(2.0 * 2.0) = 4, my = int(2.0 * 2.0) = 4 → 0.0
	// mx = int(2.99 * 2.0) = 5, my = int(2.99 * 2.0) = 5 → 0.9 > 0.5 → white
	// So we need to find the right pixel...

	// Box is (1,1)-(3,3), scale=2.0
	// (2,1): mx=int(2*2)=4, my=int(1*2)=2 → mask[2*10+4]=0.0
	// (1,2): mx=int(1*2)=2, my=int(2*2)=4 → mask[4*10+2]=0.0

	// (2,2): mx=int(2*2)=4, my=int(2*2)=4 → mask[4*10+4]=0.0

	// Hmm, maskData[5*10+5] = 0.9, which is at mx=5, my=5
	// mx=5 needs x >= 2.5 and x < 3.0 in integer iteration
	// So only x=2, y=2 where x=2 gives mx=4... that's not 5
	// Wait, (x=2,y=2) → mx=int(2*2)=4, my=int(2*2)=4 → not our hotspot

	// I need to check: for x=2, y=2, mx=int(2*2.0)=4, my=int(2*2.0)=4 → mask[4*10+4]=0.0
	// The hotspot is at mask[5*10+5], so mx=5, my=5
	// mx=5 requires x >= 2.5, but x is integer: x=2 gives mx=4, x=3 would be outside box
	// So no pixel maps to our hotspot! This test won't find the white pixel.

	// Let me redefine: put the hotspot at a reachable location
	// mx=4, my=4 requires x=2, y=2 → let's use maskData[4*10+4] = 0.9
	maskData[4*10+4] = 0.9 // pixel (4,4) in mask coords → (2,2) in original

	mask2 := resizeMask(maskData, 10, 10, image.Rect(1, 1, 3, 3), 5, 5, 0.5)
	if mask2.GrayAt(2, 2).Y != 255 {
		t.Errorf("expected white at (2,2), got %d", mask2.GrayAt(2, 2).Y)
	}
	// Pixel outside box should be black
	if mask2.GrayAt(0, 0).Y != 0 {
		t.Errorf("expected black at (0,0) outside box, got %d", mask2.GrayAt(0, 0).Y)
	}
	// Pixel inside box but not at hotspot should be black
	if mask2.GrayAt(1, 1).Y != 0 {
		t.Errorf("expected black at (1,1), got %d", mask2.GrayAt(1, 1).Y)
	}
}

// ============================================================
// postprocessSeg — filtering and result building tests
// ============================================================

// newTestEngine creates a SegEngine with a minimal config for unit testing.
func newTestEngine() *SegEngine {
	return &SegEngine{
		config: Config{
			ConfThreshold: 0.5,
			MaxDetections: 100,
			MaskThreshold: 0.5,
			InputSize:     640,
		},
	}
}

func TestPostprocessSeg_AllBelowThreshold(t *testing.T) {
	e := newTestEngine()
	// 4 detections, all scores < 0.5
	boxes := []float32{
		10, 20, 100, 200, // 0
		50, 60, 150, 250, // 1
		5, 10, 80, 120, // 2
		200, 100, 300, 300, // 3
	}
	scores := []float32{0.1, 0.2, 0.3, 0.4}
	labels := []int64{1, 2, 3, 4}

	results := e.postprocessSeg(boxes, scores, labels, nil, 4, 0, 0, imageParams{origW: 640, origH: 640})
	if len(results) != 0 {
		t.Errorf("expected 0 results (all below threshold), got %d", len(results))
	}
}

func TestPostprocessSeg_MixedThreshold(t *testing.T) {
	e := newTestEngine()
	boxes := []float32{
		10, 20, 100, 200, // 0
		50, 60, 150, 250, // 1
		5, 10, 80, 120, // 2
	}
	scores := []float32{0.9, 0.3, 0.8}
	labels := []int64{1, 2, 3}

	results := e.postprocessSeg(boxes, scores, labels, nil, 3, 0, 0, imageParams{origW: 640, origH: 640})
	if len(results) != 2 {
		t.Fatalf("expected 2 results (indices 0 and 2 above threshold), got %d", len(results))
	}

	// Results should be sorted by score descending: index 0 (0.9) first, index 2 (0.8) second
	if results[0].Score != 0.9 || results[0].ClassID != 1 {
		t.Errorf("first result: expected score=0.9 class=1, got score=%.2f class=%d", results[0].Score, results[0].ClassID)
	}
	if results[1].Score != 0.8 || results[1].ClassID != 3 {
		t.Errorf("second result: expected score=0.8 class=3, got score=%.2f class=%d", results[1].Score, results[1].ClassID)
	}
}

func TestPostprocessSeg_MaxDetections(t *testing.T) {
	e := &SegEngine{
		config: Config{
			ConfThreshold: 0.0, // include all
			MaxDetections: 3,
			MaskThreshold: 0.5,
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

	results := e.postprocessSeg(boxes, scores, labels, nil, 10, 0, 0, imageParams{origW: 640, origH: 640})
	if len(results) != 3 {
		t.Fatalf("expected 3 results (MaxDetections=3), got %d", len(results))
	}

	// Should return top 3 by score: 1.0, 0.9, 0.8
	if results[0].Score != 1.0 || results[1].Score != 0.9 || results[2].Score != 0.8 {
		t.Errorf("expected top 3 scores [1.0, 0.9, 0.8], got [%.1f, %.1f, %.1f]",
			results[0].Score, results[1].Score, results[2].Score)
	}
}

func TestPostprocessSeg_WithMasks(t *testing.T) {
	e := newTestEngine()
	// Single detection above threshold
	boxes := []float32{10, 20, 100, 200}
	scores := []float32{0.9}
	labels := []int64{1}

	// 2x2 mask plane, 4 detections worth of mask data
	maskData := make([]float32, 4*2*2) // 4 detections * 2*2 mask
	maskData[0*4+1*2+1] = 0.9          // detection 0, mask pixel (1,1) → above threshold

	results := e.postprocessSeg(boxes, scores, labels, maskData, 1, 2, 2, imageParams{origW: 640, origH: 640})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	if results[0].Mask == nil {
		t.Fatal("expected non-nil mask")
	}

	// Score should still pass through correctly
	if results[0].Score != 0.9 {
		t.Errorf("expected score 0.9, got %.2f", results[0].Score)
	}

	// Mask should be a valid image with correct size
	maskBounds := results[0].Mask.Bounds()
	if maskBounds.Dx() != 640 || maskBounds.Dy() != 640 {
		t.Errorf("expected mask size 640x640, got %dx%d", maskBounds.Dx(), maskBounds.Dy())
	}
}

func TestPostprocessSeg_NoLabels(t *testing.T) {
	e := newTestEngine()
	boxes := []float32{10, 20, 100, 200}
	scores := []float32{0.9}
	// labels is nil

	results := e.postprocessSeg(boxes, scores, nil, nil, 1, 0, 0, imageParams{origW: 640, origH: 640})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	if results[0].ClassID != -1 {
		t.Errorf("expected ClassID -1 when labels are nil, got %d", results[0].ClassID)
	}
	if results[0].Score != 0.9 {
		t.Errorf("expected score 0.9, got %.2f", results[0].Score)
	}
}

func TestPostprocessSeg_EmptyMaskSlice(t *testing.T) {
	e := newTestEngine()
	boxes := []float32{10, 20, 100, 200}
	scores := []float32{0.9}
	labels := []int64{1}
	// masks is non-nil but empty (len=0)

	results := e.postprocessSeg(boxes, scores, labels, []float32{}, 1, 8, 8, imageParams{origW: 640, origH: 640})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	if results[0].Mask != nil {
		t.Errorf("expected nil mask for empty mask data, got non-nil")
	}
}

func TestPostprocessSeg_ScoreAtExactThreshold(t *testing.T) {
	e := newTestEngine()
	boxes := []float32{
		10, 20, 100, 200, // 0
		50, 60, 150, 250, // 1
	}
	scores := []float32{0.5, 0.49} // 0.5 at exact threshold, 0.49 below
	labels := []int64{1, 2}

	results := e.postprocessSeg(boxes, scores, labels, nil, 2, 0, 0, imageParams{origW: 640, origH: 640})
	if len(results) != 1 {
		t.Fatalf("expected 1 result (0.5 passes, 0.49 fails), got %d", len(results))
	}
	if results[0].Score != 0.5 {
		t.Errorf("expected score 0.5, got %.2f", results[0].Score)
	}
}

func TestPostprocessSeg_BoxValues(t *testing.T) {
	e := newTestEngine()
	boxes := []float32{100, 200, 300, 400}
	scores := []float32{0.9}
	labels := []int64{5}

	results := e.postprocessSeg(boxes, scores, labels, nil, 1, 0, 0, imageParams{origW: 1280, origH: 720})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// 640 model → 1280x720 original
	// scaleX = 1280/640 = 2.0, scaleY = 720/640 = 1.125
	expectedBox := image.Rect(200, 225, 600, 450)
	if results[0].Box != expectedBox {
		t.Errorf("expected box %v, got %v", expectedBox, results[0].Box)
	}
}

// ============================================================
// Integration-level helper consistency
// ============================================================

func TestPostprocessSeg_ConsistentWithPredict(t *testing.T) {
	// Verify that postprocessSeg logic matches Predict's expected behavior.
	// Box coords are always in model input space (640x640 square).
	// For model input 640x640 → original 480x360:
	//   scaleX = 480/640 = 0.75, scaleY = 360/640 = 0.5625
	//   (0,0,640,480) → (0,0,480,270)
	e := newTestEngine()

	boxes := []float32{0, 0, 640, 480}
	scores := []float32{0.95}
	labels := []int64{1}

	results := e.postprocessSeg(
		boxes[0:4], scores[0:1], labels[0:1], nil,
		1, 0, 0,
		imageParams{origW: 480, origH: 360},
	)

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	// Box at (0,0)-(640,480) in model space → scales to (0,0)-(480,270)
	// scaleX = 480/640 = 0.75, scaleY = 360/640 = 0.5625
	// x2 = int(640*0.75) = 480, y2 = int(480*0.5625) = 270
	expected := image.Rect(0, 0, 480, 270)
	if results[0].Box != expected {
		t.Errorf("expected box %v, got %v", expected, results[0].Box)
	}
}

func TestPostprocessSeg_StrideBoundaries(t *testing.T) {
	// Simulate batch of 2 images, each with 3 detections
	// This tests that postprocessSeg correctly handles slice boundaries
	e := newTestEngine()

	// Batch data: [img0_det0, img0_det1, img0_det2, img1_det0, img1_det1, img1_det2]
	boxes := []float32{
		// img0: 3 detections * 4 coords
		10, 20, 100, 200, // det0
		5, 10, 50, 100, // det1
		200, 150, 400, 350, // det2
		// img1: 3 detections * 4 coords
		30, 40, 120, 180, // det0
		15, 25, 60, 90, // det1
		250, 180, 350, 300, // det2
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
	img0Results := e.postprocessSeg(
		boxes[0:12], scores[0:3], labels[0:3], nil,
		3, 0, 0, imageParams{origW: 640, origH: 640},
	)
	if len(img0Results) != 2 {
		t.Fatalf("img0: expected 2 results (dets 0,1 above threshold), got %d", len(img0Results))
	}

	// Image 1: indices 3-5, same strides
	img1Results := e.postprocessSeg(
		boxes[12:24], scores[3:6], labels[3:6], nil,
		3, 0, 0, imageParams{origW: 640, origH: 640},
	)
	if len(img1Results) != 3 {
		t.Fatalf("img1: expected 3 results (all above 0.5 threshold), got %d", len(img1Results))
	}

	// img1 top result should be det5 (score=0.95)
	if img1Results[0].Score != 0.95 && img1Results[0].ClassID != 2 {
		t.Errorf("img1 top: expected score=0.95 class=2, got score=%.2f class=%d",
			img1Results[0].Score, img1Results[0].ClassID)
	}
}

// ============================================================
// Helper function tests (sigmoid)
// ============================================================

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
	}{
		{0, 0.5},
		{1, 0.7310586},
		{-1, 0.2689414},
		{10, 0.9999546},
		{-10, 0.0000454},
	}

	for _, tc := range tests {
		got := sigmoid(tc.input)
		if math.Abs(float64(got-tc.expected)) > 1e-5 {
			t.Errorf("sigmoid(%f) = %f, want %f", tc.input, got, tc.expected)
		}
	}
}
