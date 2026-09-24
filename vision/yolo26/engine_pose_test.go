package yolo26

import (
	"image"
	"testing"
)

// TestPosePostprocess_PadCompensation 是 yolo26-pose（row-major 布局 [1,N,attrs]）的
// box/关键点坐标反算 pad 补偿回归测试。此前 postprocess 与 decodeKeyPoints 均未扣除
// letterbox 填充 (padX/padY)，对非正方形输入整体偏移 ~padY/scale px。
func TestPosePostprocess_PadCompensation(t *testing.T) {
	tests := []struct {
		name    string
		params  imageParams
		wantBox image.Rectangle
		wantKpt image.Point
	}{
		{
			// 非正方形输入 + 居中 letterbox：box 与关键点都必须先减 pad 再除 scale
			name:    "letterbox pad compensated",
			params:  imageParams{origW: 100, origH: 100, scale: 2.0, padX: 10, padY: 20},
			wantBox: image.Rect(15, 17, 25, 22),
			wantKpt: image.Point{X: 10, Y: 10}, // 关键点 (30,40) → ((30-10)/2,(40-20)/2)
		},
		{
			// 无 pad（正方形输入）：scale=1 时模型坐标即原图坐标
			name:    "no pad identity",
			params:  imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			wantBox: image.Rect(40, 55, 60, 65),
			wantKpt: image.Point{X: 30, Y: 40},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 合成 yolo26-pose row-major 输出 [1, 2, 9]：attributes = 6 + 1 个关键点*3
			const numObjects, attributes = 2, 9
			data := make([]float32, numObjects*attributes)

			i := 1
			data[i*attributes+0] = 40  // x1
			data[i*attributes+1] = 55  // y1
			data[i*attributes+2] = 60  // x2
			data[i*attributes+3] = 65  // y2
			data[i*attributes+4] = 0.9 // score
			data[i*attributes+5] = 0   // class
			data[i*attributes+6] = 30  // keypoint x
			data[i*attributes+7] = 40  // keypoint y
			data[i*attributes+8] = 0.9 // keypoint conf
			// object 0 score=0，应被阈值过滤

			e := &PoseEngine{config: Config{NumKeyPoints: 1, ConfThreshold: 0.5}}
			results, err := e.postprocess(data, []int64{1, numObjects, attributes}, tt.params)
			if err != nil {
				t.Fatalf("postprocess: %v", err)
			}
			if len(results) != 1 {
				t.Fatalf("got %d detections, want 1", len(results))
			}
			if got := results[0].Box; got != tt.wantBox {
				t.Errorf("Box = %v, want %v", got, tt.wantBox)
			}
			gotKpt := image.Point{X: results[0].KeyPoints[0].X, Y: results[0].KeyPoints[0].Y}
			if gotKpt != tt.wantKpt {
				t.Errorf("keypoint = %v, want %v", gotKpt, tt.wantKpt)
			}
		})
	}
}

// TestPosePostprocessChannelFirst_PadCompensation 覆盖 yolo26-pose 的 channel-first
// 分支（[1, channels, anchors]，channels>50 且 anchors>1000 时启用）的 box/关键点
// pad 补偿：postprocessChannelFirst（box）→ NMS → decodeKeyPoints（关键点）。
func TestPosePostprocessChannelFirst_PadCompensation(t *testing.T) {
	const nc, nkpt = 1, 17
	channels := 4 + nc + nkpt*3 // 56 > 50 → channel-first 分支
	const anchors = 2000        // > 1000
	data := make([]float32, channels*anchors)

	anchor := 500
	data[0*anchors+anchor] = 50         // cx
	data[1*anchors+anchor] = 60         // cy
	data[2*anchors+anchor] = 20         // w
	data[3*anchors+anchor] = 10         // h
	data[(4+0)*anchors+anchor] = 0.9    // class 0 score
	data[(4+nc+0)*anchors+anchor] = 30  // keypoint 0 x
	data[(4+nc+1)*anchors+anchor] = 40  // keypoint 0 y
	data[(4+nc+2)*anchors+anchor] = 0.9 // keypoint 0 conf

	e := &PoseEngine{config: Config{NumClasses: nc, NumKeyPoints: nkpt, ConfThreshold: 0.5, IOUThreshold: 0.5}}
	params := imageParams{origW: 100, origH: 100, scale: 2.0, padX: 10, padY: 20}
	results, err := e.postprocess(data, []int64{1, int64(channels), int64(anchors)}, params)
	if err != nil {
		t.Fatalf("postprocess: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d detections, want 1", len(results))
	}
	wantBox := image.Rect(15, 17, 25, 22)
	if got := results[0].Box; got != wantBox {
		t.Errorf("Box = %v, want %v", got, wantBox)
	}
	wantKpt := image.Point{X: 10, Y: 10}
	gotKpt := image.Point{X: results[0].KeyPoints[0].X, Y: results[0].KeyPoints[0].Y}
	if gotKpt != wantKpt {
		t.Errorf("keypoint = %v, want %v", gotKpt, wantKpt)
	}
}
