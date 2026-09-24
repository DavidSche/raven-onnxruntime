package yolo26

import (
	"image"
	"testing"
)

// TestOBBDecodeOutput_PadCompensation 是 OBB letterbox pad 补偿修复的回归测试。
//
// 背景：engine_obb.go 此前把模型坐标角点直接除以 scale 反算回原图，未扣除
// letterbox 填充 (padX/padY)。对非正方形输入（如 ship.jpg 963×681 → 1024×1024，
// padY≈150）旋转框整体下移 padY/scale≈141px；修复后与 ultralytics .pt 输出一致。
//
// 以合成模型输出 [1, 300, 7] 驱动 decodeOutput，覆盖完整解码路径：阈值过滤 →
// getRotatedCorners → mapCornersToOrig（含 pad 补偿）→ 结果组装。无需模型与
// ONNX DLL；若回归（去掉 pad 减法或调用路径被破坏）必然失败。
func TestOBBDecodeOutput_PadCompensation(t *testing.T) {
	tests := []struct {
		name        string
		box         [7]float32 // [cx, cy, w, h, score, class_id, angle]
		params      imageParams
		wantCorners [4]image.Point
	}{
		{
			// 非正方形输入 + 居中 letterbox：角点必须先减 pad 再除 scale（yolo26 用 math.Round）
			name:   "letterbox pad compensated",
			box:    [7]float32{50, 60, 20, 10, 0.9, 1, 0}, // 模型角点 (40,55),(60,55),(60,65),(40,65)
			params: imageParams{origW: 100, origH: 100, scale: 2.0, padX: 10, padY: 20},
			// (40-10)/2=15、(55-20)/2=17.5→18、(60-10)/2=25、(65-20)/2=22.5→23
			wantCorners: [4]image.Point{{X: 15, Y: 18}, {X: 25, Y: 18}, {X: 25, Y: 23}, {X: 15, Y: 23}},
		},
		{
			// 无 pad（正方形输入）：scale=1 时模型坐标即原图坐标
			name:        "no pad identity",
			box:         [7]float32{50, 60, 20, 10, 0.9, 1, 0},
			params:      imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			wantCorners: [4]image.Point{{X: 40, Y: 55}, {X: 60, Y: 55}, {X: 60, Y: 65}, {X: 40, Y: 65}},
		},
		{
			// 反算后超出原图边界的角点被钳制到图像边界
			name:        "clamped at image bounds",
			box:         [7]float32{20, 30, 80, 80, 0.9, 1, 0}, // 模型角点 (-20,-10),(60,-10),(60,70),(-20,70)
			params:      imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			wantCorners: [4]image.Point{{X: 0, Y: 0}, {X: 60, Y: 0}, {X: 60, Y: 70}, {X: 0, Y: 70}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const numBoxes = 300
			const numAttrs = 7
			data := make([]float32, numBoxes*numAttrs)
			// 仅第 5 个框写入有效检测，其余 299 个 score=0 应被阈值过滤
			i := 5
			for j, v := range tt.box {
				data[i*numAttrs+j] = v
			}

			e := &OBBEngine{config: Config{ConfThreshold: 0.5}}
			results := e.decodeOutput(data, []int64{1, numBoxes, numAttrs}, tt.params)
			if len(results) != 1 {
				t.Fatalf("got %d detections, want 1", len(results))
			}
			if got := results[0].Corners; got != tt.wantCorners {
				t.Errorf("corners = %v, want %v", got, tt.wantCorners)
			}
			if results[0].ClassID != 1 || results[0].Score != 0.9 {
				t.Errorf("class/score = %d/%.2f, want 1/0.90", results[0].ClassID, results[0].Score)
			}
		})
	}
}

// TestMapCornersToOrig_PadCompensation 直接钉住坐标反算公式本身：先减 pad 再除 scale。
// 与 decodeOutput 级测试互补——即使未来调用路径变化，公式回归也会在此被捕获。
func TestMapCornersToOrig_PadCompensation(t *testing.T) {
	corners := getRotatedCorners(50, 60, 20, 10, 0) // (40,55),(60,55),(60,65),(40,65)
	params := imageParams{origW: 100, origH: 100, scale: 2.0, padX: 10, padY: 20}
	want := [4]image.Point{{X: 15, Y: 18}, {X: 25, Y: 18}, {X: 25, Y: 23}, {X: 15, Y: 23}}

	if got := mapCornersToOrig(corners, params); got != want {
		t.Errorf("corners = %v, want %v", got, want)
	}
}
