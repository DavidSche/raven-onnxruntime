package yolov11

import (
	"image"
	"testing"
)

// TestOBBPostprocess_PadCompensation 是 OBB letterbox pad 补偿修复的回归测试。
//
// 背景：engine_obb.go 此前把模型坐标角点直接除以 scale 反算回原图，未扣除
// letterbox 填充 (padX/padY)。对非正方形输入（如 ship.jpg 963×681 → 1024×1024，
// padY≈150）旋转框整体下移 padY/scale≈141px；修复后与 ultralytics .pt 输出一致。
//
// 该测试以合成模型输出驱动 postprocess（覆盖 parseCandidates → NMS → 坐标反算），
// 无需模型与 ONNX DLL，任何环境下都会执行；若回归（去掉 pad 减法）必然失败。
func TestOBBPostprocess_PadCompensation(t *testing.T) {
	tests := []struct {
		name        string
		params      imageParams
		cx, cy      float32
		w, h        float32
		wantCorners [4]image.Point
	}{
		{
			// 非正方形输入 + 居中 letterbox：角点必须先减 pad 再除 scale
			name:   "letterbox pad compensated",
			params: imageParams{origW: 100, origH: 100, scale: 2.0, padX: 10, padY: 20},
			cx:     50, cy: 60, w: 20, h: 10,
			// 模型角点 (40,55),(60,55),(60,65),(40,65)：
			// ((40-10)/2,(55-20)/2)=(15,17)；((60-10)/2,17.5→17)=(25,17)；
			// ((60-10)/2,(65-20)/2)=(25,22)；((40-10)/2,22.5→22)=(15,22)
			wantCorners: [4]image.Point{{X: 15, Y: 17}, {X: 25, Y: 17}, {X: 25, Y: 22}, {X: 15, Y: 22}},
		},
		{
			// 无 pad（正方形输入）：scale=1 时模型坐标即原图坐标
			name:   "no pad identity",
			params: imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			cx:     50, cy: 60, w: 20, h: 10,
			wantCorners: [4]image.Point{{X: 40, Y: 55}, {X: 60, Y: 55}, {X: 60, Y: 65}, {X: 40, Y: 65}},
		},
		{
			// 反算后超出原图边界的角点被钳制到图像边界
			name:   "clamped at image bounds",
			params: imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			cx:     20, cy: 30, w: 80, h: 80,
			// 模型角点 (-20,-10),(60,-10),(60,70),(-20,70) → 钳制后
			wantCorners: [4]image.Point{{X: 0, Y: 0}, {X: 60, Y: 0}, {X: 60, Y: 70}, {X: 0, Y: 70}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 合成 yolov11 输出 [1, channels, anchors]（channel-major 布局），
			// channels = 4 + nc + 1（末尾为 angle 通道）
			const nc = 1
			channels := 4 + nc + 1
			anchors := 4
			data := make([]float32, channels*anchors)

			anchor := 1
			data[0*anchors+anchor] = tt.cx
			data[1*anchors+anchor] = tt.cy
			data[2*anchors+anchor] = tt.w
			data[3*anchors+anchor] = tt.h
			data[(4+0)*anchors+anchor] = 0.9      // class 0 score
			data[(channels-1)*anchors+anchor] = 0 // angle
			data[(4+0)*anchors+0] = 0.1           // 低于阈值，应被过滤

			e := &OBBEngine{config: Config{NumClasses: nc, ConfThreshold: 0.5, IOUThreshold: 0.5}}
			results, err := e.postprocess(data, []int64{1, int64(channels), int64(anchors)}, tt.params)
			if err != nil {
				t.Fatalf("postprocess: %v", err)
			}
			if len(results) != 1 {
				t.Fatalf("got %d detections, want 1", len(results))
			}
			if got := results[0].Corners; got != tt.wantCorners {
				t.Errorf("corners = %v, want %v", got, tt.wantCorners)
			}
		})
	}
}
