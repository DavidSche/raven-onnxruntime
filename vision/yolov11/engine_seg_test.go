package yolov11

import (
	"image"
	"testing"
)

// TestSegParseCandidates_PadCompensation 是 yolov11-seg box 坐标反算 pad 补偿修复的
// 回归测试。此前 parseCandidates 把模型坐标 box 直接除以 scale 反算回原图，未扣除
// letterbox 填充 (padX/padY)，对非正方形输入整体偏移 ~padY/scale px。
// 以合成 channel-major 输出驱动 parseCandidates，无需模型与 ONNX DLL。
func TestSegParseCandidates_PadCompensation(t *testing.T) {
	tests := []struct {
		name    string
		params  imageParams
		cx, cy  float32
		w, h    float32
		wantBox image.Rectangle
	}{
		{
			// 非正方形输入 + 居中 letterbox：box 必须先减 pad 再除 scale
			name:   "letterbox pad compensated",
			params: imageParams{origW: 100, origH: 100, scale: 2.0, padX: 10, padY: 20},
			cx:     50, cy: 60, w: 20, h: 10,
			// x1=40,y1=55,x2=60,y2=65 → ((40-10)/2,(55-20)/2,(60-10)/2,(65-20)/2)=(15,17,25,22)
			wantBox: image.Rect(15, 17, 25, 22),
		},
		{
			// 无 pad（正方形输入）：scale=1 时模型坐标即原图坐标
			name:   "no pad identity",
			params: imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			cx:     50, cy: 60, w: 20, h: 10,
			wantBox: image.Rect(40, 55, 60, 65),
		},
		{
			// 反算后超出原图边界的边被钳制到图像边界
			name:   "clamped at image bounds",
			params: imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			cx:     20, cy: 30, w: 80, h: 80,
			// x1=-20,y1=-10,x2=60,y2=70 → (0,0,60,70)
			wantBox: image.Rect(0, 0, 60, 70),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 合成 yolov11-seg 输出 [1, channels, anchors]（channel-major），
			// channels = 4 + nc + nmask
			const nc, nmask = 1, 1
			channels := 4 + nc + nmask
			anchors := 4
			data := make([]float32, channels*anchors)

			anchor := 1
			data[0*anchors+anchor] = tt.cx
			data[1*anchors+anchor] = tt.cy
			data[2*anchors+anchor] = tt.w
			data[3*anchors+anchor] = tt.h
			data[(4+0)*anchors+anchor] = 0.9    // class 0 score
			data[(4+nc+0)*anchors+anchor] = 1.0 // mask coeff 0
			data[(4+0)*anchors+0] = 0.1         // 低于阈值，应被过滤

			e := &SegEngine{config: Config{NumClasses: nc, NumMaskCoeffs: nmask, ConfThreshold: 0.5}}
			cands := e.parseCandidates(data, channels, anchors, tt.params)
			if len(cands) != 1 {
				t.Fatalf("got %d candidates, want 1", len(cands))
			}
			if got := cands[0].origBox; got != tt.wantBox {
				t.Errorf("origBox = %v, want %v", got, tt.wantBox)
			}
		})
	}
}

// TestSegDecodeMask_PadCompensation 验证 mask 采样的正向映射必须加回 letterbox pad：
// 原图像素 (x,y) 先映射回输入空间 x*scale+padX，再按 maskStride 定位 mask 像素。
// 若缺失 pad 加法，mask 会整体错位 padX/maskStride 个像素。
func TestSegDecodeMask_PadCompensation(t *testing.T) {
	const w, h = 160, 160

	tests := []struct {
		name    string
		params  imageParams
		protoY  int // 明亮 proto 行（my）
		protoX0 int // 明亮 proto 起始列（mx）
	}{
		{
			// padX 加法：原图 x=0..4 应命中 proto mx=10..14
			name:    "padX addition",
			params:  imageParams{origW: 160, origH: 160, scale: 1.0, padX: 10, padY: 0},
			protoY:  0,
			protoX0: 10,
		},
		{
			// padY 加法：原图 y=0 应命中 proto my=10
			name:    "padY addition",
			params:  imageParams{origW: 160, origH: 160, scale: 1.0, padX: 0, padY: 10},
			protoY:  10,
			protoX0: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 单通道 proto，仅在目标位置 5 个像素为 1（maskStride = InputSize/w = 1）
			protos := make([]float32, 1*w*h)
			for mx := tt.protoX0; mx < tt.protoX0+5; mx++ {
				protos[tt.protoY*w+mx] = 1
			}

			e := &SegEngine{config: Config{InputSize: 160, MaskThreshold: 0.5}}
			mask := e.decodeMask(candidate{origBox: image.Rect(0, 0, 5, 5), maskCoeffs: []float32{1}}, protos, 1, h, w, tt.params)

			// 只有加回对应方向的 pad 才能命中明亮像素；去掉 pad 则采样错位到全 0 区域
			for x := 0; x < 5; x++ {
				if mask.GrayAt(x, 0).Y != 255 {
					t.Errorf("mask pixel (%d,0) not lit: pad addition in mask sampling broken", x)
				}
			}
		})
	}
}
