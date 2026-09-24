package yolo26

import (
	"image"
	"testing"
)

// TestMapBoxToOrig_PadCompensation 是 yolo26-seg box 坐标反算 pad 补偿修复的回归测试。
// 此前 postprocess 把模型坐标 box 直接除以 scale 反算回原图，未扣除 letterbox 填充
// (padX/padY)，对非正方形输入整体偏移 ~padY/scale px。修复后反算逻辑集中在
// mapBoxToOrig（先减 pad 再除 scale 再钳制），此处直接钉住该公式。
func TestMapBoxToOrig_PadCompensation(t *testing.T) {
	tests := []struct {
		name           string
		params         imageParams
		x1, y1, x2, y2 float32
		want           image.Rectangle
	}{
		{
			// 非正方形输入 + 居中 letterbox：box 必须先减 pad 再除 scale
			name:   "letterbox pad compensated",
			params: imageParams{origW: 100, origH: 100, scale: 2.0, padX: 10, padY: 20},
			x1:     40, y1: 55, x2: 60, y2: 65,
			// ((40-10)/2,(55-20)/2,(60-10)/2,(65-20)/2)=(15,17,25,22)
			want: image.Rect(15, 17, 25, 22),
		},
		{
			// 无 pad（正方形输入）：scale=1 时模型坐标即原图坐标
			name:   "no pad identity",
			params: imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			x1:     40, y1: 55, x2: 60, y2: 65,
			want: image.Rect(40, 55, 60, 65),
		},
		{
			// 反算后超出原图边界的边被钳制到图像边界
			name:   "clamped at image bounds",
			params: imageParams{origW: 100, origH: 100, scale: 1.0, padX: 0, padY: 0},
			x1:     -20, y1: -10, x2: 60, y2: 70,
			want: image.Rect(0, 0, 60, 70),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mapBoxToOrig(tt.x1, tt.y1, tt.x2, tt.y2, tt.params); got != tt.want {
				t.Errorf("box = %v, want %v", got, tt.want)
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
			mask := e.decodeMask(image.Rect(0, 0, 5, 5), []float32{1}, protos, 1, h, w, tt.params)

			// 只有加回对应方向的 pad 才能命中明亮像素；去掉 pad 则采样错位到全 0 区域
			for x := 0; x < 5; x++ {
				if mask.GrayAt(x, 0).Y != 255 {
					t.Errorf("mask pixel (%d,0) not lit: pad addition in mask sampling broken", x)
				}
			}
		})
	}
}
