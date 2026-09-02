package vision

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// TestCalcLetterBox_RoundSemantics 钉住 CalcLetterBox 的内容尺寸与 ultralytics
// letterbox 的 round(w*r) 语义逐位一致（Python 银行家舍入，见 pyRound）。
//
// 背景：早期实现用 int() 截断，float 乘积略低于整数时会少 1px——
// 如 963×(1280/963)=1279.999… 截断为 1279 而 round=1280。标为
// "floor!=round" 的用例正是差异点：任何改回截断或恢复 stride 对齐的实现都会失败。
// 期望值由 ultralytics float64 计算（min(t/h,t/w) + round）核对一致。
func TestCalcLetterBox_RoundSemantics(t *testing.T) {
	tests := []struct {
		name                 string
		origW, origH, target int
		wantW, wantH         int
		wantPadX, wantPadY   int
	}{
		// yolo26n OBB @1280（floor 曾得 1279×905，round 得 1280×905）
		{"ship@1280 floor!=round W", 963, 681, 1280, 1280, 905, 0, 187},
		{"ship@1024 (yolo11m OBB)", 963, 681, 1024, 1024, 724, 0, 150},
		{"palace@640 (det/seg/pose)", 1920, 1080, 640, 640, 360, 0, 140},
		// 纵向条带：scale=640/1080，W=round(420×0.5926)=249（floor 曾得 248）
		{"palace_tall@640 floor!=round W", 420, 1080, 640, 249, 640, 195, 0},
		{"palace_square@640 no pad", 1080, 1080, 640, 640, 640, 0, 0},
		// 超宽裁剪：W=1280（floor 曾得 1279）、H=round(218×1.3292)=290（floor 曾得 289）
		{"ship_wide@1280 floor!=round", 963, 218, 1280, 1280, 290, 0, 495},
		{"ship_tall@1280", 231, 681, 1280, 434, 1280, 423, 0},
		// 纵向条带 @1024：H=round(681×1.5037)=1024（floor 曾得 1023）
		{"ship_tall@1024 floor!=round H", 231, 681, 1024, 347, 1024, 338, 0},
		// 超宽裁剪 @640：W=round(963×0.6646)=640（floor 曾得 639）
		{"palace_wide@640 floor!=round W", 963, 310, 640, 640, 206, 0, 217},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := CalcLetterBox(tc.origW, tc.origH, tc.target, DefaultStride, LetterBoxCenter)
			if err != nil {
				t.Fatalf("CalcLetterBox: %v", err)
			}
			if p.NewW != tc.wantW || p.NewH != tc.wantH {
				t.Errorf("content = %dx%d, want %dx%d (ultralytics round)", p.NewW, p.NewH, tc.wantW, tc.wantH)
			}
			if p.PadX != tc.wantPadX || p.PadY != tc.wantPadY {
				t.Errorf("pad = (%d,%d), want (%d,%d)", p.PadX, p.PadY, tc.wantPadX, tc.wantPadY)
			}
		})
	}
}

// TestPyRound_BankerSemantics 直接钉住 pyRound 的银行家舍入行为：恰好 .5 时向
// 偶数取整（round(2.5)=2、round(3.5)=4），防止未来重构静默退化为四舍五入。
// 注意：math.Round 是 half-away-from-zero（2.5→3、3.5→4），与本语义在奇数侧不同。
func TestPyRound_BankerSemantics(t *testing.T) {
	tests := []struct {
		x    float64
		want int
	}{
		{0.0, 0},
		{0.49, 0},
		{0.51, 1},
		{0.5, 0},          // 平分→偶
		{1.5, 2},          // 平分→偶
		{2.5, 2},          // 平分→偶
		{3.5, 4},          // 平分→偶
		{4.5, 4},          // 平分→偶
		{1279.9996, 1280}, // 接近整数向上
		{1280.0004, 1280}, // 接近整数向下
	}
	for _, tc := range tests {
		if got := pyRound(tc.x); got != tc.want {
			t.Errorf("pyRound(%v) = %d, want %d (banker's rounding)", tc.x, got, tc.want)
		}
	}
}

func TestFillCHWFromImageRGBA(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 0, A: 255})

	data := make([]float32, 3)
	if err := FillCHWFromImage(data, img, 1, 1, 1, 1, nil, nil); err != nil {
		t.Fatalf("FillCHWFromImage failed: %v", err)
	}

	want := []float32{1, 128.0 / 255.0, 0}
	for i, got := range data {
		if diff := math.Abs(float64(got - want[i])); diff > 1e-6 {
			t.Fatalf("channel %d: got %f want %f", i, got, want[i])
		}
	}
}

func TestFillCHWFromImageNormalized(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})

	data := make([]float32, 3)
	means := [3]float32{0.5, 0.0, 0.0}
	stds := [3]float32{0.5, 1.0, 1.0}
	if err := FillCHWFromImage(data, img, 1, 1, 1, 1, &means, &stds); err != nil {
		t.Fatalf("FillCHWFromImage failed: %v", err)
	}

	want := []float32{1, 0, 0}
	for i, got := range data {
		if diff := math.Abs(float64(got - want[i])); diff > 1e-6 {
			t.Fatalf("channel %d: got %f want %f", i, got, want[i])
		}
	}
}

// TestResizeTorchBilinear_Golden 钉住 ResizeTorchBilinear 与 torchvision
// ``F.resize(antialias=False)``（即 ATen upsample_bilinear2d, align_corners=False）
// 的逐像素一致性。
//
// 背景：普通双线性（PIL / draw.BiLinear）的源坐标约定与 torchvision 不同，
// 大倍数降采样后归一化像素可差 ~0.1+，导致临界置信度（如 0.64 vs 0.45）
// 跨过阈值。rf-detr predict 用 torchvision 管线，故引擎必须复刻其内核。
//
// 期望值：5×4 确定性图 → 3×2，由 Python torch 2.12 + torchvision 0.27
// ``F.resize(antialias=False)`` 计算后 round(*255) 得到（与 Go 端 uint8 舍入一致）。
func TestResizeTorchBilinear_Golden(t *testing.T) {
	// 5x4 确定性 RGB 图：R=(x*50+y*30)%256, G=(x*80+y*40)%256, B=(x*20+y*60)%256
	const W, H = 5, 4
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8((x*50 + y*30) % 256),
				G: uint8((x*80 + y*40) % 256),
				B: uint8((x*20 + y*60) % 256),
				A: 255,
			})
		}
	}

	out := ResizeTorchBilinear(img, 2, 3) // (width, height) = (2, 3)

	if got := out.Bounds(); got.Dx() != 2 || got.Dy() != 3 {
		t.Fatalf("bounds = %v, want 2x3", got)
	}

	// uint8 期望值（H,W,C）来自 Python torchvision F.resize(antialias=False)
	// 的 float32 输出，再按引擎的 uint8 约定 floor(v*255+0.5)（四舍五入取半向上）
	// 转换。注意 (0,1).R 的 float 值恰为 82.5000026——若用 torch.round
	//（银行家舍入）会得 82，引擎取半向上得 83；float 值本身与 torch 完全一致。
	want := [][][3]uint8{
		{{43, 67, 25}, {168, 171, 75}},
		{{83, 120, 105}, {176, 64, 155}},
		{{123, 173, 185}, {184, 117, 182}},
	}
	for y := 0; y < 3; y++ {
		for x := 0; x < 2; x++ {
			r, g, b, a := out.At(x, y).RGBA()
			if uint8(r>>8) != want[y][x][0] || uint8(g>>8) != want[y][x][1] || uint8(b>>8) != want[y][x][2] {
				t.Errorf("pixel (%d,%d) = (%d,%d,%d), want %v", x, y, r>>8, g>>8, b>>8, want[y][x])
			}
			if a != 0xffff {
				t.Errorf("pixel (%d,%d) alpha = %d, want 255", x, y, a>>8)
			}
		}
	}
}

// TestResizeTorchBilinear_SameSize 恒等：不缩放时输出应与输入逐像素一致。
func TestResizeTorchBilinear_SameSize(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(10 + x), G: uint8(20 + y), B: 99, A: 255})
		}
	}
	out := ResizeTorchBilinear(img, 3, 2)
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			if got := out.RGBAAt(x, y); got != img.RGBAAt(x, y) {
				t.Errorf("pixel (%d,%d) = %v, want %v", x, y, got, img.RGBAAt(x, y))
			}
		}
	}
}

// TestResizeTorchBilinear_ZeroSized 空图不 panic，返回对应尺寸空图。
func TestResizeTorchBilinear_ZeroSized(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 0, 0))
	if out := ResizeTorchBilinear(img, 4, 4); out.Bounds().Dx() != 4 || out.Bounds().Dy() != 4 {
		t.Fatalf("zero-size source: bounds = %v, want 4x4", out.Bounds())
	}
	out := ResizeTorchBilinear(img, 0, 0)
	if out.Bounds().Dx() != 0 || out.Bounds().Dy() != 0 {
		t.Fatalf("zero target: bounds = %v, want 0x0", out.Bounds())
	}
}
