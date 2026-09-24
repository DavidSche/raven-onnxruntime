package vision

// TextDrawer 锁语义测试（跨锁字段审计，见 AGENTS.md「并发字段锁归属」）。
//
// face / fontSize 的全部访问经 d.mu（SetSize 重建、DrawText 读、Close 关闭置 nil
// 互斥），杜绝"绘制中 face 被并发关闭/重建"的数据竞争与潜在 use-after-close。
//
// 覆盖边界（如实说明，2026-08 回退实验核实）：当前 golang.org/x/image v0.34.0 的
// opentype.Face.Close() 为 no-op（不释放资源），故旧版"Close 后继续 DrawString"
// 串行下不崩溃——串行断言（DrawAfterClose/Close 幂等）钉住的是**语义契约**（置 nil
// + 空操作 + 幂等，防御未来字体后端升级引入 use-after-free）；真正的判别点
// SetSize∥DrawText∥Close 的 face/fontSize 并发读写是 Go 数据竞争，由 CI `-race`
// 门控捕获（本机无 cgo 无法运行 -race，本测试作并发压力路径）。字体文件
// vision/fonts/NotoSansSC-Regular.ttf 与既有 TestDrawer_DrawText 同源。

import (
	"image"
	"image/color"
	"sync"
	"testing"
)

// mustNewDrawer 构造 TextDrawer，字体缺失时跳过（与 TestDrawer_DrawText 同策略）。
func mustNewDrawer(t *testing.T) *TextDrawer {
	t.Helper()
	d, err := NewTextDrawer("./fonts/NotoSansSC-Regular.ttf")
	if err != nil {
		t.Fatalf("NewTextDrawer failed: %v", err)
	}
	return d
}

// TestTextDrawer_Close_Idempotent 双重 Close 必须幂等（不 panic、不重复关闭 face）。
func TestTextDrawer_Close_Idempotent(t *testing.T) {
	d := mustNewDrawer(t)
	d.Close()
	d.Close() // 第二次：face 已置 nil，短路
}

// TestTextDrawer_DrawAfterClose_NoPanic Close 后 DrawText 必须空操作——既不能
// 崩溃，也不能把文字画到图上（判别性：旧版 Close 不置 nil、DrawText 无防御，会
// 真的绘制——像素改变；修复后 face 置 nil + 防御返回——像素保持背景色不变）。
func TestTextDrawer_DrawAfterClose_NoPanic(t *testing.T) {
	d := mustNewDrawer(t)
	d.Close()

	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	d.DrawText(img, "after close", 1, 1, color.Black) // 不应 panic

	// 判别性断言：文字未被绘制（修复后为空操作）。旧版会在此处发现像素被修改。
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			if c := img.RGBAAt(x, y); c.R != 0 || c.G != 0 || c.B != 0 || c.A != 0 {
				t.Fatalf("DrawText after Close drew at (%d,%d): rgba=%d,%d,%d,%d", x, y, c.R, c.G, c.B, c.A)
			}
		}
	}
}

// TestTextDrawer_Concurrent_SetSizeDrawClose 并发 SetSize ∥ DrawText ∥ Close（-race 门控）：
// 修复前无锁版本并发读写 face / fontSize 为数据竞争，且绘制中 face 被关闭构成
// use-after-close；修复后经 d.mu 串行。
func TestTextDrawer_Concurrent_SetSizeDrawClose(t *testing.T) {
	d := mustNewDrawer(t)

	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	closeCh := make(chan struct{})
	var wg sync.WaitGroup

	// 绘制 goroutine：持续 DrawText
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-closeCh:
					return
				default:
					d.DrawText(img, "concurrent", 1, 1, color.Black)
				}
			}
		}()
	}

	// 尺寸 goroutine：持续 SetSize（交替尺寸触发重建）
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				_ = d.SetSize(float64(12 + j%8))
			}
		}()
	}

	// Close goroutine：晚于绘制开始，验证 Close 与 DrawText/SetSize 并发安全
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 20; j++ {
			_ = d.SetSize(float64(16 + j%4))
		}
		d.Close()
		close(closeCh)
	}()

	wg.Wait()
}
