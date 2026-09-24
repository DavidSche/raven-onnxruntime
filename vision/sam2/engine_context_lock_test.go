package sam2

// ImageContext 锁语义测试（跨锁字段审计，见项目规范 §17）。
//
// isDestroyed 与 imageEmbeddings 的全部访问经 ctx.mu（Destroy 与 DecodeRaw 互斥，
// 杜绝\"检查通过后 Value 被并发销毁\"的 use-after-free / double-free）。
//
// 零值构造（Value 全 nil）：Destroy 的 nil 检查跳过实际 C 层释放，测试无需真实
// ONNX 会话；DecodeRaw 在 isDestroyed 置位后于锁内首行快速失败，不触碰 engine。
//
// 本机无 cgo 跑不了 -race：并发 Destroy 测试作为 CI -race 门控（修复前无锁版本
// 并发读写 isDestroyed 为数据竞争），本机作并发压力路径。
//
// 覆盖边界（如实说明）：真正的主竞态 DecodeRaw∥Destroy 无法并发驱动——零值 ctx 的
// engine 为 nil，DecodeRaw 会在 decoderSession.NewTensor 处 panic；该路径的 -race
// 覆盖需真实 ONNX 会话（带 DLL 环境），此处仅串行验证 Destroy→DecodeRaw 快速失败。

import (
	"strings"
	"sync"
	"testing"
)

// TestImageContext_Destroy_Idempotent 双重 Destroy 必须幂等（不 panic、不重复释放）。
func TestImageContext_Destroy_Idempotent(t *testing.T) {
	ctx := &ImageContext{}
	ctx.Destroy()
	ctx.Destroy() // 第二次：锁内 isDestroyed 短路
}

// TestImageContext_DecodeRaw_AfterDestroy_FailsFast Destroy 后 DecodeRaw 必须
// 快速失败，不再触碰已销毁的 Value（锁内检查，无 TOCTOU 窗口）。
func TestImageContext_DecodeRaw_AfterDestroy_FailsFast(t *testing.T) {
	ctx := &ImageContext{}
	ctx.Destroy()

	_, err := ctx.DecodeRaw(nil)
	if err == nil {
		t.Fatal("DecodeRaw after Destroy must fail fast")
	}
	if !strings.Contains(err.Error(), "already destroyed") {
		t.Errorf("DecodeRaw after Destroy err = %q, want contains 'already destroyed'", err)
	}
}

// TestImageContext_Destroy_Concurrent 并发 Destroy（-race 门控）：修复前无锁版本
// 并发读写 isDestroyed 为数据竞争；修复后经 ctx.mu 串行、第二次调用短路。
func TestImageContext_Destroy_Concurrent(t *testing.T) {
	ctx := &ImageContext{}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx.Destroy()
		}()
	}
	wg.Wait()
}
