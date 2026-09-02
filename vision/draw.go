package vision

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"os"
	"sync"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// TextDrawer is a text drawing utility
//
// 字段锁归属（2026-08 审计，见 AGENTS.md「并发字段锁归属」）：face / fontSize 的
// 全部访问经 mu——SetSize（重建 face）、DrawText（读 face）、Close（关闭并置 nil）
// 可能并发，无锁时"绘制中 face 被并发关闭/重建"构成 use-after-close 与数据竞争。
// font 构造后只读（SetSize 经 mu 内读取，仍安全）。Close 幂等：关闭后 face 置 nil，
// 后续 DrawText 空操作、重复 Close 短路。注意与 sam* 的 ImageContext 终态 Destroy
// 不同：Close 后 SetSize 仍可从 font 重建 face（drawer 可“复活”），属有意契约。
type TextDrawer struct {
	mu       sync.Mutex
	font     *opentype.Font
	face     font.Face
	fontSize float64
}

// NewTextDrawer creates a text drawing utility
//
// # Params:
//
//	fontPath: path to the font file
func NewTextDrawer(fontPath string) (*TextDrawer, error) {
	fontBytes, err := os.ReadFile(fontPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open font file: %w", err)
	}

	ttFont, err := opentype.Parse(fontBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse font file: %w", err)
	}

	d := &TextDrawer{font: ttFont}
	if err := d.SetSize(12); err != nil {
		return nil, err
	}
	return d, nil
}

// SetSize dynamically adjusts the font size
//
// # Params:
//
//	fontSize: font size
//
// 持锁：与 DrawText / Close 互斥，避免绘制中重建 / 关闭 face（use-after-close）。
func (d *TextDrawer) SetSize(fontSize float64) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.face != nil && d.fontSize == fontSize {
		return nil
	}

	// release old Face and immediately clear the reference so that, if
	// opentype.NewFace fails below, d.face does not point to a closed Face.
	if d.face != nil {
		d.face.Close()
		d.face = nil
	}

	nf, err := opentype.NewFace(d.font, &opentype.FaceOptions{
		Size:    fontSize,
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return err
	}

	d.face = nf
	d.fontSize = fontSize
	return nil
}

// DrawText draws text on the image
//
// # Params:
//
//	img: the image to draw on
//	text: the text to draw
//	x, y: drawing coordinates
//	c: drawing color
func (d *TextDrawer) DrawText(img draw.Image, text string, x, y int, c color.Color) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.face == nil {
		return // 已关闭或未初始化：空操作，避免 nil / 已关闭 face 的崩溃
	}

	point := fixed.Point26_6{
		X: fixed.I(x),
		Y: fixed.I(y),
	}

	d1 := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c), // text color source
		Face: d.face,
		Dot:  point, // starting drawing point
	}
	d1.DrawString(text)
}

// Close releases resources（幂等，持锁）
//
// 关闭后 face 置 nil：重复 Close 短路、后续 DrawText 空操作，杜绝已关闭 face 的使用。
func (d *TextDrawer) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.face != nil {
		d.face.Close()
		d.face = nil
	}
}
