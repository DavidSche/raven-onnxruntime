package vision

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// TextDrawer is a text drawing utility
type TextDrawer struct {
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
func (d *TextDrawer) SetSize(fontSize float64) error {
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

// Close releases resources
func (d *TextDrawer) Close() {
	if d.face != nil {
		d.face.Close()
	}
}
