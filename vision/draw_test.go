package vision

import (
	"image/color"
	"image/draw"
	"path/filepath"
	"testing"

	"github.com/up-zero/gotool/imageutil"
)

func TestDrawer_DrawText(t *testing.T) {
	d, err := NewTextDrawer("./fonts/NotoSansSC-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	img, err := imageutil.Open("../examples/test.png")
	if err != nil {
		t.Fatal(err)
	}

	srcImage, ok := img.(draw.Image)
	if !ok {
		t.Fatal("image does not support drawing")
	}

	d.DrawText(srcImage, "Hello World", 10, 10, color.Black)
	if err := imageutil.Save(filepath.Join(t.TempDir(), "draw.png"), srcImage, 100); err != nil {
		t.Fatalf("failed to save drawn image: %v", err)
	}
}
