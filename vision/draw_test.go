package vision

import (
	"github.com/up-zero/gotool/imageutil"
	"image/color"
	"image/draw"
	"path/filepath"
	"testing"
)

func TestDrawer_DrawText(t *testing.T) {
	d, err := NewTextDrawer("./fonts/NotoSansSC-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	img, err := imageutil.Open("../assets/logo.png")
	if err != nil {
		t.Fatal(err)
	}

	srcImage, ok := img.(draw.Image)
	if !ok {
		t.Fatal("image does not support drawing")
	}

	d.DrawText(srcImage, "Hello World", 10, 10, color.Black)
	imageutil.Save(filepath.Join(t.TempDir(), "draw.png"), srcImage, 100)
}
