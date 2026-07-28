package examples

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/up-zero/gotool/imageutil"
)

func mustOpenExampleImage(t *testing.T, name string) image.Image {
	t.Helper()
	img, err := imageutil.Open(ExampleImagePath(name))
	if err != nil {
		t.Skipf("%s not found: %v", name, err)
	}
	return img
}

func mustOpenImagePath(t *testing.T, path string) image.Image {
	t.Helper()
	img, err := imageutil.Open(path)
	if err != nil {
		t.Skipf("failed to open %s: %v", path, err)
	}
	return img
}

func exampleArtifactPath(parts ...string) string {
	path := ExampleArtifactPath(parts...)
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	return path
}

// loadImage loads an image from the given file path.
func loadImage(path string) (image.Image, error) {
	return imageutil.Open(path)
}
