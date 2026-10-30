package groundingdino

import (
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
)

type imageParams struct {
	origW, origH int
}

// preprocessImage 预处理图像为模型输入张量
func preprocessImage(session *ort.Session, img image.Image, inputSize int, ppCfg vision.PreprocessConfig) (*ort.Value, imageParams, error) {
	bounds := img.Bounds()
	params := imageParams{
		origW: bounds.Dx(),
		origH: bounds.Dy(),
	}

	means, stds := vision.GetNormalizeParams(ppCfg)
	resized := vision.Resize(img, inputSize, inputSize, ppCfg.Interpolation)

	data := make([]float32, 3*inputSize*inputSize)
	if err := vision.FillCHWFromImage(data, resized, inputSize*inputSize, inputSize, inputSize, inputSize, means, stds); err != nil {
		return nil, imageParams{}, err
	}

	tensor, err := session.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, params, err
}
