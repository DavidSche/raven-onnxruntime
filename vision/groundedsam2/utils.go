package groundedsam2

import (
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/DavidSche/raven-onnxruntime/vision/tokenizer"
)

var imagenetMeans = [3]float32{0.485, 0.456, 0.406}
var imagenetStds = [3]float32{0.229, 0.224, 0.225}

// SAM-2 使用自己的归一化参数 (ImageNet mean/std)
var sam2Means = [3]float32{0.485, 0.456, 0.406}
var sam2Stds = [3]float32{0.229, 0.224, 0.225}

type imageParams struct {
	origW, origH int
	tpadX, padY  int
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

// tokenizeCaptions 对文本提示进行 tokenize
// 复用 GroundingDINO 的 tokenization 逻辑
func tokenizeCaptions(session *ort.Session, captions []string, maxTextLen int) (
	inputIds, attentionMask, tokenTypeIds, textSelfAttnMasks, positionIds *ort.Value, err error,
) {
	return tokenizer.TokenizeCaptions(session, captions, maxTextLen)
}

// simpleTokenize 简单的占位 tokenize（仅用于测试）
func simpleTokenize(session *ort.Session, captions []string, maxTextLen int) (
	*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error,
) {
	return tokenizer.SimpleTokenize(session, captions, maxTextLen)
}

// loadCachedTokenization 从预计算的缓存文件加载 tokenization
func loadCachedTokenization(session *ort.Session, captions []string, maxTextLen int) (
	*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error,
) {
	return tokenizer.LoadCachedTokenization(session, captions, maxTextLen)
}

// deserializeTokenization 反序列化预计算的 tokenization
func deserializeTokenization(session *ort.Session, data []byte, batchSize, maxTextLen int) (
	*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error,
) {
	bundle, err := tokenizer.DeserializeTokenization(session, data, batchSize, maxTextLen)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	return bundle.InputIds, bundle.AttentionMask, bundle.TokenTypeIds, bundle.TextSelfAttnMasks, bundle.PositionIds, nil
}

func sanitizeFilename(s string) string {
	return tokenizer.SanitizeFilename(s)
}
