package groundingdino

import (
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/DavidSche/raven-onnxruntime/vision/tokenizer"
)

var imagenetMeans = [3]float32{0.485, 0.456, 0.406}
var imagenetStds = [3]float32{0.229, 0.224, 0.225}

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

// tokenizeCaptions 对文本提示进行 tokenize，生成 ONNX 文本编码器所需的输入张量
//
// GroundingDINO 使用 BERT tokenizer，需要生成：
//   - input_ids:               [B, L] int64
//   - attention_mask:          [B, L] int64
//   - token_type_ids:          [B, L] int64
//   - text_self_attention_masks: [B, L, L] float32
//   - position_ids:            [B, L] int64
//
// 由于 Go 端没有 BERT tokenizer，这里采用预 tokenize 方案：
// 使用 Python 端预先 tokenize 并保存为二进制文件，或使用简化的 token 映射。
// 推荐做法：在部署时使用 Python 预处理服务或导出 tokenizer 词表。
func tokenizeCaptions(session *ort.Session, captions []string, maxTextLen int) (inputIds, attentionMask, tokenTypeIds, textSelfAttnMasks, positionIds *ort.Value, err error) {
	return tokenizer.TokenizeCaptions(session, captions, maxTextLen)
}

// simpleTokenize 简单的占位 tokenize（仅用于测试）
// 生产环境必须替换为真正的 BERT tokenizer
func simpleTokenize(session *ort.Session, captions []string, maxTextLen int) (*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error) {
	return tokenizer.SimpleTokenize(session, captions, maxTextLen)
}

// loadCachedTokenization 从预计算的缓存文件加载 tokenization
func loadCachedTokenization(session *ort.Session, captions []string, maxTextLen int) (*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error) {
	return tokenizer.LoadCachedTokenization(session, captions, maxTextLen)
}

// deserializeTokenization 反序列化预计算的 tokenization
// 二进制格式：
//
//	[input_ids: B*L int64] [attention_mask: B*L int64] [token_type_ids: B*L int64]
//	[text_self_attention_masks: B*L*L float32] [position_ids: B*L int64]
func deserializeTokenization(session *ort.Session, data []byte, batchSize, maxTextLen int) (*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error) {
	bundle, err := tokenizer.DeserializeTokenization(session, data, batchSize, maxTextLen)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	return bundle.InputIds, bundle.AttentionMask, bundle.TokenTypeIds, bundle.TextSelfAttnMasks, bundle.PositionIds, nil
}

func sanitizeFilename(s string) string {
	return tokenizer.SanitizeFilename(s)
}
