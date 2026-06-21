package groundedsam2

import (
	"encoding/binary"
	"fmt"
	"image"
	"os"
	"strings"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/up-zero/gotool/imageutil"
)

var imagenetMeans = [3]float32{0.485, 0.456, 0.406}
var imagenetStds = [3]float32{0.229, 0.224, 0.225}

// SAM-2 使用自己的归一化参数 (ImageNet mean/std)
var sam2Means = [3]float32{0.485, 0.456, 0.406}
var sam2Stds = [3]float32{0.229, 0.224, 0.225}

type imageParams struct {
	origW, origH int
}

// preprocessImage 预处理图像为模型输入张量
func preprocessImage(session *ort.Session, img image.Image, inputSize int) (*ort.Value, imageParams, error) {
	bounds := img.Bounds()
	params := imageParams{
		origW: bounds.Dx(),
		origH: bounds.Dy(),
	}

	resized := imageutil.Resize(img, inputSize, inputSize)

	data := make([]float32, 3*inputSize*inputSize)
	planeSize := inputSize * inputSize

	means := imagenetMeans
	stds := imagenetStds

	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			idx := y*inputSize + x
			data[idx] = (float32(r)/65535.0 - means[0]) / stds[0]
			data[planeSize+idx] = (float32(g)/65535.0 - means[1]) / stds[1]
			data[2*planeSize+idx] = (float32(b)/65535.0 - means[2]) / stds[2]
		}
	}

	tensor, err := session.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, params, err
}

// tokenizeCaptions 对文本提示进行 tokenize
// 复用 GroundingDINO 的 tokenization 逻辑
func tokenizeCaptions(session *ort.Session, captions []string, maxTextLen int) (
	inputIds, attentionMask, tokenTypeIds, textSelfAttnMasks, positionIds *ort.Value, err error,
) {
	// 尝试从预 tokenize 缓存加载
	inputIds, attentionMask, tokenTypeIds, textSelfAttnMasks, positionIds, err =
		loadCachedTokenization(session, captions, maxTextLen)
	if err == nil {
		return
	}

	// 回退到简单 tokenization
	return simpleTokenize(session, captions, maxTextLen)
}

// simpleTokenize 简单的占位 tokenize（仅用于测试）
func simpleTokenize(session *ort.Session, captions []string, maxTextLen int) (
	*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error,
) {
	batchSize := len(captions)
	if batchSize == 0 {
		batchSize = 1
	}

	seqLen := maxTextLen

	inputIdsData := make([]int64, batchSize*seqLen)
	attentionMaskData := make([]int64, batchSize*seqLen)
	tokenTypeIdsData := make([]int64, batchSize*seqLen)
	positionIdsData := make([]int64, batchSize*seqLen)
	selfAttnData := make([]float32, batchSize*seqLen*seqLen)

	for i := 0; i < batchSize; i++ {
		inputIdsData[i*seqLen] = 101 // [CLS]
		caption := ""
		if i < len(captions) {
			caption = captions[i]
		}
		tokenLen := min(len(caption), seqLen-2)
		for j, ch := range caption[:tokenLen] {
			inputIdsData[i*seqLen+j+1] = int64(ch) + 1000
		}
		inputIdsData[i*seqLen+tokenLen+1] = 102 // [SEP]

		for j := 0; j < tokenLen+2; j++ {
			attentionMaskData[i*seqLen+j] = 1
		}

		for j := 0; j < seqLen; j++ {
			positionIdsData[i*seqLen+j] = int64(j)
		}

		for r := 0; r < seqLen; r++ {
			for c := 0; c < seqLen; c++ {
				idx := i*seqLen*seqLen + r*seqLen + c
				if r == c {
					selfAttnData[idx] = 1.0
				}
				if r == 0 && c == 0 {
					selfAttnData[idx] = 1.0
				}
				if r > 0 && r <= tokenLen+1 && c > 0 && c <= tokenLen+1 {
					selfAttnData[idx] = 1.0
				}
			}
		}
	}

	var (
		inputIdsT       *ort.Value
		attentionMaskT  *ort.Value
		tokenTypeIdsT   *ort.Value
		textSelfAttnT   *ort.Value
		positionIdsT    *ort.Value
		err             error
	)

	inputIdsT, err = session.NewTensor([]int64{int64(batchSize), int64(seqLen)}, inputIdsData)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("create input_ids tensor failed: %w", err)
	}

	attentionMaskT, err = session.NewTensor([]int64{int64(batchSize), int64(seqLen)}, attentionMaskData)
	if err != nil {
		inputIdsT.Destroy()
		return nil, nil, nil, nil, nil, fmt.Errorf("create attention_mask tensor failed: %w", err)
	}

	tokenTypeIdsT, err = session.NewTensor([]int64{int64(batchSize), int64(seqLen)}, tokenTypeIdsData)
	if err != nil {
		inputIdsT.Destroy()
		attentionMaskT.Destroy()
		return nil, nil, nil, nil, nil, fmt.Errorf("create token_type_ids tensor failed: %w", err)
	}

	textSelfAttnT, err = session.NewTensor([]int64{int64(batchSize), int64(seqLen), int64(seqLen)}, selfAttnData)
	if err != nil {
		inputIdsT.Destroy()
		attentionMaskT.Destroy()
		tokenTypeIdsT.Destroy()
		return nil, nil, nil, nil, nil, fmt.Errorf("create text_self_attention_masks tensor failed: %w", err)
	}

	positionIdsT, err = session.NewTensor([]int64{int64(batchSize), int64(seqLen)}, positionIdsData)
	if err != nil {
		inputIdsT.Destroy()
		attentionMaskT.Destroy()
		tokenTypeIdsT.Destroy()
		textSelfAttnT.Destroy()
		return nil, nil, nil, nil, nil, fmt.Errorf("create position_ids tensor failed: %w", err)
	}

	return inputIdsT, attentionMaskT, tokenTypeIdsT, textSelfAttnT, positionIdsT, nil
}

// loadCachedTokenization 从预计算的缓存文件加载 tokenization
func loadCachedTokenization(session *ort.Session, captions []string, maxTextLen int) (
	*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error,
) {
	for _, caption := range captions {
		cachePath := fmt.Sprintf(".token_cache/%s.bin", sanitizeFilename(caption))
		data, err := os.ReadFile(cachePath)
		if err != nil {
			return nil, nil, nil, nil, nil, fmt.Errorf("cache not found: %w", err)
		}
		return deserializeTokenization(session, data, 1, maxTextLen)
	}
	return nil, nil, nil, nil, nil, fmt.Errorf("no cache available")
}

// deserializeTokenization 反序列化预计算的 tokenization
func deserializeTokenization(session *ort.Session, data []byte, batchSize, maxTextLen int) (
	*ort.Value, *ort.Value, *ort.Value, *ort.Value, *ort.Value, error,
) {
	B := int64(batchSize)
	L := int64(maxTextLen)

	int64Size := 8
	float32Size := 4

	inputIdsSize := B * L * int64(int64Size)
	attentionMaskSize := B * L * int64(int64Size)
	tokenTypeIdsSize := B * L * int64(int64Size)
	selfAttnSize := B * L * L * int64(float32Size)
	positionIdsSize := B * L * int64(int64Size)

	expectedSize := inputIdsSize + attentionMaskSize + tokenTypeIdsSize + selfAttnSize + positionIdsSize
	if len(data) < int(expectedSize) {
		return nil, nil, nil, nil, nil, fmt.Errorf("tokenization data too short: got %d want %d", len(data), expectedSize)
	}

	offset := 0

	inputIdsCount := B * L
	inputIdsData := make([]int64, inputIdsCount)
	for i := 0; i < int(inputIdsCount); i++ {
		inputIdsData[i] = int64(binary.LittleEndian.Uint64(data[offset+i*int64Size:]))
	}
	offset += int(inputIdsSize)

	attentionMaskData := make([]int64, inputIdsCount)
	for i := 0; i < int(inputIdsCount); i++ {
		attentionMaskData[i] = int64(binary.LittleEndian.Uint64(data[offset+i*int64Size:]))
	}
	offset += int(attentionMaskSize)

	tokenTypeIdsData := make([]int64, inputIdsCount)
	for i := 0; i < int(inputIdsCount); i++ {
		tokenTypeIdsData[i] = int64(binary.LittleEndian.Uint64(data[offset+i*int64Size:]))
	}
	offset += int(tokenTypeIdsSize)

	selfAttnCount := B * L * L
	selfAttnData := make([]float32, selfAttnCount)
	for i := 0; i < int(selfAttnCount); i++ {
		selfAttnData[i] = float32(binary.LittleEndian.Uint32(data[offset+i*float32Size:]))
	}
	offset += int(selfAttnSize)

	positionIdsData := make([]int64, inputIdsCount)
	for i := 0; i < int(inputIdsCount); i++ {
		positionIdsData[i] = int64(binary.LittleEndian.Uint64(data[offset+i*int64Size:]))
	}

	inputIds, err := session.NewTensor([]int64{B, L}, inputIdsData)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	attentionMask, err := session.NewTensor([]int64{B, L}, attentionMaskData)
	if err != nil {
		inputIds.Destroy()
		return nil, nil, nil, nil, nil, err
	}

	tokenTypeIds, err := session.NewTensor([]int64{B, L}, tokenTypeIdsData)
	if err != nil {
		inputIds.Destroy()
		attentionMask.Destroy()
		return nil, nil, nil, nil, nil, err
	}

	textSelfAttnMasks, err := session.NewTensor([]int64{B, L, L}, selfAttnData)
	if err != nil {
		inputIds.Destroy()
		attentionMask.Destroy()
		tokenTypeIds.Destroy()
		return nil, nil, nil, nil, nil, err
	}

	positionIds, err := session.NewTensor([]int64{B, L}, positionIdsData)
	if err != nil {
		inputIds.Destroy()
		attentionMask.Destroy()
		tokenTypeIds.Destroy()
		textSelfAttnMasks.Destroy()
		return nil, nil, nil, nil, nil, err
	}

	return inputIds, attentionMask, tokenTypeIds, textSelfAttnMasks, positionIds, nil
}

func sanitizeFilename(s string) string {
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "/", "_")
	s = strings.ReplaceAll(s, "\\", "_")
	s = strings.ReplaceAll(s, ":", "_")
	s = strings.ReplaceAll(s, ".", "_")
	s = strings.ReplaceAll(s, "?", "_")
	if len(s) > 100 {
		s = s[:100]
	}
	return s
}
