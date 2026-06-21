package groundedsam2

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// SegEngine Grounded-SAM-2 分割引擎
// 推理流程:
//  1. 图像 -> GDINO image_encoder -> 图像特征
//  2. 文本 -> GDINO text_encoder  -> 文本特征
//  3. 图像特征 + 文本特征 -> GDINO detector -> 检测框
//  4. 图像 -> SAM2 image_encoder  -> 图像嵌入 + 高分辨率特征
//  5. 图像嵌入 + 检测框 -> SAM2 mask_decoder -> 分割掩码
type SegEngine struct {
	// GroundingDINO 子模型
	gdinoImageEncoder *ort.Session
	gdinoTextEncoder  *ort.Session
	gdinoDetector     *ort.Session

	// SAM-2 子模型
	sam2ImageEncoder *ort.Session
	sam2MaskDecoder  *ort.Session

	// 共享引擎
	engine *ort.Engine

	config Config
	params *modelParams
}

// NewSegEngine 创建 Grounded-SAM-2 分割引擎
func NewSegEngine(cfg Config) (*SegEngine, error) {
	// 从 manifest.json 解析子模型路径
	paths, params, err := cfg.resolveSubModelPaths()
	if err != nil {
		return nil, fmt.Errorf("failed to resolve sub-model paths: %w", err)
	}

	ortlog.Infow("creating Grounded-SAM-2 segmentation engine",
		"modelPath", cfg.ModelPath,
		"gdinoImageEncoder", paths.gdinoImageEncoder,
		"gdinoTextEncoder", paths.gdinoTextEncoder,
		"gdinoDetector", paths.gdinoDetector,
		"sam2ImageEncoder", paths.sam2ImageEncoder,
		"sam2MaskDecoder", paths.sam2MaskDecoder,
		"inputSize", cfg.InputSize,
		"sam2ImageSize", cfg.Sam2ImageSize,
		"confThreshold", cfg.ConfThreshold,
	)

	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}
	if err := oc.New(); err != nil {
		return nil, fmt.Errorf("initialization failed: %w", err)
	}

	e := &SegEngine{config: cfg, engine: oc.OnnxEngine, params: params}

	// 创建 GroundingDINO 子模型 sessions
	e.gdinoImageEncoder, err = e.engine.NewSession(paths.gdinoImageEncoder, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create GDINO image encoder session: %w", err)
	}

	e.gdinoTextEncoder, err = e.engine.NewSession(paths.gdinoTextEncoder, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create GDINO text encoder session: %w", err)
	}

	e.gdinoDetector, err = e.engine.NewSession(paths.gdinoDetector, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create GDINO detector session: %w", err)
	}

	// 创建 SAM-2 子模型 sessions
	e.sam2ImageEncoder, err = e.engine.NewSession(paths.sam2ImageEncoder, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create SAM2 image encoder session: %w", err)
	}

	e.sam2MaskDecoder, err = e.engine.NewSession(paths.sam2MaskDecoder, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create SAM2 mask decoder session: %w", err)
	}

	oc.Destroy()

	ortlog.Infow("Grounded-SAM-2 segmentation engine created successfully")
	return e, nil
}

// Predict 执行开放词汇实例分割
func (e *SegEngine) Predict(img image.Image, captions []string) ([]SegResult, error) {
	// Step 1: GroundingDINO 检测
	detBoxes, detScores, detLabels, _, err := e.runGroundingDINO(img, captions)
	if err != nil {
		return nil, fmt.Errorf("groundingdino detection failed: %w", err)
	}

	if len(detBoxes) == 0 {
		return nil, nil
	}

	// Step 2: SAM-2 分割
	masks, iouScores, err := e.runSAM2(img, detBoxes)
	if err != nil {
		return nil, fmt.Errorf("sam2 segmentation failed: %w", err)
	}

	// Step 3: 组装结果
	results := make([]SegResult, 0, len(detBoxes))
	for i := range detBoxes {
		mask := masks[i]
		iouScore := float32(0)
		if i < len(iouScores) {
			iouScore = iouScores[i]
		}

		results = append(results, SegResult{
			ClassID:   i,
			ClassName: detLabels[i],
			Score:     detScores[i],
			Box:       detBoxes[i],
			Mask:      mask,
			IoUScore:  iouScore,
		})
	}

	return results, nil
}

// runGroundingDINO 运行 GroundingDINO 检测流程
func (e *SegEngine) runGroundingDINO(img image.Image, captions []string) (
	boxes []image.Rectangle, scores []float32, labels []string,
	params imageParams, err error,
) {
	// 1. 图像编码
	inputTensor, params, err := preprocessImage(e.gdinoImageEncoder, img, e.config.InputSize)
	if err != nil {
		return nil, nil, nil, params, fmt.Errorf("preprocess image failed: %w", err)
	}
	defer inputTensor.Destroy()

	gdinoImgOutput, err := e.gdinoImageEncoder.Run(map[string]*ort.Value{"images": inputTensor})
	if err != nil {
		return nil, nil, nil, params, fmt.Errorf("gdino image encoder run failed: %w", err)
	}
	defer func() {
		for _, v := range gdinoImgOutput {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	// 2. 文本编码
	inputIds, attentionMask, tokenTypeIds, textSelfAttnMasks, positionIds, err :=
		tokenizeCaptions(e.gdinoTextEncoder, captions, e.config.MaxTextLen)
	if err != nil {
		return nil, nil, nil, params, fmt.Errorf("tokenize captions failed: %w", err)
	}
	defer func() {
		if inputIds != nil {
			inputIds.Destroy()
		}
		if attentionMask != nil {
			attentionMask.Destroy()
		}
		if tokenTypeIds != nil {
			tokenTypeIds.Destroy()
		}
		if textSelfAttnMasks != nil {
			textSelfAttnMasks.Destroy()
		}
		if positionIds != nil {
			positionIds.Destroy()
		}
	}()

	textInputs := map[string]*ort.Value{
		"input_ids":                 inputIds,
		"attention_mask":            attentionMask,
		"token_type_ids":            tokenTypeIds,
		"text_self_attention_masks": textSelfAttnMasks,
		"position_ids":              positionIds,
	}

	gdinoTxtOutput, err := e.gdinoTextEncoder.Run(textInputs)
	if err != nil {
		return nil, nil, nil, params, fmt.Errorf("gdino text encoder run failed: %w", err)
	}
	defer func() {
		for _, v := range gdinoTxtOutput {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	// 3. 检测
	detInputs := make(map[string]*ort.Value)
	for k, v := range gdinoImgOutput {
		detInputs[k] = v
	}
	for k, v := range gdinoTxtOutput {
		detInputs[k] = v
	}

	detOutput, err := e.gdinoDetector.Run(detInputs)
	if err != nil {
		return nil, nil, nil, params, fmt.Errorf("gdino detector run failed: %w", err)
	}
	defer func() {
		for _, v := range detOutput {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	// 4. 后处理检测框
	boxes, scores, labels = e.postprocessDetections(detOutput, captions, params)

	return boxes, scores, labels, params, nil
}

// runSAM2 运行 SAM-2 分割流程
func (e *SegEngine) runSAM2(img image.Image, boxes []image.Rectangle) ([]*image.Gray, []float32, error) {
	// 1. SAM-2 图像编码
	sam2Input, sam2Params, err := preprocessImage(e.sam2ImageEncoder, img, e.config.Sam2ImageSize)
	if err != nil {
		return nil, nil, fmt.Errorf("sam2 preprocess image failed: %w", err)
	}
	defer sam2Input.Destroy()

	sam2ImgOutput, err := e.sam2ImageEncoder.Run(map[string]*ort.Value{"images": sam2Input})
	if err != nil {
		return nil, nil, fmt.Errorf("sam2 image encoder run failed: %w", err)
	}
	defer func() {
		for _, v := range sam2ImgOutput {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	// 2. 将检测框转换为 SAM-2 输入格式
	boxCoords, err := e.convertBoxesToSAM2Input(boxes, sam2Params)
	if err != nil {
		return nil, nil, fmt.Errorf("convert boxes failed: %w", err)
	}
	defer boxCoords.Destroy()

	// 3. SAM-2 掩码解码
	decoderInputs := map[string]*ort.Value{
		"image_embed":     sam2ImgOutput["image_embed"],
		"high_res_feat_0": sam2ImgOutput["high_res_feat_0"],
		"high_res_feat_1": sam2ImgOutput["high_res_feat_1"],
		"box_coords":      boxCoords,
	}

	maskOutput, err := e.sam2MaskDecoder.Run(decoderInputs)
	if err != nil {
		return nil, nil, fmt.Errorf("sam2 mask decoder run failed: %w", err)
	}
	defer func() {
		for _, v := range maskOutput {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	// 4. 后处理掩码
	masks, iouScores := e.postprocessMasks(maskOutput, boxes, sam2Params)

	return masks, iouScores, nil
}

// postprocessDetections 后处理 GroundingDINO 检测结果
func (e *SegEngine) postprocessDetections(
	output map[string]*ort.Value,
	captions []string,
	params imageParams,
) ([]image.Rectangle, []float32, []string) {
	logitsVal, ok := output["pred_logits"]
	if !ok || logitsVal == nil {
		return nil, nil, nil
	}

	boxesVal, ok := output["pred_boxes"]
	if !ok || boxesVal == nil {
		return nil, nil, nil
	}

	logitsData, err := ort.GetTensorData[float32](logitsVal)
	if err != nil {
		return nil, nil, nil
	}

	boxesData, err := ort.GetTensorData[float32](boxesVal)
	if err != nil {
		return nil, nil, nil
	}

	logitsShape, _ := logitsVal.GetShape()
	boxesShape, _ := boxesVal.GetShape()

	if len(logitsShape) < 2 || len(boxesShape) < 2 {
		return nil, nil, nil
	}

	numQueries := int(logitsShape[1])

	type detection struct {
		box   image.Rectangle
		score float32
		label string
	}

	var dets []detection

	for q := 0; q < numQueries; q++ {
		logit := logitsData[q]
		score := float32(math.Exp(float64(logit)))
		score = score / (1.0 + score) // sigmoid

		if score < e.config.ConfThreshold {
			continue
		}

		// cxcywh -> xyxy (归一化坐标)
		cx := boxesData[q*4+0]
		cy := boxesData[q*4+1]
		w := boxesData[q*4+2]
		h := boxesData[q*4+3]

		x1 := (cx - w/2) * float32(params.origW)
		y1 := (cy - h/2) * float32(params.origH)
		x2 := (cx + w/2) * float32(params.origW)
		y2 := (cy + h/2) * float32(params.origH)

		// 裁剪到图像范围
		if x1 < 0 {
			x1 = 0
		}
		if y1 < 0 {
			y1 = 0
		}
		if x2 > float32(params.origW) {
			x2 = float32(params.origW)
		}
		if y2 > float32(params.origH) {
			y2 = float32(params.origH)
		}

		// 确定标签
		label := "object"
		if len(captions) > 0 {
			label = captions[0]
		}

		dets = append(dets, detection{
			box:   image.Rect(int(x1), int(y1), int(x2), int(y2)),
			score: score,
			label: label,
		})
	}

	// 按置信度排序
	sort.Slice(dets, func(i, j int) bool {
		return dets[i].score > dets[j].score
	})

	// 限制最大检测数
	if len(dets) > e.config.MaxDetections {
		dets = dets[:e.config.MaxDetections]
	}

	resultBoxes := make([]image.Rectangle, len(dets))
	resultScores := make([]float32, len(dets))
	resultLabels := make([]string, len(dets))

	for i, d := range dets {
		resultBoxes[i] = d.box
		resultScores[i] = d.score
		resultLabels[i] = d.label
	}

	return resultBoxes, resultScores, resultLabels
}

// convertBoxesToSAM2Input 将检测框转换为 SAM-2 mask decoder 输入格式
// 检测框为像素坐标 (xyxy)，SAM-2 需要归一化坐标 [0,1]
func (e *SegEngine) convertBoxesToSAM2Input(boxes []image.Rectangle, params imageParams) (*ort.Value, error) {
	numBoxes := len(boxes)
	if numBoxes == 0 {
		numBoxes = 1
	}

	boxData := make([]float32, numBoxes*4)
	imgW := float32(params.origW)
	imgH := float32(params.origH)

	for i, box := range boxes {
		boxData[i*4+0] = float32(box.Min.X) / imgW // x1
		boxData[i*4+1] = float32(box.Min.Y) / imgH // y1
		boxData[i*4+2] = float32(box.Max.X) / imgW // x2
		boxData[i*4+3] = float32(box.Max.Y) / imgH // y2
	}

	return e.sam2MaskDecoder.NewTensor([]int64{1, int64(numBoxes), 4}, boxData)
}

// postprocessMasks 后处理 SAM-2 掩码结果
func (e *SegEngine) postprocessMasks(
	output map[string]*ort.Value,
	boxes []image.Rectangle,
	params imageParams,
) ([]*image.Gray, []float32) {
	masksVal, ok := output["masks"]
	if !ok || masksVal == nil {
		return nil, nil
	}

	iouVal := output["iou_predictions"]

	masksData, err := ort.GetTensorData[float32](masksVal)
	if err != nil {
		return nil, nil
	}

	masksShape, _ := masksVal.GetShape()
	if len(masksShape) < 4 {
		return nil, nil
	}

	numBoxes := int(masksShape[1])
	maskH := int(masksShape[2])
	maskW := int(masksShape[3])

	var iouScores []float32
	if iouVal != nil {
		iouData, err := ort.GetTensorData[float32](iouVal)
		if err == nil {
			iouScores = make([]float32, numBoxes)
			for i := range iouScores {
				if i < len(iouData) {
					iouScores[i] = iouData[i]
				}
			}
		}
	}

	// 将掩码调整到原始图像尺寸并转换为 image.Gray
	result := make([]*image.Gray, numBoxes)
	for i := 0; i < numBoxes; i++ {
		gray := image.NewGray(image.Rect(0, 0, params.origW, params.origH))

		for y := 0; y < params.origH; y++ {
			for x := 0; x < params.origW; x++ {
				// 双线性插值
				srcY := float64(y) * float64(maskH) / float64(params.origH)
				srcX := float64(x) * float64(maskW) / float64(params.origW)

				sy := int(math.Min(srcY, float64(maskH-1)))
				sx := int(math.Min(srcX, float64(maskW-1)))

				idx := i*maskH*maskW + sy*maskW + sx
				val := float32(0)
				if idx < len(masksData) {
					val = masksData[idx]
				}

				// 阈值化
				if val > e.config.MaskThreshold {
					gray.SetGray(x, y, color.Gray{Y: 255})
				} else {
					gray.SetGray(x, y, color.Gray{Y: 0})
				}
			}
		}

		result[i] = gray
	}

	return result, iouScores
}

// Destroy 销毁引擎，释放所有资源
func (e *SegEngine) Destroy() {
	if e.gdinoImageEncoder != nil {
		e.gdinoImageEncoder.Destroy()
		e.gdinoImageEncoder = nil
	}
	if e.gdinoTextEncoder != nil {
		e.gdinoTextEncoder.Destroy()
		e.gdinoTextEncoder = nil
	}
	if e.gdinoDetector != nil {
		e.gdinoDetector.Destroy()
		e.gdinoDetector = nil
	}
	if e.sam2ImageEncoder != nil {
		e.sam2ImageEncoder.Destroy()
		e.sam2ImageEncoder = nil
	}
	if e.sam2MaskDecoder != nil {
		e.sam2MaskDecoder.Destroy()
		e.sam2MaskDecoder = nil
	}
}
