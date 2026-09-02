package sam3

import (
	"fmt"
	"image"
	"sync"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/imageutil"
)

type Engine struct {
	visionSession  *ort.Session
	textSession    *ort.Session
	decoderSession *ort.Session
	config         Config
}

func NewEngine(cfg Config) (*Engine, error) {
	// 从 manifest.json 解析子模型路径
	visionPath, textPath, decoderPath, err := cfg.resolveSubModelPaths()
	if err != nil {
		return nil, fmt.Errorf("failed to resolve sub-model paths: %w", err)
	}

	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}
	if err := oc.New(); err != nil {
		return nil, err
	}

	visSession, err := oc.OnnxEngine.NewSession(visionPath, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create vision encoder ONNX session: %w", err)
	}

	txtSession, err := oc.OnnxEngine.NewSession(textPath, oc.SessionOptions)
	if err != nil {
		visSession.Destroy()
		oc.Destroy()
		return nil, fmt.Errorf("failed to create text encoder ONNX session: %w", err)
	}

	decSession, err := oc.OnnxEngine.NewSession(decoderPath, oc.SessionOptions)
	oc.Destroy()
	if err != nil {
		visSession.Destroy()
		txtSession.Destroy()
		return nil, fmt.Errorf("failed to create decoder ONNX session: %w", err)
	}

	return &Engine{
		visionSession:  visSession,
		textSession:    txtSession,
		decoderSession: decSession,
		config:         cfg,
	}, nil
}

func (e *Engine) Destroy() {
	if e.visionSession != nil {
		e.visionSession.Destroy()
	}
	if e.textSession != nil {
		e.textSession.Destroy()
	}
	if e.decoderSession != nil {
		e.decoderSession.Destroy()
	}
}

type ImageContext struct {
	engine       *Engine
	fpnFeat0     *ort.Value
	fpnFeat1     *ort.Value
	fpnFeat2     *ort.Value
	fpnPos2      *ort.Value
	textFeatures *ort.Value
	textMask     *ort.Value

	origW, origH   int
	scaleX, scaleY float32
	newW, newH     int
	isDestroyed    bool

	// 字段锁归属（跨锁字段审计结论，见项目规范 §17）：isDestroyed 与 fpnFeat*/text*
	// 的全部访问经 mu——Destroy 与 DecodeRaw 可能并发，无锁时"检查通过后 Value 被
	// 并发销毁"构成 use-after-free / double-free（与 raven-go simulation 的 stopped 同类）。
	mu sync.Mutex
}

func (e *Engine) EncodeImage(img image.Image) (*ImageContext, error) {
	if img == nil {
		return nil, fmt.Errorf("input image is nil")
	}
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()
	if origW == 0 || origH == 0 {
		return nil, fmt.Errorf("invalid image dimensions: %dx%d", origW, origH)
	}

	scaleX := float32(inputSize) / float32(origW)
	scaleY := float32(inputSize) / float32(origH)
	newW := inputSize
	newH := inputSize

	resizedImg := imageutil.Resize(img, newW, newH)
	tensorData := preprocessImage(resizedImg, inputSize, inputSize)

	inputTensor, err := e.visionSession.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, tensorData)
	if err != nil {
		return nil, fmt.Errorf("failed to create image input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	visOutputs, err := e.visionSession.Run(map[string]*ort.Value{
		"images": inputTensor,
	})
	if err != nil {
		return nil, fmt.Errorf("vision encoder inference failed: %w", err)
	}

	for name, val := range visOutputs {
		shape, _ := val.GetShape()
		data, err := ort.GetTensorData[float32](val)
		if err == nil && len(data) > 0 {
			vmin, vmax := data[0], data[0]
			for _, v := range data {
				if v < vmin {
					vmin = v
				}
				if v > vmax {
					vmax = v
				}
			}
			ortlog.Debugw("sam3 vision output",
				"name", name,
				"shape", shape,
				"min", vmin,
				"max", vmax)
		}
	}

	textFeatures, textMask, err := e.encodeDummyText()
	if err != nil {
		for _, v := range visOutputs {
			v.Destroy()
		}
		return nil, fmt.Errorf("text encoder inference failed: %w", err)
	}

	ctx := &ImageContext{
		engine:       e,
		fpnFeat0:     visOutputs["fpn_feat_0"],
		fpnFeat1:     visOutputs["fpn_feat_1"],
		fpnFeat2:     visOutputs["fpn_feat_2"],
		fpnPos2:      visOutputs["fpn_pos_2"],
		textFeatures: textFeatures,
		textMask:     textMask,
		origW:        origW,
		origH:        origH,
		scaleX:       scaleX,
		scaleY:       scaleY,
		newW:         newW,
		newH:         newH,
	}

	return ctx, nil
}

func (e *Engine) encodeDummyText() (*ort.Value, *ort.Value, error) {
	inputIds := make([]int64, textSeqLen)
	attentionMask := make([]int64, textSeqLen)
	for i := 0; i < textSeqLen; i++ {
		inputIds[i] = padTokenId
	}
	attentionMask[0] = 1

	tInputIds, err := e.textSession.NewTensor([]int64{1, textSeqLen}, inputIds)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create text input_ids tensor: %w", err)
	}
	defer tInputIds.Destroy()

	tAttentionMask, err := e.textSession.NewTensor([]int64{1, textSeqLen}, attentionMask)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create text attention_mask tensor: %w", err)
	}
	defer tAttentionMask.Destroy()

	outputs, err := e.textSession.Run(map[string]*ort.Value{
		"input_ids":      tInputIds,
		"attention_mask": tAttentionMask,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("text encoder run failed: %w", err)
	}
	// 注意：此处不得 defer ort.DestroyValues(outputs)——返回的 text_features/text_mask
	// 所有权移交给 ImageContext（由 ctx.Destroy 统一释放）。若在此销毁，EncodeImage 存入
	// ctx 后 DecodeRaw 将读取已释放的 OrtValue（use-after-free），与 efficientsam3 一致。

	textFeatShape, _ := outputs["text_features"].GetShape()
	textMaskShape, _ := outputs["text_mask"].GetShape()
	ortlog.Debugw("sam3 text output",
		"featuresShape", textFeatShape,
		"maskShape", textMaskShape)

	return outputs["text_features"], outputs["text_mask"], nil
}

func (ctx *ImageContext) Destroy() { // 幂等，持锁
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.isDestroyed {
		return
	}
	for _, v := range []*ort.Value{ctx.fpnFeat0, ctx.fpnFeat1, ctx.fpnFeat2, ctx.fpnPos2, ctx.textFeatures, ctx.textMask} {
		if v != nil {
			v.Destroy()
		}
	}
	ctx.isDestroyed = true
}

type Result struct {
	Mask   []uint8
	Score  float32
	Width  int
	Height int
}

func (ctx *ImageContext) DecodeRaw(points []Point) (*Result, error) {
	// 全程持 ctx.mu：与 Destroy 互斥，保证解码期间 Value 不被并发销毁（含排队场景——
	// DecodeRaw 先拿到锁时 Destroy 等待解码完成；Destroy 先拿到锁时 DecodeRaw 快速失败）。
	// 代价：同一 ImageContext 上的并发 DecodeRaw 被串行化（Value 无克隆 API，这是防
	// use-after-free 的最小正确方案；勿“优化”回只锁 isDestroyed 检查，那会重开竞态）。
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.isDestroyed {
		return nil, fmt.Errorf("image features already destroyed")
	}

	boxes, boxLabels := pointsToBoxes(points, ctx.scaleX, ctx.scaleY, float32(ctx.newW), float32(ctx.newH))
	numBoxes := int64(len(boxes) / 4)

	tBoxes, err := ctx.engine.decoderSession.NewTensor([]int64{1, numBoxes, 4}, boxes)
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder boxes tensor: %w", err)
	}
	defer tBoxes.Destroy()

	tBoxLabels, err := ctx.engine.decoderSession.NewTensor([]int64{1, numBoxes}, boxLabels)
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder box_labels tensor: %w", err)
	}
	defer tBoxLabels.Destroy()

	inputValues := map[string]*ort.Value{
		"fpn_feat_0":         ctx.fpnFeat0,
		"fpn_feat_1":         ctx.fpnFeat1,
		"fpn_feat_2":         ctx.fpnFeat2,
		"fpn_pos_2":          ctx.fpnPos2,
		"text_features":      ctx.textFeatures,
		"text_mask":          ctx.textMask,
		"input_boxes":        tBoxes,
		"input_boxes_labels": tBoxLabels,
	}

	outputs, err := ctx.engine.decoderSession.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("decoder inference failed: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Destroy()
		}
	}()

	predMasks, err := ort.GetTensorData[float32](outputs["pred_masks"])
	if err != nil {
		return nil, fmt.Errorf("failed to get pred_masks: %w", err)
	}
	predLogits, err := ort.GetTensorData[float32](outputs["pred_logits"])
	if err != nil {
		return nil, fmt.Errorf("failed to get pred_logits: %w", err)
	}
	presenceLogits, err := ort.GetTensorData[float32](outputs["presence_logits"])
	if err != nil {
		return nil, fmt.Errorf("failed to get presence_logits: %w", err)
	}

	presenceScore := sigmoid(presenceLogits[0])

	maskShape, _ := outputs["pred_masks"].GetShape()
	logitsH := int(maskShape[2])
	logitsW := int(maskShape[3])
	pixelsPerMask := logitsH * logitsW

	bestIdx := 0
	bestScore := float32(-100.0)
	for i := 0; i < len(predLogits); i++ {
		score := sigmoid(predLogits[i]) * presenceScore
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	start := bestIdx * pixelsPerMask
	end := start + pixelsPerMask
	if end > len(predMasks) {
		return nil, fmt.Errorf("mask index out of range: start=%d end=%d len=%d", start, end, len(predMasks))
	}
	bestMaskLogits := predMasks[start:end]

	finalMask := upscaleMaskLogits(bestMaskLogits, logitsH, logitsW, ctx.origW, ctx.origH)

	return &Result{
		Mask:   finalMask,
		Score:  bestScore,
		Width:  ctx.origW,
		Height: ctx.origH,
	}, nil
}

func (ctx *ImageContext) Decode(points []Point) (image.Image, float32, error) {
	result, err := ctx.DecodeRaw(points)
	if err != nil {
		return nil, 0, err
	}

	img := image.NewGray(image.Rect(0, 0, result.Width, result.Height))
	copy(img.Pix, result.Mask)
	return img, result.Score, nil
}

func pointsToBoxes(points []Point, scaleX, scaleY, imgW, imgH float32) ([]float32, []int64) {
	var boxes []float32
	var labels []int64

	i := 0
	for i < len(points) {
		if i+1 < len(points) && points[i].Label == LabelPosBox {
			x1 := points[i].X * scaleX
			y1 := points[i].Y * scaleY
			x2 := points[i+1].X * scaleX
			y2 := points[i+1].Y * scaleY

			cx := (x1 + x2) / 2.0 / imgW
			cy := (y1 + y2) / 2.0 / imgH
			w := (x2 - x1) / imgW
			h := (y2 - y1) / imgH

			boxes = append(boxes, cx, cy, w, h)
			labels = append(labels, 1)
			i += 2
		} else {
			px := points[i].X * scaleX
			py := points[i].Y * scaleY
			cx := px / imgW
			cy := py / imgH
			w := 10.0 / imgW
			h := 10.0 / imgH

			boxes = append(boxes, cx, cy, w, h)
			lbl := int64(1)
			if points[i].Label == LabelNegBox {
				lbl = 0
			}
			labels = append(labels, lbl)
			i++
		}
	}

	return boxes, labels
}
