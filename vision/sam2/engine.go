package sam2

import (
	"fmt"
	"image"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/imageutil"
)

// Engine holds ONNX Sessions and is responsible for creating ImageContext.
type Engine struct {
	encoderSession *ort.Session
	decoderSession *ort.Session
	config         Config
}

// NewEngine initializes the SAM2 engine
func NewEngine(cfg Config) (*Engine, error) {
	// 从 manifest.json 解析子模型路径
	encoderPath, decoderPath, err := cfg.resolveSubModelPaths()
	if err != nil {
		return nil, fmt.Errorf("failed to resolve sub-model paths: %w", err)
	}

	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}
	// initialize ONNX
	if err := oc.New(); err != nil {
		return nil, err
	}

	// encoder session
	encSession, err := oc.OnnxEngine.NewSession(encoderPath, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create encoder ONNX session: %w", err)
	}

	// decoder session
	decSession, err := oc.OnnxEngine.NewSession(decoderPath, oc.SessionOptions)
	oc.Destroy()
	if err != nil {
		encSession.Destroy()
		return nil, fmt.Errorf("failed to create decoder ONNX session: %w", err)
	}

	return &Engine{
		encoderSession: encSession,
		decoderSession: decSession,
		config:         cfg,
	}, nil
}

// Destroy releases all resources
func (e *Engine) Destroy() {
	if e.encoderSession != nil {
		e.encoderSession.Destroy()
	}
	if e.decoderSession != nil {
		e.decoderSession.Destroy()
	}
}

// ImageContext holds image-specific feature caches and parameters
type ImageContext struct {
	engine          *Engine
	imageEmbeddings map[string]*ort.Value

	origW, origH int
	scale        float32
	newW, newH   int
	isDestroyed  bool
}

// EncodeImage extracts image features
func (e *Engine) EncodeImage(img image.Image) (*ImageContext, error) {
	if img == nil {
		return nil, fmt.Errorf("input image is nil")
	}
	// preprocess
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()
	if origW == 0 || origH == 0 {
		return nil, fmt.Errorf("invalid image dimensions: %dx%d", origW, origH)
	}

	scale := float32(inputSize) / float32(max(origW, origH))
	newW := int(float32(origW) * scale)
	newH := int(float32(origH) * scale)

	resizedImg := imageutil.Resize(img, newW, newH)
	tensorData := normalizeAndPad(resizedImg, inputSize, inputSize)

	// create input tensor
	inputTensor, err := e.encoderSession.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, tensorData)
	if err != nil {
		return nil, fmt.Errorf("failed to create image input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// encoder inference
	inputValues := map[string]*ort.Value{
		"pixel_values": inputTensor,
	}
	outputs, err := e.encoderSession.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("encoder inference failed: %w", err)
	}

	ctx := &ImageContext{
		engine:          e,
		imageEmbeddings: outputs,
		origW:           origW,
		origH:           origH,
		scale:           scale,
		newW:            newW,
		newH:            newH,
	}

	return ctx, nil
}

// Destroy releases the image feature cache
func (ctx *ImageContext) Destroy() {
	if ctx.isDestroyed {
		return
	}
	for _, v := range ctx.imageEmbeddings {
		if v != nil {
			v.Destroy()
		}
	}
	ctx.imageEmbeddings = nil
	ctx.isDestroyed = true
}

// Result holds the mask prediction result
type Result struct {
	Mask   []uint8 // 0 or 255
	Score  float32
	Width  int
	Height int
}

// DecodeRaw decodes the mask and returns the raw result
func (ctx *ImageContext) DecodeRaw(points []Point) (*Result, error) {
	if ctx.isDestroyed {
		return nil, fmt.Errorf("image features already destroyed")
	}

	// coordinate conversion
	coords := make([]float32, 0, len(points)*2)
	labels := make([]int64, 0, len(points))

	for _, pt := range points {
		coords = append(coords, pt.X*ctx.scale, pt.Y*ctx.scale)
		labels = append(labels, int64(pt.Label))
	}

	numPoints := int64(len(points))

	// prepare decoder tensors
	tPoints, err := ctx.engine.decoderSession.NewTensor([]int64{1, 1, numPoints, 2}, coords)
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder points tensor: %w", err)
	}
	defer tPoints.Destroy()

	tLabels, err := ctx.engine.decoderSession.NewTensor([]int64{1, 1, numPoints}, labels)
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder labels tensor: %w", err)
	}
	defer tLabels.Destroy()

	// box is controlled via point
	tBoxes, err := ctx.engine.decoderSession.NewTensor([]int64{1, 0, 4}, []float32{})
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder boxes tensor: %w", err)
	}
	defer tBoxes.Destroy()

	inputValues := map[string]*ort.Value{
		"input_points": tPoints,
		"input_labels": tLabels,
		"input_boxes":  tBoxes,
	}
	for k, v := range ctx.imageEmbeddings {
		inputValues[k] = v
	}

	// decoder inference
	outputs, err := ctx.engine.decoderSession.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("decoder inference failed: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Destroy()
		}
	}()

	// get best mask
	rawScores, err := ort.GetTensorData[float32](outputs["iou_scores"])
	if err != nil {
		return nil, fmt.Errorf("failed to get decoder output data: %w", err)
	}
	rawMasks, err := ort.GetTensorData[float32](outputs["pred_masks"])
	if err != nil {
		return nil, fmt.Errorf("failed to get decoder output data: %w", err)
	}

	bestIdx := 0
	bestScore := float32(-100.0)

	for i := 0; i < len(rawScores); i++ {
		if rawScores[i] > bestScore {
			bestScore = rawScores[i]
			bestIdx = i
		}
	}

	// extract the corresponding mask logits (256x256)
	pixelsPerMask := 256 * 256
	start := bestIdx * pixelsPerMask
	end := start + pixelsPerMask
	if end > len(rawMasks) {
		return nil, fmt.Errorf("mask index out of range: start=%d end=%d len=%d", start, end, len(rawMasks))
	}
	bestMaskLogits := rawMasks[start:end]

	validMaskW := int(float32(ctx.newW) / 4.0)
	validMaskH := int(float32(ctx.newH) / 4.0)

	finalMask := upscaleMaskLogits(bestMaskLogits, 256, validMaskW, validMaskH, ctx.origW, ctx.origH)

	return &Result{
		Mask:   finalMask,
		Score:  bestScore,
		Width:  ctx.origW,
		Height: ctx.origH,
	}, nil
}

// Decode decodes the mask and returns an image
func (ctx *ImageContext) Decode(points []Point) (image.Image, float32, error) {
	result, err := ctx.DecodeRaw(points)
	if err != nil {
		return nil, 0, err
	}

	img := image.NewGray(image.Rect(0, 0, result.Width, result.Height))
	copy(img.Pix, result.Mask)
	return img, result.Score, nil
}
