package groundingdino

import (
	"fmt"
	"image"
	"math"
	"sort"
	"sync/atomic"
	"time"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/DavidSche/raven-onnxruntime/vision/manifest"
	"github.com/DavidSche/raven-onnxruntime/vision/tokenizer"
	"github.com/up-zero/gotool/convertutil"
)

// DetEngine GroundingDINO 检测引擎
// 使用三个 ONNX 子模型实现开放词汇目标检测：
//   - image_encoder: 提取多尺度图像特征
//   - text_encoder:  提取文本嵌入
//   - detector:      跨模态融合 + 检测头
type DetEngine struct {
	imageEncoder *ort.Session
	textEncoder  *ort.Session
	detector     *ort.Session
	config       Config
	metadata     *ModelMetadata
	runCount     uint64
}

// ModelMetadata ONNX 模型元数据
type ModelMetadata struct {
	HiddenDim        int `json:"hidden_dim"`
	NumQueries       int `json:"num_queries"`
	NumFeatureLevels int `json:"num_feature_levels"`
	InputSize        int `json:"input_size"`
	MaxTextLen       int `json:"max_text_len"`
}

// NewDetEngine 创建 GroundingDINO 检测引擎
func NewDetEngine(cfg Config) (*DetEngine, error) {
	// 从 manifest.json 解析子模型路径
	imageEncoderPath, textEncoderPath, detectorPath, err := cfg.resolveSubModelPaths()
	if err != nil {
		return nil, fmt.Errorf("failed to resolve sub-model paths: %w", err)
	}

	ortlog.Infow("creating GroundingDINO detection engine",
		"modelPath", cfg.ModelPath,
		"imageEncoderPath", imageEncoderPath,
		"textEncoderPath", textEncoderPath,
		"detectorPath", detectorPath,
		"inputSize", cfg.InputSize,
		"confThreshold", cfg.ConfThreshold,
		"maxTextLen", cfg.MaxTextLen,
		"numThreads", cfg.NumThreads,
		"useCuda", cfg.UseCuda)

	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}

	if err := oc.New(); err != nil {
		ortlog.Errorw("failed to initialize ONNX config", "error", err)
		return nil, fmt.Errorf("initialization failed: %w", err)
	}

	// 创建三个子模型会话
	imageEncoder, err := oc.OnnxEngine.NewSession(imageEncoderPath, oc.SessionOptions)
	if err != nil {
		oc.Destroy()
		return nil, fmt.Errorf("failed to create image encoder session: %w", err)
	}

	textEncoder, err := oc.OnnxEngine.NewSession(textEncoderPath, oc.SessionOptions)
	if err != nil {
		imageEncoder.Destroy()
		oc.Destroy()
		return nil, fmt.Errorf("failed to create text encoder session: %w", err)
	}

	detector, err := oc.OnnxEngine.NewSession(detectorPath, oc.SessionOptions)
	if err != nil {
		imageEncoder.Destroy()
		textEncoder.Destroy()
		oc.Destroy()
		return nil, fmt.Errorf("failed to create detector session: %w", err)
	}

	ortlog.Infow("GroundingDINO sessions created",
		"imageEncoderInputs", imageEncoder.InputNames,
		"imageEncoderOutputs", imageEncoder.OutputNames,
		"textEncoderInputs", textEncoder.InputNames,
		"textEncoderOutputs", textEncoder.OutputNames,
		"detectorInputs", detector.InputNames,
		"detectorOutputs", detector.OutputNames)

	// 从 manifest.json 加载元数据
	metadata := loadMetadataFromManifest(cfg)

	// 如果元数据中没有 inputSize，使用配置值
	if metadata.InputSize == 0 {
		metadata.InputSize = cfg.InputSize
	}
	if metadata.MaxTextLen == 0 {
		metadata.MaxTextLen = cfg.MaxTextLen
	}
	if metadata.NumQueries == 0 {
		metadata.NumQueries = 900 // GroundingDINO-SwinB 默认
	}
	if metadata.NumFeatureLevels == 0 {
		metadata.NumFeatureLevels = 4
	}
	if metadata.HiddenDim == 0 {
		metadata.HiddenDim = 256
	}

	return &DetEngine{
		imageEncoder: imageEncoder,
		textEncoder:  textEncoder,
		detector:     detector,
		config:       cfg,
		metadata:     metadata,
	}, nil
}

// Destroy 销毁引擎，释放资源
func (e *DetEngine) Destroy() {
	if e.imageEncoder != nil {
		ortlog.Infow("destroying GroundingDINO image encoder", "modelPath", e.config.ModelPath)
		e.imageEncoder.Destroy()
	}
	if e.textEncoder != nil {
		ortlog.Infow("destroying GroundingDINO text encoder", "modelPath", e.config.ModelPath)
		e.textEncoder.Destroy()
	}
	if e.detector != nil {
		ortlog.Infow("destroying GroundingDINO detector", "modelPath", e.config.ModelPath)
		e.detector.Destroy()
	}
}

// Predict 执行开放词汇目标检测
func (e *DetEngine) Predict(img image.Image, captions []string) ([]DetResult, error) {
	startedAt := time.Now()

	// 1. 图像编码
	preprocessStart := time.Now()
	inputTensor, params, err := preprocessImage(e.imageEncoder, img, e.config.InputSize, e.config.PreprocessConfig)
	if err != nil {
		return nil, fmt.Errorf("image preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()
	preprocessElapsed := time.Since(preprocessStart)

	imageEncodeStart := time.Now()
	imageFeatures, err := e.runImageEncoder(inputTensor)
	if err != nil {
		return nil, fmt.Errorf("image encoding failed: %w", err)
	}
	defer func() {
		for _, v := range imageFeatures {
			v.Destroy()
		}
	}()
	imageEncodeElapsed := time.Since(imageEncodeStart)

	// 2. 文本编码
	textEncodeStart := time.Now()
	textFeatures, err := e.runTextEncoder(captions)
	if err != nil {
		return nil, fmt.Errorf("text encoding failed: %w", err)
	}
	defer func() {
		for _, v := range textFeatures {
			v.Destroy()
		}
	}()
	textEncodeElapsed := time.Since(textEncodeStart)

	// 3. 检测
	detectStart := time.Now()
	outputValues, err := e.runDetector(imageFeatures, textFeatures)
	if err != nil {
		return nil, fmt.Errorf("detection failed: %w", err)
	}
	defer func() {
		for _, v := range outputValues {
			v.Destroy()
		}
	}()
	detectElapsed := time.Since(detectStart)

	// 4. 后处理
	postprocessStart := time.Now()
	results, err := e.postprocess(outputValues, captions, params)
	if err != nil {
		return nil, fmt.Errorf("postprocess failed: %w", err)
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(preprocessElapsed, imageEncodeElapsed, textEncodeElapsed, detectElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

// runImageEncoder 运行图像编码器
func (e *DetEngine) runImageEncoder(inputTensor *ort.Value) (map[string]*ort.Value, error) {
	if len(e.imageEncoder.InputNames) == 0 {
		return nil, fmt.Errorf("image encoder has no input")
	}
	inputName := e.imageEncoder.InputNames[0]

	inputValues := map[string]*ort.Value{
		inputName: inputTensor,
	}

	outputValues, err := e.imageEncoder.Run(inputValues)
	if err != nil {
		return nil, err
	}

	return outputValues, nil
}

// runTextEncoder 运行文本编码器
func (e *DetEngine) runTextEncoder(captions []string) (map[string]*ort.Value, error) {
	bundle, err := tokenizer.Acquire(e.textEncoder, captions, e.config.MaxTextLen)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}
	defer bundle.Destroy()
	if bundle.Source == tokenizer.SourcePlaceholder {
		ortlog.Warnw("groundingdino using placeholder tokenizer fallback",
			"modelPath", e.config.ModelPath,
			"maxTextLen", e.config.MaxTextLen)
	}

	inputValues := map[string]*ort.Value{
		"input_ids":                 bundle.InputIds,
		"attention_mask":            bundle.AttentionMask,
		"token_type_ids":            bundle.TokenTypeIds,
		"text_self_attention_masks": bundle.TextSelfAttnMasks,
		"position_ids":              bundle.PositionIds,
	}

	outputValues, err := e.textEncoder.Run(inputValues)
	if err != nil {
		return nil, err
	}

	return outputValues, nil
}

// runDetector 运行检测器
func (e *DetEngine) runDetector(imageFeatures map[string]*ort.Value, textFeatures map[string]*ort.Value) (map[string]*ort.Value, error) {
	inputValues := make(map[string]*ort.Value)

	// 图像特征
	for _, name := range e.detector.InputNames {
		switch name {
		case "encoded_text":
			if v, ok := textFeatures[name]; ok {
				inputValues[name] = v
			}
		case "text_token_mask":
			if v, ok := textFeatures[name]; ok {
				inputValues[name] = v
			}
		case "text_self_attention_masks":
			if v, ok := textFeatures[name]; ok {
				inputValues[name] = v
			}
		case "position_ids":
			if v, ok := textFeatures[name]; ok {
				inputValues[name] = v
			}
		default:
			// src_0, pos_0, src_1, pos_1, ...
			if v, ok := imageFeatures[name]; ok {
				inputValues[name] = v
			}
		}
	}

	return e.detector.Run(inputValues)
}

// postprocess 后处理：将模型输出转换为检测结果
func (e *DetEngine) postprocess(outputValues map[string]*ort.Value, captions []string, params imageParams) ([]DetResult, error) {
	// 获取 pred_logits 和 pred_boxes
	var logitsOut, boxesOut *ort.Value
	for _, name := range e.detector.OutputNames {
		switch name {
		case "pred_logits":
			logitsOut = outputValues[name]
		case "pred_boxes":
			boxesOut = outputValues[name]
		}
	}

	if logitsOut == nil || boxesOut == nil {
		return nil, fmt.Errorf("detector output missing pred_logits or pred_boxes")
	}

	logitsData, err := ort.GetTensorData[float32](logitsOut)
	if err != nil {
		return nil, fmt.Errorf("failed to get logits data: %w", err)
	}
	logitsShape, err := logitsOut.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get logits shape: %w", err)
	}

	boxesData, err := ort.GetTensorData[float32](boxesOut)
	if err != nil {
		return nil, fmt.Errorf("failed to get boxes data: %w", err)
	}
	boxesShape, err := boxesOut.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get boxes shape: %w", err)
	}

	if len(logitsShape) != 3 || len(boxesShape) != 3 {
		return nil, fmt.Errorf("unexpected output shapes: logits=%v boxes=%v", logitsShape, boxesShape)
	}

	numQueries := int(logitsShape[1])

	type candidate struct {
		score  float32
		cx, cy float32
		w, h   float32
	}

	candidates := make([]candidate, 0, numQueries)
	for i := 0; i < numQueries; i++ {
		// pred_logits shape: [1, num_queries, 1]
		score := sigmoid(logitsData[i])
		if score < e.config.ConfThreshold {
			continue
		}

		boxOffset := i * 4
		candidates = append(candidates, candidate{
			score: score,
			cx:    boxesData[boxOffset+0],
			cy:    boxesData[boxOffset+1],
			w:     boxesData[boxOffset+2],
			h:     boxesData[boxOffset+3],
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > e.config.MaxDetections {
		candidates = candidates[:e.config.MaxDetections]
	}

	// 确定类别名称
	className := "object"
	if len(captions) > 0 {
		className = captions[0]
	}

	results := make([]DetResult, 0, len(candidates))
	for _, cand := range candidates {
		box := boxCxcywhToXyxy(cand.cx, cand.cy, cand.w, cand.h, params.origW, params.origH)
		results = append(results, DetResult{
			ClassID:   0,
			ClassName: className,
			Score:     cand.score,
			Box:       box,
		})
	}

	return results, nil
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// boxCxcywhToXyxy 将 cxcywh 归一化坐标转换为像素坐标 xyxy
func boxCxcywhToXyxy(cx, cy, w, h float32, origW, origH int) image.Rectangle {
	x1 := cx - w/2
	y1 := cy - h/2
	x2 := cx + w/2
	y2 := cy + h/2

	origX1 := max(0, int(x1*float32(origW)))
	origY1 := max(0, int(y1*float32(origH)))
	origX2 := min(origW, int(x2*float32(origW)))
	origY2 := min(origH, int(y2*float32(origH)))

	return image.Rect(origX1, origY1, origX2, origY2)
}

func (e *DetEngine) logPredictTimings(preprocess, imageEncode, textEncode, detect, postprocess, total time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("groundingdino timings",
		"modelPath", e.config.ModelPath,
		"preprocess", preprocess.String(),
		"imageEncode", imageEncode.String(),
		"textEncode", textEncode.String(),
		"detect", detect.String(),
		"postprocess", postprocess.String(),
		"total", total.String(),
		"count", count)
}

// loadMetadataFromManifest 从 manifest.json 加载模型元数据
func loadMetadataFromManifest(cfg Config) *ModelMetadata {
	meta := &ModelMetadata{}

	mf, err := manifest.Load(cfg.ModelPath)
	if err != nil {
		ortlog.Infow("no manifest.json found, using config defaults", "modelPath", cfg.ModelPath, "error", err)
		return meta
	}

	meta.HiddenDim = mf.ParamInt("hidden_dim", 0)
	meta.NumQueries = mf.ParamInt("num_queries", 0)
	meta.NumFeatureLevels = mf.ParamInt("num_feature_levels", 0)
	meta.InputSize = mf.InputSizeAt(0, 0)
	meta.MaxTextLen = mf.ParamInt("max_text_len", 0)

	ortlog.Infow("loaded GroundingDINO metadata from manifest", "modelPath", cfg.ModelPath, "metadata", meta)
	return meta
}
