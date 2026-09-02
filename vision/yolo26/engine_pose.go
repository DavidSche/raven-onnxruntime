package yolo26

import (
	"fmt"
	"image"
	"sync/atomic"
	"time"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
)

// PoseEngine YOLO26-pose Engine
type PoseEngine struct {
	session  *ort.Session
	config   Config
	runCount uint64
}

// NewPoseEngine initializes the pose estimation engine
func NewPoseEngine(cfg Config) (*PoseEngine, error) {
	ortlog.Infow("creating YOLO26 pose engine",
		"modelPath", cfg.ModelPath,
		"inputSize", cfg.InputSize,
		"confThreshold", cfg.ConfThreshold,
		"numThreads", cfg.NumThreads,
		"useCuda", cfg.UseCuda,
		"numKeyPoints", cfg.NumKeyPoints)

	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}

	// initialize ONNX
	if err := oc.New(); err != nil {
		ortlog.Errorw("failed to initialize ONNX config", "error", err)
		return nil, err
	}

	// create session
	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	oc.Destroy()
	if err != nil {
		ortlog.Errorw("failed to create ONNX session", "modelPath", cfg.ModelPath, "error", err)
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	// 自动适配模型真实输入尺寸（与 NewOBBEngine 相同的 GetInputShape 自适应路径）：
	// 静态输入模型（如 yolo26x-pose.onnx 固定 640×640 输入）必须按模型输入形状预处理，
	// 否则会话运行会因张量形状不匹配而失败；动态输入模型（n/s/m/l 官方导出 imgsz=1280）
	// 接受任意尺寸，沿用配置的 InputSize。
	if size := staticSquareInputSize(session); size > 0 && size != cfg.InputSize {
		ortlog.Infow("YOLO26 pose engine auto-adapting input size to static model shape",
			"modelPath", cfg.ModelPath, "configured", cfg.InputSize, "actual", size)
		cfg.InputSize = size
	}

	ortlog.Infow("YOLO26 pose engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)

	return &PoseEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *PoseEngine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying YOLO26 pose engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict executes pose estimation
func (e *PoseEngine) Predict(img image.Image) ([]PoseResult, error) {
	results, err := e.PredictBatch([]image.Image{img})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return results[0], nil
}

// PredictBatch executes batch pose inference
func (e *PoseEngine) PredictBatch(imgs []image.Image) ([][]PoseResult, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	startedAt := time.Now()

	// preprocess
	preprocessStart := time.Now()
	inputTensor, paramsList, err := preprocessBatch(imgs, e.config.InputSize, e.session, e.config.PreprocessConfig)
	if err != nil {
		return nil, fmt.Errorf("preprocess failed: %w", err)
	}
	defer inputTensor.Destroy()
	preprocessElapsed := time.Since(preprocessStart)

	// get actual input name (compatible with different models)
	if len(e.session.InputNames) == 0 {
		return nil, fmt.Errorf("model has no input")
	}
	inputName := e.session.InputNames[0]

	// inference
	inputValues := map[string]*ort.Value{
		inputName: inputTensor,
	}
	runStart := time.Now()
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer ort.DestroyValues(outputValues)
	runElapsed := time.Since(runStart)

	// get first output (compatible with different output names)
	if len(e.session.OutputNames) == 0 {
		return nil, fmt.Errorf("model has no output")
	}
	outputName := e.session.OutputNames[0]
	outputValue, ok := outputValues[outputName]
	if !ok || outputValue == nil {
		return nil, fmt.Errorf("output %q does not exist", outputName)
	}
	defer outputValue.Destroy()

	// Output Shaper: [1, 300, 57]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}

	shape, err := outputValue.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get output shape: %w", err)
	}

	ortlog.Debugw("pose model output",
		"shape", shape,
		"dataLen", len(data))

	if len(shape) != 3 {
		return nil, fmt.Errorf("unsupported output shape: %v", shape)
	}

	batchSize := int(shape[0])
	numObjects := int(shape[1])
	attributes := int(shape[2])
	if batchSize != len(imgs) {
		return nil, fmt.Errorf("batch output mismatch: got %d want %d", batchSize, len(imgs))
	}

	stride := numObjects * attributes
	postprocessStart := time.Now()
	results := make([][]PoseResult, batchSize)
	for i := 0; i < batchSize; i++ {
		start := i * stride
		end := start + stride

		sampleShape := []int64{1, int64(numObjects), int64(attributes)}
		sampleData := data[start:end]

		sampleResults, err := e.postprocess(sampleData, sampleShape, paramsList[i])
		if err != nil {
			return nil, err
		}
		results[i] = sampleResults
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(batchSize, preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return results, nil
}

func (e *PoseEngine) logPredictTimings(batchSize int, preprocessElapsed, runElapsed, postprocessElapsed, totalElapsed time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("yolo26 pose timings",
		"modelPath", e.config.ModelPath,
		"batchSize", batchSize,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}

// postprocess performs post-processing on the output
func (e *PoseEngine) postprocess(data []float32, shape []int64, params imageParams) ([]PoseResult, error) {
	ortlog.Debugw("postprocess starting",
		"shape", shape,
		"dataLen", len(data),
		"confThreshold", e.config.ConfThreshold)

	if len(shape) < 2 {
		return nil, fmt.Errorf("invalid output shape: %v", shape)
	}

	results := make([]PoseResult, 0)

	if len(shape) == 3 && shape[0] == 1 {
		numChannels := int(shape[1])
		numAnchors := int(shape[2])

		if numChannels > 50 && numAnchors > 1000 {
			ortlog.Debugw("using channel-first format",
				"numChannels", numChannels,
				"numAnchors", numAnchors)

			return e.postprocessChannelFirst(data, numChannels, numAnchors, params)
		}
	}

	var numObjects int
	var attributes int

	if len(shape) == 3 {
		numObjects = int(shape[1])
		attributes = int(shape[2])
	} else if len(shape) == 2 {
		numObjects = int(shape[0])
		attributes = int(shape[1])
	} else {
		return nil, fmt.Errorf("unsupported output shape: %v", shape)
	}

	ortlog.Debugw("postprocess config",
		"numObjects", numObjects,
		"attributes", attributes,
		"expectedKpts", attributes-6)

	allMaxScore := float32(0.0)
	allMaxIdx := -1
	for i := 0; i < numObjects && i*attributes < len(data); i++ {
		offset := i * attributes
		if offset+4 < len(data) {
			score := data[offset+4]
			if score > allMaxScore {
				allMaxScore = score
				allMaxIdx = i
			}
		}
	}
	ortlog.Debugw("max score in all objects", "allMaxScore", allMaxScore, "allMaxIdx", allMaxIdx, "confThreshold", e.config.ConfThreshold)

	for i := 0; i < numObjects && i*attributes < len(data); i++ {
		offset := i * attributes

		if offset+attributes > len(data) {
			break
		}

		score := data[offset+4]
		if score < e.config.ConfThreshold {
			continue
		}

		classID := int(data[offset+5])

		x1 := data[offset+0]
		y1 := data[offset+1]
		x2 := data[offset+2]
		y2 := data[offset+3]

		origX1 := max(0, int((x1-float32(params.padX))/params.scale))
		origY1 := max(0, int((y1-float32(params.padY))/params.scale))
		origX2 := min(params.origW, int((x2-float32(params.padX))/params.scale))
		origY2 := min(params.origH, int((y2-float32(params.padY))/params.scale))

		kptLen := attributes - 6
		if offset+6+kptLen > len(data) {
			kptLen = len(data) - offset - 6
		}
		rawKpts := data[offset+6 : offset+6+kptLen]
		kpts := e.decodeKeyPoints(rawKpts, params)

		results = append(results, PoseResult{
			ClassID:   classID,
			Score:     score,
			Box:       image.Rect(origX1, origY1, origX2, origY2),
			KeyPoints: kpts,
		})
	}

	ortlog.Debugw("postprocess completed",
		"resultCount", len(results),
		"allMaxScore", allMaxScore,
		"allMaxIdx", allMaxIdx)

	return results, nil
}

type poseCandidate struct {
	box          [4]float32
	origBox      image.Rectangle
	score        float32
	classID      int
	rawKeyPoints []float32
}

func (c poseCandidate) GetBox() [4]float32 { return c.box }
func (c poseCandidate) GetScore() float32  { return c.score }

// postprocessChannelFirst handles channel-first format [1, channels, anchors]
func (e *PoseEngine) postprocessChannelFirst(data []float32, channels, anchors int, params imageParams) ([]PoseResult, error) {
	expectedChannels := 4 + e.config.NumClasses + e.config.NumKeyPoints*3
	ortlog.Debugw("channel-first format check",
		"channels", channels,
		"expectedChannels", expectedChannels,
		"numClasses", e.config.NumClasses,
		"numKeyPoints", e.config.NumKeyPoints)

	kptStartIdx := 4 + e.config.NumClasses
	numKptValues := e.config.NumKeyPoints * 3

	candidates := make([]poseCandidate, 0)

	for i := 0; i < anchors; i++ {
		maxScore := float32(0.0)
		classID := -1
		for c := 0; c < e.config.NumClasses; c++ {
			score := data[(4+c)*anchors+i]
			if score > maxScore {
				maxScore = score
				classID = c
			}
		}
		if maxScore < e.config.ConfThreshold {
			continue
		}

		cx := data[0*anchors+i]
		cy := data[1*anchors+i]
		w := data[2*anchors+i]
		h := data[3*anchors+i]

		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2
		origX1 := max(0, int((x1-float32(params.padX))/params.scale))
		origY1 := max(0, int((y1-float32(params.padY))/params.scale))
		origX2 := min(params.origW, int((x2-float32(params.padX))/params.scale))
		origY2 := min(params.origH, int((y2-float32(params.padY))/params.scale))

		rawKpts := make([]float32, numKptValues)
		for k := 0; k < numKptValues; k++ {
			rawKpts[k] = data[(kptStartIdx+k)*anchors+i]
		}

		candidates = append(candidates, poseCandidate{
			box:          [4]float32{x1, y1, x2, y2},
			origBox:      image.Rect(origX1, origY1, origX2, origY2),
			score:        maxScore,
			classID:      classID,
			rawKeyPoints: rawKpts,
		})
	}

	ortlog.Debugw("candidates before NMS", "count", len(candidates))

	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]PoseResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]
		kpts := e.decodeKeyPoints(cand.rawKeyPoints, params)

		results = append(results, PoseResult{
			ClassID:   cand.classID,
			Score:     cand.score,
			Box:       cand.origBox,
			KeyPoints: kpts,
		})
	}

	ortlog.Debugw("postprocessChannelFirst completed", "resultCount", len(results))

	return results, nil
}

// decodeKeyPoints decodes keypoint coordinates
func (e *PoseEngine) decodeKeyPoints(raw []float32, params imageParams) []KeyPoint {
	kpts := make([]KeyPoint, e.config.NumKeyPoints)

	for i := 0; i < e.config.NumKeyPoints; i++ {
		idx := i * 3
		x := raw[idx]
		y := raw[idx+1]
		conf := raw[idx+2]

		// map coordinates back to original image
		origX := min(max(0, int((x-float32(params.padX))/params.scale)), params.origW)
		origY := min(max(0, int((y-float32(params.padY))/params.scale)), params.origH)

		kpts[i] = KeyPoint{
			X:     origX,
			Y:     origY,
			Score: conf,
		}
	}
	return kpts
}
