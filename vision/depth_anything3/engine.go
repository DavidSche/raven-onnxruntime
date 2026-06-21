package depth_anything3

import (
	"fmt"
	"image"
	"sync/atomic"
	"time"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/ort/ortlog"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/imageutil"
)

// Engine Depth-Anything-3 depth estimation engine
type Engine struct {
	session  *ort.Session
	config   Config
	runCount uint64
}

// NewEngine initializes the Depth-Anything-3 engine
func NewEngine(cfg Config) (*Engine, error) {
	ortlog.Infow("creating Depth-Anything-3 engine",
		"modelPath", cfg.ModelPath,
		"inputHeight", cfg.InputHeight,
		"inputWidth", cfg.InputWidth,
		"numThreads", cfg.NumThreads,
		"useCuda", cfg.UseCuda)

	// Validate input dimensions
	if cfg.InputHeight%PatchSize != 0 || cfg.InputWidth%PatchSize != 0 {
		return nil, fmt.Errorf("input dimensions (%d, %d) must be multiples of %d",
			cfg.InputHeight, cfg.InputWidth, PatchSize)
	}

	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("failed to copy config properties: %w", err)
	}

	if err := oc.New(); err != nil {
		ortlog.Errorw("failed to initialize ONNX config", "error", err)
		return nil, fmt.Errorf("initialization failed: %w", err)
	}

	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	oc.Destroy()
	if err != nil {
		ortlog.Errorw("failed to create ONNX session", "modelPath", cfg.ModelPath, "error", err)
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	ortlog.Infow("Depth-Anything-3 engine created successfully",
		"modelPath", cfg.ModelPath,
		"inputs", session.InputNames,
		"outputs", session.OutputNames)

	return &Engine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy releases all resources
func (e *Engine) Destroy() {
	if e.session != nil {
		ortlog.Infow("destroying Depth-Anything-3 engine", "modelPath", e.config.ModelPath)
		e.session.Destroy()
	}
}

// Predict performs depth estimation inference on a single image
func (e *Engine) Predict(img image.Image) (*DepthResult, error) {
	startedAt := time.Now()

	// Preprocess
	preprocessStart := time.Now()
	inputTensor, params, err := e.preprocess(img, e.config.InputHeight, e.config.InputWidth)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}
	defer inputTensor.Destroy()
	preprocessElapsed := time.Since(preprocessStart)

	// Get input name
	if len(e.session.InputNames) == 0 {
		return nil, fmt.Errorf("model has no inputs")
	}
	inputName := e.session.InputNames[0]

	// Inference
	inputValues := map[string]*ort.Value{
		inputName: inputTensor,
	}
	runStart := time.Now()
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	runElapsed := time.Since(runStart)

	// Postprocess
	postprocessStart := time.Now()
	result, err := e.postprocess(outputValues, params)
	if err != nil {
		return nil, fmt.Errorf("postprocessing failed: %w", err)
	}
	postprocessElapsed := time.Since(postprocessStart)

	e.logPredictTimings(preprocessElapsed, runElapsed, postprocessElapsed, time.Since(startedAt))

	return result, nil
}

// postprocess parses ONNX output into DepthResult
func (e *Engine) postprocess(outputValues map[string]*ort.Value, params imageParams) (*DepthResult, error) {
	// Get depth output
	depthOutput, ok := outputValues["depth"]
	if !ok || depthOutput == nil {
		// Fallback to first output
		if len(e.session.OutputNames) > 0 {
			depthOutput = outputValues[e.session.OutputNames[0]]
		}
	}
	if depthOutput == nil {
		return nil, fmt.Errorf("model has no depth output")
	}

	depthData, err := ort.GetTensorData[float32](depthOutput)
	if err != nil {
		depthOutput.Destroy()
		return nil, fmt.Errorf("failed to get depth data: %w", err)
	}
	depthShape, _ := depthOutput.GetShape()
	depthOutput.Destroy()

	// Get depth_conf output
	confOutput, ok := outputValues["depth_conf"]
	if !ok || confOutput == nil {
		// Fallback to second output
		if len(e.session.OutputNames) > 1 {
			confOutput = outputValues[e.session.OutputNames[1]]
		}
	}

	var confData []float32
	if confOutput != nil {
		confData, err = ort.GetTensorData[float32](confOutput)
		confOutput.Destroy()
		if err != nil {
			return nil, fmt.Errorf("failed to get depth_conf data: %w", err)
		}
	}

	// Destroy remaining outputs
	for name, v := range outputValues {
		if v != nil && v != depthOutput && v != confOutput {
			v.Destroy()
			_ = name
		}
	}

	// Parse output shape
	// depth: (B, 1, H, W) -> extract (H, W)
	var outH, outW int
	if len(depthShape) == 4 {
		outH = int(depthShape[2])
		outW = int(depthShape[3])
	} else if len(depthShape) == 3 {
		outH = int(depthShape[1])
		outW = int(depthShape[2])
	} else {
		return nil, fmt.Errorf("unexpected depth output shape: %v", depthShape)
	}

	// Extract first depth map (batch=0)
	pixelCount := outH * outW
	depthMap := make([]float32, pixelCount)
	copy(depthMap, depthData[:pixelCount])

	// Extract confidence map
	confMap := make([]float32, pixelCount)
	if len(confData) >= pixelCount {
		copy(confMap, confData[:pixelCount])
	}

	// Resize back to original image size
	if outH != params.origH || outW != params.origW {
		depthMap = resizeDepthMap(depthMap, outW, outH, params.origW, params.origH)
		confMap = resizeDepthMap(confMap, outW, outH, params.origW, params.origH)
	}

	return &DepthResult{
		Depth:      depthMap,
		Confidence: confMap,
		Width:      params.origW,
		Height:     params.origH,
	}, nil
}

func (e *Engine) logPredictTimings(preprocessElapsed, runElapsed, postprocessElapsed, totalElapsed time.Duration) {
	count := atomic.AddUint64(&e.runCount, 1)
	if count%60 != 0 {
		return
	}

	ortlog.Infow("depth_anything3 timings",
		"modelPath", e.config.ModelPath,
		"preprocess", preprocessElapsed.String(),
		"run", runElapsed.String(),
		"postprocess", postprocessElapsed.String(),
		"total", totalElapsed.String(),
		"count", count)
}

// preprocess converts the input image to an ONNX input tensor
// DA3 input format: (B, 1, 3, H, W) — B=batch, 1=single frame, 3=RGB, H/W=dimensions
func (e *Engine) preprocess(img image.Image, inputH, inputW int) (*ort.Value, imageParams, error) {
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	// upper_bound_resize: scale by long edge, ensuring it does not exceed inputH/inputW
	scale := float32(inputH) / float32(max(origW, origH))
	newW := int(float32(origW) * scale)
	newH := int(float32(origH) * scale)

	// Ensure dimensions are multiples of PatchSize
	newW = (newW / PatchSize) * PatchSize
	newH = (newH / PatchSize) * PatchSize

	if newW == 0 {
		newW = PatchSize
	}
	if newH == 0 {
		newH = PatchSize
	}

	params := imageParams{
		origW: origW,
		origH: origH,
		newW:  newW,
		newH:  newH,
	}

	// Resize image
	var resized image.Image
	if newW == origW && newH == origH {
		resized = img
	} else {
		resized = imageutil.Resize(img, newW, newH)
	}

	// Normalize and convert to CHW format
	// DA3 input: (B, 1, 3, H, W), here B=1, N=1
	data := make([]float32, 1*1*3*newH*newW)
	normalizeToCHW(data, resized, newW, newH)

	// Create tensor (1, 1, 3, H, W)
	tensor, err := e.session.NewTensor([]int64{1, 1, 3, int64(newH), int64(newW)}, data)
	if err != nil {
		return nil, imageParams{}, fmt.Errorf("failed to create tensor: %w", err)
	}

	return tensor, params, nil
}

// normalizeToCHW normalizes the image and fills data in CHW format
func normalizeToCHW(data []float32, img image.Image, w, h int) {
	bounds := img.Bounds()
	planeSize := w * h

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			rf := float32(r) / 65535.0
			gf := float32(g) / 65535.0
			bf := float32(b) / 65535.0

			idx := y*w + x
			data[idx] = (rf - MeanR) / StdR
			data[planeSize+idx] = (gf - MeanG) / StdG
			data[2*planeSize+idx] = (bf - MeanB) / StdB
		}
	}
}

// resizeDepthMap resizes the depth map to the target dimensions (nearest-neighbor interpolation)
func resizeDepthMap(data []float32, srcW, srcH, dstW, dstH int) []float32 {
	output := make([]float32, dstW*dstH)
	xRatio := float32(srcW) / float32(dstW)
	yRatio := float32(srcH) / float32(dstH)

	for y := 0; y < dstH; y++ {
		srcY := int(float32(y) * yRatio)
		if srcY >= srcH {
			srcY = srcH - 1
		}
		for x := 0; x < dstW; x++ {
			srcX := int(float32(x) * xRatio)
			if srcX >= srcW {
				srcX = srcW - 1
			}
			output[y*dstW+x] = data[srcY*srcW+srcX]
		}
	}

	return output
}
