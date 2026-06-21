package efficientsam3

import (
	"path/filepath"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision/manifest"
)

type Label int

const (
	LabelBackground Label = 0
	LabelForeground Label = 1
	LabelNegBox     Label = 0
	LabelPosBox     Label = 1
)

const (
	inputSize     = 1008
	maskThreshold = 0.0
	textSeqLen    = 16 // EfficientSAM3 LiteText default context length
	padTokenId    = 0  // MobileCLIP padding token
)

type Point struct {
	X, Y  float32
	Label Label
}

// Config EfficientSAM3 模型配置
type Config struct {
	// ModelPath 模型包目录路径 (包含 manifest.json 和 ONNX 子模型)
	ModelPath          string
	OnnxRuntimeLibPath string

	UseCuda           bool
	NumThreads        int
	EnableCpuMemArena bool
	ApiVersion        ort.ApiVersion
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		ModelPath:          modelpath.ModelPath("efficientsam3"),
	}
}

// resolveSubModelPaths 从 manifest.json 解析子模型路径
func (c *Config) resolveSubModelPaths() (visionPath, textPath, decoderPath string, err error) {
	mf, loadErr := manifest.Load(c.ModelPath)
	if loadErr == nil {
		// 从 manifest.json 解析
		visionPath = mf.SubModelPath(c.ModelPath, "vision_encoder")
		textPath = mf.SubModelPath(c.ModelPath, "text_encoder")
		decoderPath = mf.SubModelPath(c.ModelPath, "decoder")
	} else {
		// 回退：使用目录内默认文件名
		visionPath = filepath.Join(c.ModelPath, "efficientsam3_image_encoder.onnx")
		textPath = filepath.Join(c.ModelPath, "efficientsam3_language_encoder.onnx")
		decoderPath = filepath.Join(c.ModelPath, "efficientsam3_decoder.onnx")
	}
	return
}
