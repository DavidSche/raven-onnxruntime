package groundingdino

import (
	"image"
	"path/filepath"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision"
	"github.com/DavidSche/raven-onnxruntime/vision/manifest"
)

// Config GroundingDINO 模型配置
type Config struct {
	// ModelPath 模型包目录路径 (包含 manifest.json 和 ONNX 子模型)
	// raven-onnxruntime 从 manifest.json 自动发现子模型路径
	ModelPath          string
	OnnxRuntimeLibPath string

	// 推理参数
	ConfThreshold float32
	InputSize     int
	MaxTextLen    int
	MaxDetections int

	// 运行时参数
	UseCuda           bool
	NumThreads        int
	EnableCpuMemArena bool
	PreprocessConfig  vision.PreprocessConfig
	ApiVersion        ort.ApiVersion
}

// DefaultConfig 返回默认配置
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		PreprocessConfig:   vision.DefaultImageNetPreprocessConfig(),
		ModelPath:          modelpath.ModelPath("groundingdino"),
		ConfThreshold:      0.35,
		InputSize:          800,
		MaxTextLen:         256,
		MaxDetections:      300,
	}
}

// resolveSubModelPaths 从 manifest.json 解析子模型路径
// 如果 manifest.json 不存在，回退到目录内默认文件名
func (c *Config) resolveSubModelPaths() (imageEncoderPath, textEncoderPath, detectorPath string, err error) {
	mf, loadErr := manifest.Load(c.ModelPath)
	if loadErr == nil {
		// 从 manifest.json 解析
		imageEncoderPath = mf.SubModelPath(c.ModelPath, "image_encoder")
		textEncoderPath = mf.SubModelPath(c.ModelPath, "text_encoder")
		detectorPath = mf.SubModelPath(c.ModelPath, "detector")

		// 从 manifest 补充参数（如果配置为零值）
		if c.InputSize == 0 {
			c.InputSize = mf.InputSizeAt(0, 800)
		}
		if c.MaxTextLen == 0 {
			c.MaxTextLen = mf.ParamInt("max_text_len", 256)
		}
	} else {
		// 回退：使用目录内默认文件名
		imageEncoderPath = filepath.Join(c.ModelPath, "groundingdino_image_encoder.onnx")
		textEncoderPath = filepath.Join(c.ModelPath, "groundingdino_text_encoder.onnx")
		detectorPath = filepath.Join(c.ModelPath, "groundingdino_detector.onnx")
	}
	return
}

// DetResult 检测结果
type DetResult struct {
	ClassID   int
	ClassName string
	Score     float32
	Box       image.Rectangle
}
