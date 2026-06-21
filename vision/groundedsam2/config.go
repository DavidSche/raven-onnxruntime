package groundedsam2

import (
	"image"
	"path/filepath"

	"github.com/DavidSche/raven-onnxruntime/internal/modelpath"
	ort "github.com/DavidSche/raven-onnxruntime/ort"
	"github.com/DavidSche/raven-onnxruntime/vision/manifest"
)

// Config Grounded-SAM-2 模型配置
// Grounded-SAM-2 由 GroundingDINO (开放词汇检测) + SAM-2 (分割) 组成
type Config struct {
	// ModelPath 模型包目录路径 (包含 manifest.json 和 ONNX 子模型)
	// raven-onnxruntime 从 manifest.json 自动发现子模型路径
	ModelPath          string
	OnnxRuntimeLibPath string
	UseCuda            bool
	NumThreads         int
	EnableCpuMemArena  bool
	ApiVersion         ort.ApiVersion

	// 检测参数
	ConfThreshold float32
	InputSize     int // GroundingDINO 输入尺寸 (默认 800)
	MaxTextLen    int // 最大文本长度 (默认 256)
	MaxDetections int // 最大检测数量 (默认 100)

	// SAM-2 参数
	Sam2ImageSize int // SAM-2 输入尺寸 (默认 1024)
	MaskThreshold float32
}

// DefaultSegConfig 返回默认的 Grounded-SAM-2 分割配置
func DefaultSegConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		ModelPath:          modelpath.ModelPath("grounded-sam2"),
		UseCuda:            false,
		NumThreads:         1,
		EnableCpuMemArena:  true,
		ApiVersion:         ort.ApiVersion17,
		ConfThreshold:      0.35,
		InputSize:          800,
		MaxTextLen:         256,
		MaxDetections:      100,
		Sam2ImageSize:      1024,
		MaskThreshold:      0.0,
	}
}

// subModelPaths 解析后的子模型路径
type subModelPaths struct {
	gdinoImageEncoder string
	gdinoTextEncoder  string
	gdinoDetector     string
	sam2ImageEncoder  string
	sam2MaskDecoder   string
}

// modelParams 从 manifest 解析的模型结构参数
type modelParams struct {
	hiddenDim          int
	numFeatureLevels   int
	sam2BackboneStride int
	sam2EmbeddingSize  int
	useHighResFeatures bool
}

// resolveSubModelPaths 从 manifest.json 解析子模型路径
// 如果 manifest.json 不存在，回退到目录内默认文件名
func (c *Config) resolveSubModelPaths() (*subModelPaths, *modelParams, error) {
	paths := &subModelPaths{}
	params := &modelParams{
		hiddenDim:          256,
		numFeatureLevels:   4,
		sam2BackboneStride: 16,
		sam2EmbeddingSize:  64,
		useHighResFeatures: true,
	}

	mf, loadErr := manifest.Load(c.ModelPath)
	if loadErr == nil {
		// 从 manifest.json 解析子模型路径
		paths.gdinoImageEncoder = mf.SubModelPath(c.ModelPath, "gdino_image_encoder")
		paths.gdinoTextEncoder = mf.SubModelPath(c.ModelPath, "gdino_text_encoder")
		paths.gdinoDetector = mf.SubModelPath(c.ModelPath, "gdino_detector")
		paths.sam2ImageEncoder = mf.SubModelPath(c.ModelPath, "sam2_image_encoder")
		paths.sam2MaskDecoder = mf.SubModelPath(c.ModelPath, "sam2_mask_decoder")

		// 从 manifest 补充参数
		if c.InputSize == 0 {
			c.InputSize = mf.InputSizeAt(0, 800)
		}

		// 解析 gdino 参数
		gdinoParams := mf.ParamMap("gdino")
		if gdinoParams != nil {
			if v, ok := gdinoParams["hidden_dim"]; ok {
				if f, ok := v.(float64); ok {
					params.hiddenDim = int(f)
				}
			}
			if v, ok := gdinoParams["num_feature_levels"]; ok {
				if f, ok := v.(float64); ok {
					params.numFeatureLevels = int(f)
				}
			}
			if v, ok := gdinoParams["max_text_len"]; ok {
				if f, ok := v.(float64); ok && c.MaxTextLen == 0 {
					c.MaxTextLen = int(f)
				}
			}
		}

		// 解析 sam2 参数
		sam2Params := mf.ParamMap("sam2")
		if sam2Params != nil {
			if v, ok := sam2Params["backbone_stride"]; ok {
				if f, ok := v.(float64); ok {
					params.sam2BackboneStride = int(f)
				}
			}
			if v, ok := sam2Params["sam_image_embedding_size"]; ok {
				if f, ok := v.(float64); ok {
					params.sam2EmbeddingSize = int(f)
				}
			}
			if v, ok := sam2Params["use_high_res_features"]; ok {
				if b, ok := v.(bool); ok {
					params.useHighResFeatures = b
				}
			}
			if v, ok := sam2Params["image_size"]; ok {
				if f, ok := v.(float64); ok && c.Sam2ImageSize == 0 {
					c.Sam2ImageSize = int(f)
				}
			}
		}
	} else {
		// 回退：使用目录内默认文件名
		paths.gdinoImageEncoder = filepath.Join(c.ModelPath, "grounded_sam2_gdino_image_encoder.onnx")
		paths.gdinoTextEncoder = filepath.Join(c.ModelPath, "grounded_sam2_gdino_text_encoder.onnx")
		paths.gdinoDetector = filepath.Join(c.ModelPath, "grounded_sam2_gdino_detector.onnx")
		paths.sam2ImageEncoder = filepath.Join(c.ModelPath, "grounded_sam2_sam2_image_encoder.onnx")
		paths.sam2MaskDecoder = filepath.Join(c.ModelPath, "grounded_sam2_sam2_mask_decoder.onnx")
	}

	return paths, params, nil
}

// SegResult 分割检测结果
type SegResult struct {
	ClassID   int
	ClassName string
	Score     float32
	Box       image.Rectangle
	Mask      *image.Gray
	IoUScore  float32
}
