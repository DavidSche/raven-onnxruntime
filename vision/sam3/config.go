package sam3

import (
	ort "github.com/DavidSche/raven-onnxruntime/ort"
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
	textSeqLen    = 32
	padTokenId    = 49407
)

type Point struct {
	X, Y  float32
	Label Label
}

type Config struct {
	OnnxRuntimeLibPath string
	VisionModelPath    string
	TextModelPath      string
	DecoderModelPath   string

	UseCuda           bool
	NumThreads        int
	EnableCpuMemArena bool
	ApiVersion        ort.ApiVersion
}

func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: ort.DefaultLibraryPath(),
		VisionModelPath:    "./models/sam3_vision_encoder.onnx",
		TextModelPath:      "./models/sam3_text_encoder.onnx",
		DecoderModelPath:   "./models/sam3_decoder.onnx",
	}
}
