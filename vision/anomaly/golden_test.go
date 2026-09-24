package anomaly

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
)

const (
	goldenFixtureSchemaVersion  = "anomaly-golden-outputs-v1"
	goldenFixtureOpset          = 14
	goldenFixtureSampleCount    = 32
	goldenFixtureNormalCount    = 16
	goldenFixtureAnomalousCount = 16
)

type goldenFixture struct {
	SchemaVersion     string
	ModelArtifactHash string
	Opset             int64

	Input      []float32
	InputShape []int
	Score      []float32
	Map        []float32
	MapShape   []int
	Labels     []uint8
	Seed       int64
}

func readFloat32NPY(data []byte) ([]float32, []int, error) {
	header, shape, err := parseNPYHeader(data)
	if err != nil {
		return nil, nil, err
	}
	if header.descr != "<f4" && header.descr != "|f4" {
		return nil, nil, fmt.Errorf("unsupported npy dtype %q, want little-endian float32", header.descr)
	}
	if header.fortranOrder {
		return nil, nil, fmt.Errorf("fortran-order npy arrays are not supported")
	}
	if len(data) < header.dataOffset {
		return nil, nil, fmt.Errorf("truncated npy payload")
	}
	payload := data[header.dataOffset:]
	if len(payload)%4 != 0 || len(payload)/4 != planeSize(shape) {
		return nil, nil, fmt.Errorf("npy payload size %d does not match shape %v", len(payload), shape)
	}
	values := make([]float32, len(payload)/4)
	for index := range values {
		value := binary.LittleEndian.Uint32(payload[index*4:])
		values[index] = math.Float32frombits(value)
		if math.IsNaN(float64(values[index])) || math.IsInf(float64(values[index]), 0) {
			return nil, nil, fmt.Errorf("npy contains non-finite value at %d", index)
		}
	}
	return values, shape, nil
}

type npyHeader struct {
	descr        string
	fortranOrder bool
	dataOffset   int
}

var npyHeaderShapePattern = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)

func parseNPYHeader(data []byte) (npyHeader, []int, error) {
	if len(data) < 10 || !bytes.Equal(data[:6], []byte("\x93NUMPY")) {
		return npyHeader{}, nil, fmt.Errorf("invalid npy magic")
	}
	version := data[6]
	var headerLength int
	switch version {
	case 1:
		headerLength = int(binary.LittleEndian.Uint16(data[8:10]))
	case 2, 3:
		if len(data) < 12 {
			return npyHeader{}, nil, fmt.Errorf("truncated npy header")
		}
		headerLength = int(binary.LittleEndian.Uint32(data[8:12]))
	default:
		return npyHeader{}, nil, fmt.Errorf("unsupported npy version %d", version)
	}
	offset := 10
	if version >= 2 {
		offset = 12
	}
	if headerLength <= 0 || offset+headerLength > len(data) {
		return npyHeader{}, nil, fmt.Errorf("invalid npy header length %d", headerLength)
	}
	header := string(data[offset : offset+headerLength])
	if !strings.Contains(header, "'fortran_order': False") {
		return npyHeader{}, nil, fmt.Errorf("npy is not C-order")
	}
	match := npyHeaderShapePattern.FindStringSubmatch(header)
	if match == nil {
		return npyHeader{}, nil, fmt.Errorf("npy header has no shape")
	}
	shape := make([]int, 0, strings.Count(match[1], ",")+1)
	fields := strings.Fields(strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(match[1]), ","), ","))
	for _, field := range fields {
		field = strings.TrimSuffix(field, ",")
		var value int
		if _, err := fmt.Sscan(field, &value); err != nil || value <= 0 {
			return npyHeader{}, nil, fmt.Errorf("invalid npy shape value %q", field)
		}
		shape = append(shape, value)
	}
	descrStart := strings.Index(header, "'descr': '")
	if descrStart < 0 {
		return npyHeader{}, nil, fmt.Errorf("npy header has no descr")
	}
	descrStart += len("'descr': '")
	descrEnd := strings.IndexByte(header[descrStart:], '\'')
	if descrEnd < 0 {
		return npyHeader{}, nil, fmt.Errorf("npy descr is unterminated")
	}
	return npyHeader{descr: header[descrStart : descrStart+descrEnd], dataOffset: offset + headerLength}, shape, nil
}

func readGoldenNPZ(t *testing.T, path string) goldenFixture {
	t.Helper()
	archive, err := zip.OpenReader(path)
	if err != nil {
		t.Fatalf("open golden fixture: %v", err)
	}
	defer archive.Close()

	values := make(map[string]*zip.File, len(archive.File))
	for _, file := range archive.File {
		values[strings.TrimSuffix(file.Name, ".npy")] = file
	}
	for _, required := range []string{"input", "pred_score", "anomaly_map", "schema_version", "model_artifact_hash", "opset", "labels", "generation_seed"} {
		if _, exists := values[required]; !exists {
			t.Fatalf("golden fixture missing %q", required)
		}
	}

	schemaVersion, err := readStringNpzEntry(values["schema_version"])
	if err != nil {
		t.Fatalf("read golden schema_version: %v", err)
	}
	artifactHash, err := readStringNpzEntry(values["model_artifact_hash"])
	if err != nil {
		t.Fatalf("read golden model_artifact_hash: %v", err)
	}
	opset, err := readInt64NpzEntry(values["opset"])
	if err != nil {
		t.Fatalf("read golden opset: %v", err)
	}
	labels, labelShape, err := readUint8NpzEntry(values["labels"])
	if err != nil {
		t.Fatalf("read golden labels: %v", err)
	}
	if len(labelShape) != 1 {
		t.Fatalf("golden labels shape = %v, want 1-D", labelShape)
	}
	seedValues, seedShape, err := readInt64VectorNpzEntry(values["generation_seed"])
	if err != nil {
		t.Fatalf("read golden generation_seed: %v", err)
	}
	if len(seedShape) != 0 || len(seedValues) != 1 {
		t.Fatalf("golden generation_seed shape = %v, want scalar", seedShape)
	}
	if err != nil {
		t.Fatalf("read golden opset: %v", err)
	}

	input, inputShape, err := readFloat32NpzEntry(values["input"])
	if err != nil {
		t.Fatalf("read golden input: %v", err)
	}
	score, scoreShape, err := readFloat32NpzEntry(values["pred_score"])
	if err != nil {
		t.Fatalf("read golden score: %v", err)
	}
	if len(scoreShape) != 1 || scoreShape[0] != inputShape[0] {
		t.Fatalf("golden score shape = %v, want [%d]", scoreShape, inputShape[0])
	}
	mapValues, mapShape, err := readFloat32NpzEntry(values["anomaly_map"])
	if err != nil {
		t.Fatalf("read golden map: %v", err)
	}
	return goldenFixture{
		SchemaVersion:     schemaVersion,
		ModelArtifactHash: artifactHash,
		Opset:             opset,
		Labels:            labels,
		Seed:              seedValues[0],
		Input:             input,
		InputShape:        inputShape,
		Score:             score,
		Map:               mapValues,
		MapShape:          mapShape,
	}
}

func readFloat32NpzEntry(file *zip.File) ([]float32, []int, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, nil, fmt.Errorf("open npz float entry: %w", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, fmt.Errorf("read npz float entry: %w", err)
	}
	return readFloat32NPY(data)
}

func readStringNpzEntry(file *zip.File) (string, error) {
	reader, err := file.Open()
	if err != nil {
		return "", fmt.Errorf("open npz string entry: %w", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return "", fmt.Errorf("read npz string entry: %w", err)
	}

	header, shape, err := parseNPYHeader(data)
	if err != nil {
		return "", err
	}
	if !strings.HasPrefix(header.descr, "<U") || header.fortranOrder || len(shape) != 0 {
		return "", fmt.Errorf("unsupported npy scalar string dtype %q or shape %v", header.descr, shape)
	}
	payload := data[header.dataOffset:]
	if len(payload)%4 != 0 {
		return "", fmt.Errorf("npy string payload size %d is not UTF-32 aligned", len(payload))
	}

	runes := make([]rune, 0, len(payload)/4)
	for index := 0; index < len(payload); index += 4 {
		runes = append(runes, rune(binary.LittleEndian.Uint32(payload[index:])))
	}
	return string(runes), nil
}

func readInt64NpzEntry(file *zip.File) (int64, error) {
	reader, err := file.Open()
	if err != nil {
		return 0, fmt.Errorf("open npz integer entry: %w", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return 0, fmt.Errorf("read npz integer entry: %w", err)
	}

	header, shape, err := parseNPYHeader(data)
	if err != nil {
		return 0, err
	}
	if header.descr != "<i8" || header.fortranOrder || len(shape) != 0 {
		return 0, fmt.Errorf("unsupported npy scalar integer dtype %q or shape %v", header.descr, shape)
	}
	payload := data[header.dataOffset:]
	if len(payload) != 8 {
		return 0, fmt.Errorf("npy integer payload size %d, want 8", len(payload))
	}
	return int64(binary.LittleEndian.Uint64(payload)), nil
}

func readUint8NpzEntry(file *zip.File) ([]uint8, []int, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, nil, fmt.Errorf("open npz uint8 entry: %w", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, fmt.Errorf("read npz uint8 entry: %w", err)
	}
	header, shape, err := parseNPYHeader(data)
	if err != nil {
		return nil, nil, err
	}
	if header.descr != "<u1" && header.descr != "|u1" {
		return nil, nil, fmt.Errorf("unsupported npy dtype %q, want uint8", header.descr)
	}
	if header.fortranOrder {
		return nil, nil, fmt.Errorf("fortran-order uint8 npy arrays are not supported")
	}
	payload := data[header.dataOffset:]
	return payload, shape, nil
}

func readInt64VectorNpzEntry(file *zip.File) ([]int64, []int, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, nil, fmt.Errorf("open npz int64 entry: %w", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, fmt.Errorf("read npz int64 entry: %w", err)
	}
	header, shape, err := parseNPYHeader(data)
	if err != nil {
		return nil, nil, err
	}
	if header.descr != "<i8" || header.fortranOrder {
		return nil, nil, fmt.Errorf("unsupported npy dtype %q, want little-endian int64", header.descr)
	}
	payload := data[header.dataOffset:]
	if len(payload)%8 != 0 {
		return nil, nil, fmt.Errorf("npy int64 payload size %d is not aligned", len(payload))
	}
	values := make([]int64, len(payload)/8)
	for index := range values {
		values[index] = int64(binary.LittleEndian.Uint64(payload[index*8:]))
	}
	return values, shape, nil
}

func goldenModelArtifactHash(t *testing.T, cfg Config) string {
	t.Helper()
	type modelCard struct {
		ModelArtifactHash string `json:"model_artifact_hash"`
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(cfg.ModelPath), "reference", "model_card.json"))
	if err != nil {
		t.Fatalf("read model card: %v", err)
	}
	var card modelCard
	if err := json.Unmarshal(data, &card); err != nil {
		t.Fatalf("decode model card: %v", err)
	}
	return strings.TrimSpace(card.ModelArtifactHash)
}

func planeSize(shape []int) int {
	size := 1
	for _, dimension := range shape {
		if dimension <= 0 || size > math.MaxInt/dimension {
			return 0
		}
		size *= dimension
	}
	return size
}

func floatMetric(got, expected []float32) (maxError, mae, rmse float64) {
	var sumSquared, sumAbsolute float64
	for index, value := range got {
		errorValue := math.Abs(float64(value - expected[index]))
		if errorValue > maxError {
			maxError = errorValue
		}
		sumAbsolute += errorValue
		sumSquared += errorValue * errorValue
	}
	mae = sumAbsolute / float64(len(got))
	rmse = math.Sqrt(sumSquared / float64(len(got)))
	return maxError, mae, rmse
}

func cosineSimilarity(got, expected []float32) float64 {
	var dot, gotNorm, expectedNorm float64
	for index := range got {
		left := float64(got[index])
		right := float64(expected[index])
		dot += left * right
		gotNorm += left * left
		expectedNorm += right * right
	}
	denominator := math.Sqrt(gotNorm) * math.Sqrt(expectedNorm)
	if denominator == 0 {
		if dot == 0 {
			return 1
		}
		return 0
	}
	return dot / denominator
}

func combinedTolerance(expected float64) float64 {
	return 1e-5 + 5e-2*math.Max(math.Abs(expected), 1e-6)
}

func runGoldenPrediction(engine *Engine, input []float32, width, height int) (*AnomalyResult, error) {
	tensor, err := engine.session.NewTensor([]int64{1, 3, int64(height), int64(width)}, input)
	if err != nil {
		return nil, fmt.Errorf("create golden input tensor: %w", err)
	}
	defer tensor.Destroy()

	outputs, err := engine.session.Run(map[string]*ort.Value{engine.session.InputNames[0]: tensor})
	if err != nil {
		return nil, fmt.Errorf("golden ONNX run: %w", err)
	}
	defer ort.DestroyValues(outputs)
	return postprocess(outputs, engine.config, preprocessResult{originalWidth: width, originalHeight: height})
}

func TestAnomalyGoldenNumericalFixture(t *testing.T) {
	for _, variant := range []string{"efficientad-s", "padim-r18"} {
		t.Run(variant, func(t *testing.T) {
			cfg := gatedFixtureConfig(t, variant)
			// Keep map output enabled so the Go postprocess result can be compared
			// point-by-point to the raw ONNX golden map.
			cfg.OutputAnomalyMap = true
			cfg.MapDownsample = 1
			engine, err := NewEngine(cfg)
			if err != nil {
				t.Fatalf("NewEngine() error = %v", err)
			}
			defer engine.Destroy()

			fixture := readGoldenNPZ(t, filepath.Join(filepath.Dir(cfg.ModelPath), "reference", "golden_outputs.npz"))
			if got := fixture.InputShape; len(got) != 4 || got[0] != goldenFixtureSampleCount || got[1] != 3 || got[2] != cfg.InputSize.Y || got[3] != cfg.InputSize.X {
				t.Fatalf("golden input shape = %v, want [N,3,%d,%d]", got, cfg.InputSize.Y, cfg.InputSize.X)
			}
			if len(fixture.MapShape) != 4 || fixture.MapShape[0] != fixture.InputShape[0] || fixture.MapShape[1] != 1 || fixture.MapShape[2] != cfg.InputSize.Y || fixture.MapShape[3] != cfg.InputSize.X {
				t.Fatalf("golden map shape = %v, want [%d,1,%d,%d]", fixture.MapShape, fixture.InputShape[0], cfg.InputSize.Y, cfg.InputSize.X)
			}
			if fixture.SchemaVersion != goldenFixtureSchemaVersion {
				t.Fatalf("golden schema_version = %q, want %q", fixture.SchemaVersion, goldenFixtureSchemaVersion)
			}
			if fixture.Opset != goldenFixtureOpset {
				t.Fatalf("golden opset = %d, want %d", fixture.Opset, goldenFixtureOpset)
			}

			if fixture.Seed <= 0 {
				t.Fatalf("golden generation seed = %d, want positive", fixture.Seed)
			}
			if len(fixture.Labels) != fixture.InputShape[0] {
				t.Fatalf("golden label count = %d, want %d", len(fixture.Labels), fixture.InputShape[0])
			}
			normalCount := 0
			anomalousCount := 0
			for _, label := range fixture.Labels {
				switch label {
				case 0:
					normalCount++
				case 1:
					anomalousCount++
				default:
					t.Fatalf("golden label %d is neither normal nor anomalous", label)
				}
			}
			if normalCount != goldenFixtureNormalCount || anomalousCount != goldenFixtureAnomalousCount {
				t.Fatalf("golden label balance = normal %d, anomalous %d, want 16/16", normalCount, anomalousCount)
			}

			artifactHash := goldenModelArtifactHash(t, cfg)
			if artifactHash == "" {
				t.Fatal("model card does not contain model_artifact_hash")
			}
			if strings.TrimSpace(fixture.ModelArtifactHash) != artifactHash {
				t.Fatalf("golden model artifact hash = %q, want %q", fixture.ModelArtifactHash, artifactHash)
			}

			for batchIndex := 0; batchIndex < fixture.InputShape[0]; batchIndex++ {
				inputStart := batchIndex * 3 * cfg.InputSize.X * cfg.InputSize.Y
				inputEnd := (batchIndex + 1) * 3 * cfg.InputSize.X * cfg.InputSize.Y
				result, err := runGoldenPrediction(engine, fixture.Input[inputStart:inputEnd], cfg.InputSize.X, cfg.InputSize.Y)
				if err != nil {
					t.Fatalf("golden prediction[%d] error = %v", batchIndex, err)
				}
				if result == nil || len(result.AnomalyMap) != planeSize(fixture.MapShape[1:]) {
					t.Fatalf("golden prediction[%d] returned invalid map", batchIndex)
				}

				expectedScore := fixture.Score[batchIndex]
				scoreTolerance := 1e-6 + 1e-3*math.Max(math.Abs(float64(expectedScore)), 1e-6)
				if scoreError := math.Abs(float64(result.Score - expectedScore)); scoreError > scoreTolerance {
					t.Fatalf("golden score[%d] = %v, expected %v, error %v > tolerance %v", batchIndex, result.Score, expectedScore, scoreError, scoreTolerance)
				}
				if scoreMax := maxFloat32(result.AnomalyMap); math.Abs(float64(result.Score-scoreMax)) > cfg.ScoreContract.Assertion.Atol+cfg.ScoreContract.Assertion.Rtol*math.Max(math.Abs(float64(scoreMax)), 1e-12) {
					t.Fatalf("golden score[%d] is not map max: score=%v map_max=%v", batchIndex, result.Score, scoreMax)
				}

				expectedMapStart := batchIndex * planeSize(fixture.MapShape[1:])
				expectedMapEnd := (batchIndex + 1) * planeSize(fixture.MapShape[1:])
				expectedMap := fixture.Map[expectedMapStart:expectedMapEnd]
				cosine := cosineSimilarity(result.AnomalyMap, expectedMap)
				if cosine <= 0.999 {
					t.Fatalf("golden map[%d] cosine = %v, want > 0.999", batchIndex, cosine)
				}
				p99Error := 0.0
				errorsOut := make([]float64, len(expectedMap))
				for index := range expectedMap {
					errorsOut[index] = math.Abs(float64(result.AnomalyMap[index] - expectedMap[index]))
					tolerance := combinedTolerance(float64(expectedMap[index]))
					if errorsOut[index] > tolerance {
						t.Fatalf("golden map[%d][%d] error %v > tolerance %v", batchIndex, index, errorsOut[index], tolerance)
					}
				}
				sortFloat64(errorsOut)
				p99Error = errorsOut[int(math.Floor(0.99*float64(len(errorsOut)-1)))]
				if p99Error > combinedTolerance(float64(expectedMap[maxFloat32Index(expectedMap)])) {
					t.Fatalf("golden map[%d] P99 error %v exceeds maximum-point combined tolerance", batchIndex, p99Error)
				}
				maxError, mae, rmse := floatMetric(result.AnomalyMap, expectedMap)
				t.Logf("golden[%d]: score_error=%g map_max_error=%g map_mae=%g map_rmse=%g cosine=%g p99_error=%g", batchIndex, math.Abs(float64(result.Score-expectedScore)), maxError, mae, rmse, cosine, p99Error)
			}
		})
	}
}

func maxFloat32(values []float32) float32 {
	maximum := values[0]
	for _, value := range values[1:] {
		if value > maximum {
			maximum = value
		}
	}
	return maximum
}

func maxFloat32Index(values []float32) int {
	index := 0
	for candidate, value := range values[1:] {
		if value > values[index] {
			index = candidate + 1
		}
	}
	return index
}

func sortFloat64(values []float64) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}
