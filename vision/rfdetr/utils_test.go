package rfdetr

import (
	"os"
	"path/filepath"
	"testing"
)

// ─────────────────────────────────────────────────────────────
// Minimal protobuf builders.
//
// Field numbers mirror the ONNX serialization this scanner targets
// (verified against the raven-rfdetr exported models, onnx 1.22.0):
//   ModelProto.graph=7, ModelProto.metadata_props=14 (StringStringEntryProto:
//   key=1, value=2), GraphProto.input=11, ValueInfoProto.type=2,
//   TypeProto.tensor_type=1, TypeProto.Tensor.shape=2,
//   TensorShapeProto.dim=1 (Dimension: dim_value=1, dim_param=2).
// ─────────────────────────────────────────────────────────────

func pbVarint(v uint64) []byte {
	var out []byte
	for v >= 0x80 {
		out = append(out, byte(v)|0x80)
		v >>= 7
	}
	return append(out, byte(v))
}

func pbTag(field, wireType int) []byte {
	return pbVarint(uint64(field<<3 | wireType))
}

func pbBytes(field int, payload []byte) []byte {
	out := pbTag(field, 2)
	out = append(out, pbVarint(uint64(len(payload)))...)
	return append(out, payload...)
}

func pbVarintField(field int, v uint64) []byte {
	out := pbTag(field, 0)
	return append(out, pbVarint(v)...)
}

func dimValue(v int64) []byte  { return pbVarintField(1, uint64(v)) }
func dimParam(s string) []byte { return pbBytes(2, []byte(s)) }

func tensorShape(dims ...[]byte) []byte {
	var out []byte
	for _, d := range dims {
		out = append(out, pbBytes(1, d)...)
	}
	return out
}

func tensorType(shape []byte) []byte {
	out := pbVarintField(1, 1) // TypeProto.Tensor.elem_type = FLOAT (required by onnx checker)
	return append(out, pbBytes(2, shape)...)
}
func typeProto(tensor []byte) []byte { return pbBytes(1, tensor) }
func valueInfo(typ []byte) []byte {
	out := pbBytes(1, []byte("input")) // ValueInfoProto.name (required by onnx checker)
	return append(out, pbBytes(2, typ)...)
}

func graphInputs(inputs ...[]byte) []byte {
	out := pbBytes(2, []byte("graph")) // GraphProto.name (required by onnx checker)
	for _, in := range inputs {
		out = append(out, pbBytes(11, in)...)
	}
	return out
}

func metaEntry(key, value string) []byte {
	out := pbBytes(1, []byte(key))
	return append(out, pbBytes(2, []byte(value))...)
}

func modelProto(graph []byte, metas ...[]byte) []byte {
	out := pbVarintField(1, 8) // ir_version (matches the raven-rfdetr exports)
	// opset_import (OperatorSetIdProto: domain=1, version=2) — required by onnx
	out = append(out, pbBytes(8, append(pbBytes(1, nil), pbVarintField(2, 17)...))...)
	out = append(out, pbBytes(7, graph)...)
	for _, m := range metas {
		out = append(out, pbBytes(14, m)...)
	}
	return out
}

// staticModel384 builds a model whose single graph input has concrete
// [1, 3, 384, 384] dims — the input size is detectable without metadata.
func staticModel384() []byte {
	shape := tensorShape(dimValue(1), dimValue(3), dimValue(384), dimValue(384))
	vi := valueInfo(typeProto(tensorType(shape)))
	return modelProto(graphInputs(vi))
}

// dynamicModel builds a model whose input dims are symbolic
// [batch, 3, height, width]; withMeta=true also embeds the raven-rfdetr
// `resolution` metadata prop.
func dynamicModel(resolution string, withMeta bool) []byte {
	shape := tensorShape(dimParam("batch"), dimValue(3), dimParam("height"), dimParam("width"))
	vi := valueInfo(typeProto(tensorType(shape)))
	g := graphInputs(vi)
	if withMeta {
		return modelProto(g, metaEntry("model_name", "rf-detr-nano"), metaEntry("resolution", resolution))
	}
	return modelProto(g)
}

func writeTempModel(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "v1.0.onnx")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write model fixture: %v", err)
	}
	return path
}

// ─────────────────────────────────────────────────────────────
// detectInputSizeAndDynamicBatch
// ─────────────────────────────────────────────────────────────

func TestDetectInputSizeAndDynamicBatch_ResolutionTableByName(t *testing.T) {
	cases := []struct {
		name     string
		wantSize int
	}{
		{"rf-detr-nano.onnx", 384},
		{"rf-detr-small-v2.onnx", 512},
		{"rf-detr-medium.onnx", 576},
		{"rf-detr-large-2026.onnx", 704},
		{"rf-detr-xlarge.onnx", 700},
		{"rf-detr-seg-nano.onnx", 312},
		{"rf-detr-seg-large.onnx", 504},
	}
	for _, tc := range cases {
		// The resolution table is consulted before any file I/O, so the
		// path itself does not need to exist.
		path := filepath.Join("nonexistent", tc.name)
		size, dynamic := detectInputSizeAndDynamicBatch(path)
		if size != tc.wantSize || dynamic {
			t.Errorf("%s: got (%d, %v), want (%d, false)", tc.name, size, dynamic, tc.wantSize)
		}
	}
}

func TestDetectInputSizeAndDynamicBatch_StaticInputDims(t *testing.T) {
	path := writeTempModel(t, staticModel384())
	size, dynamic := detectInputSizeAndDynamicBatch(path)
	if size != 384 || dynamic {
		t.Errorf("got (%d, %v), want (384, false)", size, dynamic)
	}
}

func TestDetectInputSizeAndDynamicBatch_DynamicDimsMetadataFallback(t *testing.T) {
	// Symbolic H/W dims cannot report a size from the graph input alone —
	// the `resolution` metadata prop (raven-rfdetr exporter) must be used,
	// keeping the parsed dynamic-batch flag.
	path := writeTempModel(t, dynamicModel("384", true))
	size, dynamic := detectInputSizeAndDynamicBatch(path)
	if size != 384 || !dynamic {
		t.Errorf("got (%d, %v), want (384, true) via metadata fallback", size, dynamic)
	}
}

func TestDetectInputSizeAndDynamicBatch_DynamicDimsNoMetadata(t *testing.T) {
	path := writeTempModel(t, dynamicModel("", false))
	size, dynamic := detectInputSizeAndDynamicBatch(path)
	if size != 640 || dynamic {
		t.Errorf("got (%d, %v), want (640, false)", size, dynamic)
	}
}

func TestDetectInputSizeAndDynamicBatch_MissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.onnx")
	size, dynamic := detectInputSizeAndDynamicBatch(path)
	if size != 640 || dynamic {
		t.Errorf("got (%d, %v), want (640, false)", size, dynamic)
	}
}

func TestDetectInputSize_StaticAndMetadata(t *testing.T) {
	if size := detectInputSize(writeTempModel(t, staticModel384())); size != 384 {
		t.Errorf("static: got %d, want 384", size)
	}
	if size := detectInputSize(writeTempModel(t, dynamicModel("512", true))); size != 512 {
		t.Errorf("metadata fallback: got %d, want 512", size)
	}
	if size := detectInputSize(writeTempModel(t, dynamicModel("", false))); size != 640 {
		t.Errorf("no info fallback: got %d, want 640", size)
	}
}

// ─────────────────────────────────────────────────────────────
// parseOnnxMetadataResolution / parseStringStringEntry
// ─────────────────────────────────────────────────────────────

func TestParseOnnxMetadataResolution(t *testing.T) {
	t.Run("present_after_other_keys", func(t *testing.T) {
		p := writeTempModel(t, modelProto(graphInputs(),
			metaEntry("model_name", "rf-detr-nano"),
			metaEntry("resolution", "384"),
		))
		size, ok := parseOnnxMetadataResolution(p)
		if !ok || size != 384 {
			t.Errorf("got (%d, %v), want (384, true)", size, ok)
		}
	})
	t.Run("surrounding_whitespace", func(t *testing.T) {
		p := writeTempModel(t, modelProto(graphInputs(), metaEntry("resolution", " 384 ")))
		size, ok := parseOnnxMetadataResolution(p)
		if !ok || size != 384 {
			t.Errorf("got (%d, %v), want (384, true)", size, ok)
		}
	})
	t.Run("absent", func(t *testing.T) {
		p := writeTempModel(t, modelProto(graphInputs(), metaEntry("model_name", "rf-detr-nano")))
		size, ok := parseOnnxMetadataResolution(p)
		if ok || size != 0 {
			t.Errorf("got (%d, %v), want (0, false)", size, ok)
		}
	})
	t.Run("empty_model", func(t *testing.T) {
		p := writeTempModel(t, modelProto(graphInputs()))
		size, ok := parseOnnxMetadataResolution(p)
		if ok || size != 0 {
			t.Errorf("got (%d, %v), want (0, false)", size, ok)
		}
	})
	t.Run("non_numeric", func(t *testing.T) {
		p := writeTempModel(t, modelProto(graphInputs(), metaEntry("resolution", "abc")))
		size, ok := parseOnnxMetadataResolution(p)
		if ok || size != 0 {
			t.Errorf("got (%d, %v), want (0, false)", size, ok)
		}
	})
	t.Run("zero", func(t *testing.T) {
		p := writeTempModel(t, modelProto(graphInputs(), metaEntry("resolution", "0")))
		size, ok := parseOnnxMetadataResolution(p)
		if ok || size != 0 {
			t.Errorf("got (%d, %v), want (0, false)", size, ok)
		}
	})
	t.Run("negative", func(t *testing.T) {
		p := writeTempModel(t, modelProto(graphInputs(), metaEntry("resolution", "-384")))
		size, ok := parseOnnxMetadataResolution(p)
		if ok || size != 0 {
			t.Errorf("got (%d, %v), want (0, false)", size, ok)
		}
	})
	t.Run("missing_file", func(t *testing.T) {
		size, ok := parseOnnxMetadataResolution(filepath.Join(t.TempDir(), "nope.onnx"))
		if ok || size != 0 {
			t.Errorf("got (%d, %v), want (0, false)", size, ok)
		}
	})
}

func TestParseStringStringEntry(t *testing.T) {
	data := append(pbBytes(1, []byte("resolution")), pbBytes(2, []byte("384"))...)
	key, val, ok := parseStringStringEntry(data)
	if !ok || key != "resolution" || val != "384" {
		t.Errorf("got (%q, %q, %v), want (\"resolution\", \"384\", true)", key, val, ok)
	}

	// Entry without a key is not a usable key/value pair.
	if _, _, ok := parseStringStringEntry(pbBytes(2, []byte("x"))); ok {
		t.Error("entry without key should report not-ok")
	}
	// Entry without a value is likewise not usable.
	if _, _, ok := parseStringStringEntry(pbBytes(1, []byte("resolution"))); ok {
		t.Error("entry without value should report not-ok")
	}
}

// ─────────────────────────────────────────────────────────────
// parseOnnxInputSizeAndDynamicBatch (graph input dims)
// ─────────────────────────────────────────────────────────────

func TestParseOnnxInputSizeAndDynamicBatch(t *testing.T) {
	// Concrete dims -> size detected, static batch.
	p := writeTempModel(t, staticModel384())
	size, dynamic, err := parseOnnxInputSizeAndDynamicBatch(p)
	if err != nil || size != 384 || dynamic {
		t.Errorf("static: got (%d, %v, %v), want (384, false, nil)", size, dynamic, err)
	}

	// Symbolic dims -> size 0, but batch dynamism is reported so callers can
	// keep it when falling back to the metadata resolution.
	p = writeTempModel(t, dynamicModel("", false))
	size, dynamic, err = parseOnnxInputSizeAndDynamicBatch(p)
	if err != nil || size != 0 || !dynamic {
		t.Errorf("dynamic: got (%d, %v, %v), want (0, true, nil)", size, dynamic, err)
	}
}
