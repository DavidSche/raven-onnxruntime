// Package manifest implements the Raven Model Package (RMP) specification.
//
// RMP defines a standard directory layout for deployed models:
//
//	model_package/
//	├── manifest.json       ← Model self-description
//	├── *.onnx              ← ONNX model files
//	├── labels.txt          ← (optional) Label file
//	└── ...                 ← Other assets
//
// manifest.json format (v1.0):
//
//	{
//	  "format_version": "1.0",
//	  "model_type": "grounded-sam2",
//	  "model_version": "1.0.0",
//	  "task": "segmentation",
//	  "runtime": "onnxruntime",
//	  "input_size": [800, 800],
//	  "labels": "labels.txt",
//	  "sub_models": {
//	    "gdino_image_encoder": "gdino_image_encoder.onnx",
//	    ...
//	  },
//	  "params": {
//	    ...
//	  }
//	}
package manifest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Manifest represents a RMP manifest.json file
type Manifest struct {
	FormatVersion string            `json:"format_version"`
	ModelType     string            `json:"model_type"`
	ModelVersion  string            `json:"model_version"`
	Task          string            `json:"task"`
	Runtime       string            `json:"runtime"`
	InputSize     []int             `json:"input_size"`
	Labels        string            `json:"labels,omitempty"`
	SubModels     map[string]string `json:"sub_models"`
	Params        map[string]any    `json:"params,omitempty"`
}

// Load reads and parses a manifest.json from the given model directory.
// modelPath can be either a directory containing manifest.json, or a file path
// within such a directory.
func Load(modelPath string) (*Manifest, error) {
	dir := modelPath

	// If modelPath points to a file, use its directory
	info, err := os.Stat(modelPath)
	if err != nil {
		return nil, fmt.Errorf("cannot stat model path %q: %w", modelPath, err)
	}
	if !info.IsDir() {
		dir = filepath.Dir(modelPath)
	}

	manifestPath := filepath.Join(dir, "manifest.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("cannot read manifest %q: %w", manifestPath, err)
	}

	m := &Manifest{}
	if err := json.Unmarshal(data, m); err != nil {
		return nil, fmt.Errorf("cannot parse manifest %q: %w", manifestPath, err)
	}

	return m, nil
}

// SubModelPath returns the absolute path for a named sub-model.
// The name corresponds to a key in manifest.json's "sub_models" field.
func (m *Manifest) SubModelPath(modelDir, name string) string {
	if rel, ok := m.SubModels[name]; ok {
		return filepath.Join(modelDir, rel)
	}
	return ""
}

// ParamInt returns an integer parameter from the params section.
func (m *Manifest) ParamInt(key string, defaultVal int) int {
	if m.Params == nil {
		return defaultVal
	}
	val, ok := m.Params[key]
	if !ok {
		return defaultVal
	}
	switch v := val.(type) {
	case float64:
		return int(v)
	case int:
		return v
	case json.Number:
		if n, err := v.Int64(); err == nil {
			return int(n)
		}
	}
	return defaultVal
}

// ParamBool returns a boolean parameter from the params section.
func (m *Manifest) ParamBool(key string, defaultVal bool) bool {
	if m.Params == nil {
		return defaultVal
	}
	val, ok := m.Params[key]
	if !ok {
		return defaultVal
	}
	if b, ok := val.(bool); ok {
		return b
	}
	return defaultVal
}

// ParamFloat returns a float64 parameter from the params section.
func (m *Manifest) ParamFloat(key string, defaultVal float64) float64 {
	if m.Params == nil {
		return defaultVal
	}
	val, ok := m.Params[key]
	if !ok {
		return defaultVal
	}
	switch v := val.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	}
	return defaultVal
}

// ParamMap returns a nested map parameter from the params section.
func (m *Manifest) ParamMap(key string) map[string]any {
	if m.Params == nil {
		return nil
	}
	val, ok := m.Params[key]
	if !ok {
		return nil
	}
	if m, ok := val.(map[string]any); ok {
		return m
	}
	return nil
}

// InputSizeAt returns the input size at the given dimension index.
func (m *Manifest) InputSizeAt(dim int, defaultVal int) int {
	if m.InputSize == nil || dim >= len(m.InputSize) {
		return defaultVal
	}
	return m.InputSize[dim]
}
