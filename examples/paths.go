package examples

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var (
	examplesDirOnce sync.Once
	examplesDir     string
	workspaceRoot   string
)

func ExamplesDir() string {
	examplesDirOnce.Do(func() {
		_, file, _, ok := runtime.Caller(0)
		if !ok {
			examplesDir = "."
			workspaceRoot = "."
			return
		}
		examplesDir = filepath.Dir(file)
		workspaceRoot = filepath.Clean(filepath.Join(examplesDir, "..", ".."))
	})
	return examplesDir
}

func WorkspaceRoot() string {
	_ = ExamplesDir()
	return workspaceRoot
}

func ExamplePath(parts ...string) string {
	all := append([]string{ExamplesDir()}, parts...)
	return filepath.Clean(filepath.Join(all...))
}

func WorkspacePath(parts ...string) string {
	all := append([]string{WorkspaceRoot()}, parts...)
	return filepath.Clean(filepath.Join(all...))
}

func ExampleModelsRoot() string {
	if env := os.Getenv("RAVEN_MODELS_DIR"); env != "" {
		return env
	}
	return WorkspacePath("models")
}

func ExampleORTLibraryPath() string {
	if env := os.Getenv("RAVEN_ORT_LIB_PATH"); env != "" {
		return env
	}
	candidates := []string{
		WorkspacePath("lib", "onnxruntime.dll"),
		WorkspacePath("lib", "onnxruntime.so"),
		WorkspacePath("lib", "onnxruntime.dylib"),
		ExamplePath("..", "lib", "onnxruntime.dll"),
	}
	for _, candidate := range candidates {
		if fileExists(candidate) {
			return candidate
		}
	}
	return candidates[0]
}

func ExampleModelPath(parts ...string) string {
	all := append([]string{ExampleModelsRoot()}, parts...)
	return filepath.Clean(filepath.Join(all...))
}

func ExampleImagePath(name string) string {
	return ExamplePath(name)
}

func ExampleArtifactPath(parts ...string) string {
	if env := os.Getenv("RAVEN_BENCH_ARTIFACTS_DIR"); env != "" {
		all := append([]string{env}, parts...)
		return filepath.Clean(filepath.Join(all...))
	}
	all := append([]string{WorkspacePath("artifacts", "benchmarks")}, parts...)
	return filepath.Clean(filepath.Join(all...))
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func normalizeModelPath(path string) string {
	if path == "" {
		return path
	}
	if filepath.IsAbs(path) || fileExists(path) {
		return path
	}
	if strings.HasPrefix(path, "..") || strings.HasPrefix(path, ".") {
		candidate := filepath.Clean(filepath.Join(ExamplesDir(), path))
		if fileExists(candidate) {
			return candidate
		}
		candidate = filepath.Clean(filepath.Join(WorkspaceRoot(), path))
		if fileExists(candidate) {
			return candidate
		}
	}
	return path
}
