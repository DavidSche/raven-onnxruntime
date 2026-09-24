package anomaly

import "os"

func writeFile(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}

func readFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}
