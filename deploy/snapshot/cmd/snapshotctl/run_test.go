package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoadPodParsesJSONManifest(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "worker.json")
	require.NoError(t, os.WriteFile(path, []byte(`{
  "apiVersion": "v1",
  "kind": "Pod",
  "metadata": {
    "name": "test-worker",
    "namespace": "test-ns"
  },
  "spec": {
    "containers": [
      {
        "name": "main",
        "image": "example.com/test:latest",
        "resources": {
          "limits": {
            "nvidia.com/gpu": "1"
          }
        }
      }
    ]
  }
}`), 0o644))

	pod, err := loadPod(path)
	require.NoError(t, err)
	require.Equal(t, "test-worker", pod.Name)
	require.Equal(t, "test-ns", pod.Namespace)
	require.Len(t, pod.Spec.Containers, 1)
}
