/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestRunAppliesVersionedOverride(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	blueprintPath := writeTestFile(t, directory, "blueprint.yaml", betaBlueprintYAML)
	overridePath := writeTestFile(t, directory, "override.json", alphaOverrideDGDJSON)
	outputPath := filepath.Join(directory, "effective.yaml")
	stderr := &bytes.Buffer{}
	stdout := &bytes.Buffer{}

	err := run([]string{
		"--blueprint", blueprintPath,
		"--override", overridePath,
		"--output", outputPath,
	}, stdout, stderr)
	require.NoError(t, err)
	assert.Empty(t, stdout.String())
	assert.Empty(t, stderr.String())

	effective := mustReadDGD(t, outputPath)
	assert.Equal(t, "nvidia.com/v1beta1", effective.GetAPIVersion())
	assert.Equal(t, "DynamoGraphDeployment", effective.GetKind())
	assert.Equal(t, "generated", effective.GetName())
	assert.Equal(t, "new-image", mainContainerImage(t, effective))
}

func TestRunWithEmptyOverridePreservesBlueprint(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	blueprintPath := writeTestFile(t, directory, "blueprint.yaml", betaBlueprintYAML)
	overridePath := writeTestFile(t, directory, "override.yaml", `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
`)
	outputPath := filepath.Join(directory, "effective.yaml")
	stderr := &bytes.Buffer{}

	err := run([]string{
		"--blueprint", blueprintPath,
		"--override", overridePath,
		"--output", outputPath,
	}, &bytes.Buffer{}, stderr)
	require.NoError(t, err)
	assert.Empty(t, stderr.String())
	assert.Equal(t, mustReadDGD(t, blueprintPath), mustReadDGD(t, outputPath))
}

func TestRunEmitsWarningsForIgnoredTopology(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	blueprintPath := writeTestFile(t, directory, "blueprint.yaml", betaBlueprintYAML)
	overridePath := writeTestFile(t, directory, "override.yaml", `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - name: Missing
    replicas: 2
`)
	outputPath := filepath.Join(directory, "effective.yaml")
	stderr := &bytes.Buffer{}

	err := run([]string{
		"--blueprint", blueprintPath,
		"--override", overridePath,
		"--output", outputPath,
	}, &bytes.Buffer{}, stderr)
	require.NoError(t, err)
	assert.Contains(
		t,
		stderr.String(),
		"warning: spec.components[name=Missing]: ignored because the generated blueprint has no such component",
	)
	assert.Equal(t, "old-image", mainContainerImage(t, mustReadDGD(t, outputPath)))
}

func TestRunDoesNotReplaceOutputOnFailure(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	blueprintPath := writeTestFile(t, directory, "blueprint.yaml", betaBlueprintYAML)
	overridePath := writeTestFile(t, directory, "override.yaml", `
apiVersion: nvidia.com/v2
kind: DynamoGraphDeployment
`)
	outputPath := writeTestFile(t, directory, "effective.yaml", "existing output\n")

	err := run([]string{
		"--blueprint", blueprintPath,
		"--override", overridePath,
		"--output", outputPath,
	}, &bytes.Buffer{}, &bytes.Buffer{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "apply DGD override")
	content, readErr := os.ReadFile(outputPath)
	require.NoError(t, readErr)
	assert.Equal(t, "existing output\n", string(content))
}

func TestRunInstallsExecutable(t *testing.T) {
	t.Parallel()

	installPath := filepath.Join(t.TempDir(), "dgd-apply-overrides")
	stderr := &bytes.Buffer{}
	require.NoError(t, run([]string{"--install-to", installPath}, &bytes.Buffer{}, stderr))
	assert.Empty(t, stderr.String())

	installed, err := os.Stat(installPath)
	require.NoError(t, err)
	currentPath, err := os.Executable()
	require.NoError(t, err)
	current, err := os.Stat(currentPath)
	require.NoError(t, err)
	assert.Equal(t, current.Size(), installed.Size())
	assert.Equal(t, os.FileMode(0o755), installed.Mode().Perm())
}

func TestRunPrintsProtocolVersion(t *testing.T) {
	t.Parallel()

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	require.NoError(t, run([]string{"--protocol-version"}, stdout, stderr))
	assert.Equal(t, protocolVersion+"\n", stdout.String())
	assert.Empty(t, stderr.String())
}

func TestRunRejectsInvalidArgumentsAndInput(t *testing.T) {
	t.Parallel()

	t.Run("missing flags", func(t *testing.T) {
		err := run(nil, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "--blueprint, --override, --output")
	})

	t.Run("unexpected positional argument", func(t *testing.T) {
		err := run([]string{"extra"}, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unexpected positional arguments")
	})

	t.Run("install mixed with merge flags", func(t *testing.T) {
		err := run([]string{"--install-to", "/tmp/tool", "--blueprint", "/tmp/blueprint"}, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "--install-to cannot be combined")
	})

	t.Run("protocol version mixed with merge flags", func(t *testing.T) {
		err := run([]string{"--protocol-version", "--blueprint", "/tmp/blueprint"}, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "--protocol-version cannot be combined")
	})

	t.Run("protocol version mixed with install mode", func(t *testing.T) {
		err := run([]string{"--protocol-version", "--install-to", "/tmp/tool"}, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "--install-to and --protocol-version are mutually exclusive")
	})

	t.Run("malformed blueprint", func(t *testing.T) {
		directory := t.TempDir()
		blueprintPath := writeTestFile(t, directory, "blueprint.yaml", "spec: [\n")
		overridePath := writeTestFile(t, directory, "override.yaml", alphaOverrideDGDJSON)
		err := run([]string{
			"--blueprint", blueprintPath,
			"--override", overridePath,
			"--output", filepath.Join(directory, "effective.yaml"),
		}, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "decode blueprint YAML")
	})

	t.Run("malformed override", func(t *testing.T) {
		directory := t.TempDir()
		blueprintPath := writeTestFile(t, directory, "blueprint.yaml", betaBlueprintYAML)
		overridePath := writeTestFile(t, directory, "override.yaml", "spec: [\n")
		err := run([]string{
			"--blueprint", blueprintPath,
			"--override", overridePath,
			"--output", filepath.Join(directory, "effective.yaml"),
		}, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "decode override YAML")
	})

	t.Run("DGDR resource instead of DGD override", func(t *testing.T) {
		directory := t.TempDir()
		blueprintPath := writeTestFile(t, directory, "blueprint.yaml", betaBlueprintYAML)
		overridePath := writeTestFile(t, directory, "dgdr.yaml", `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
spec: {}
`)
		err := run([]string{
			"--blueprint", blueprintPath,
			"--override", overridePath,
			"--output", filepath.Join(directory, "effective.yaml"),
		}, &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "apply DGD override")
		assert.Contains(t, err.Error(), `kind "DynamoGraphDeploymentRequest"`)
	})
}

func writeTestFile(t *testing.T, directory string, name string, content string) string {
	t.Helper()
	path := filepath.Join(directory, name)
	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
	return path
}

func mustReadDGD(t *testing.T, path string) *unstructured.Unstructured {
	t.Helper()
	object, err := readDGD(path, "test DGD")
	require.NoError(t, err)
	return object
}

func mainContainerImage(t *testing.T, dgd *unstructured.Unstructured) string {
	t.Helper()
	components, found, err := unstructured.NestedSlice(dgd.Object, "spec", "components")
	require.NoError(t, err)
	require.True(t, found)
	for _, value := range components {
		component, ok := value.(map[string]interface{})
		if !ok || component["name"] != "Worker" {
			continue
		}
		containers, found, err := unstructured.NestedSlice(component, "podTemplate", "spec", "containers")
		require.NoError(t, err)
		require.True(t, found)
		for _, containerValue := range containers {
			container, ok := containerValue.(map[string]interface{})
			if ok && container["name"] == "main" {
				image, ok := container["image"].(string)
				require.True(t, ok)
				return image
			}
		}
	}
	t.Fatal("main container not found")
	return ""
}

const betaBlueprintYAML = `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
  namespace: default
spec:
  backendFramework: vllm
  components:
  - name: Frontend
    type: frontend
  - name: Worker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          image: old-image
`

const alphaOverrideDGDJSON = `
{
  "apiVersion": "nvidia.com/v1alpha1",
  "kind": "DynamoGraphDeployment",
  "spec": {
    "services": {
      "Worker": {
        "extraPodSpec": {
          "mainContainer": {
            "image": "new-image"
          }
        }
      }
    }
  }
}
`
