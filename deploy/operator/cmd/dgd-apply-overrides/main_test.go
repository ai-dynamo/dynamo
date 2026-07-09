/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestRunAppliesVersionedOverride(t *testing.T) {
	t.Parallel()

	stderr := &bytes.Buffer{}
	stdout := &bytes.Buffer{}

	err := run(nil, applyRequestReader(t, betaBlueprintJSON, alphaOverrideDGDJSON), stdout, stderr)
	require.NoError(t, err)
	assert.Empty(t, stderr.String())

	effective := mustDecodeDGD(t, stdout.Bytes())
	assert.Equal(t, "nvidia.com/v1beta1", effective.GetAPIVersion())
	assert.Equal(t, "DynamoGraphDeployment", effective.GetKind())
	assert.Equal(t, "generated", effective.GetName())
	assert.Equal(t, "new-image", mainContainerImage(t, effective))
}

func TestRunWithEmptyOverridePreservesBlueprint(t *testing.T) {
	t.Parallel()

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}

	err := run(
		nil,
		applyRequestReader(t, betaBlueprintJSON, `{
			"apiVersion": "nvidia.com/v1beta1",
			"kind": "DynamoGraphDeployment"
		}`),
		stdout,
		stderr,
	)
	require.NoError(t, err)
	assert.Empty(t, stderr.String())
	assert.Equal(t, mustDecodeDGD(t, []byte(betaBlueprintJSON)), mustDecodeDGD(t, stdout.Bytes()))
}

func TestRunEmitsWarningsForIgnoredTopology(t *testing.T) {
	t.Parallel()

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}

	err := run(
		nil,
		applyRequestReader(t, betaBlueprintJSON, `{
			"apiVersion": "nvidia.com/v1beta1",
			"kind": "DynamoGraphDeployment",
			"spec": {"components": [{"name": "Missing", "replicas": 2}]}
		}`),
		stdout,
		stderr,
	)
	require.NoError(t, err)
	assert.Contains(
		t,
		stderr.String(),
		"warning: spec.components[name=Missing]: ignored because the generated blueprint has no such component",
	)
	assert.Equal(t, "old-image", mainContainerImage(t, mustDecodeDGD(t, stdout.Bytes())))
}

func TestRunDoesNotWriteOutputOnFailure(t *testing.T) {
	t.Parallel()

	stdout := &bytes.Buffer{}
	err := run(
		nil,
		applyRequestReader(t, betaBlueprintJSON, `{
			"apiVersion": "nvidia.com/v2",
			"kind": "DynamoGraphDeployment"
		}`),
		stdout,
		&bytes.Buffer{},
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "apply DGD override")
	assert.Empty(t, stdout.String())
}

func TestRunInstallsExecutable(t *testing.T) {
	t.Parallel()

	installPath := filepath.Join(t.TempDir(), "dgd-apply-overrides")
	stderr := &bytes.Buffer{}
	require.NoError(t, run([]string{"--install-to", installPath}, strings.NewReader(""), &bytes.Buffer{}, stderr))
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
	require.NoError(t, run([]string{"--protocol-version"}, strings.NewReader(""), stdout, stderr))
	assert.Equal(t, protocolVersion+"\n", stdout.String())
	assert.Empty(t, stderr.String())
}

func TestRunRejectsInvalidArgumentsAndInput(t *testing.T) {
	t.Parallel()

	t.Run("empty request", func(t *testing.T) {
		err := run(nil, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "decode request JSON")
	})

	t.Run("unexpected positional argument", func(t *testing.T) {
		err := run([]string{"extra"}, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unexpected positional arguments")
	})

	t.Run("removed file flag", func(t *testing.T) {
		err := run([]string{"--blueprint", "/tmp/blueprint"}, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "flag provided but not defined: -blueprint")
	})

	t.Run("protocol version mixed with install mode", func(t *testing.T) {
		err := run(
			[]string{"--protocol-version", "--install-to", "/tmp/tool"},
			strings.NewReader(""),
			&bytes.Buffer{},
			&bytes.Buffer{},
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "--install-to and --protocol-version are mutually exclusive")
	})

	t.Run("malformed request", func(t *testing.T) {
		err := run(nil, strings.NewReader("{"), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "decode request JSON")
	})

	t.Run("missing blueprint", func(t *testing.T) {
		err := run(nil, strings.NewReader(`{"override": {}}`), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "request blueprint is required")
	})

	t.Run("missing override", func(t *testing.T) {
		err := run(nil, strings.NewReader(`{"blueprint": {}}`), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "request override is required")
	})

	t.Run("unknown request field", func(t *testing.T) {
		err := run(nil, strings.NewReader(`{"blueprint": {}, "override": {}, "extra": {}}`), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), `unknown field "extra"`)
	})

	t.Run("multiple request values", func(t *testing.T) {
		input := `{"blueprint":` + betaBlueprintJSON + `,"override":` + alphaOverrideDGDJSON + `}{}`
		err := run(nil, strings.NewReader(input), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "multiple JSON values")
	})

	t.Run("non-object blueprint", func(t *testing.T) {
		err := run(nil, applyRequestReader(t, `"not-an-object"`, alphaOverrideDGDJSON), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "decode request blueprint object")
	})

	t.Run("DGDR resource instead of DGD override", func(t *testing.T) {
		err := run(nil, applyRequestReader(t, betaBlueprintJSON, `{
			"apiVersion": "nvidia.com/v1beta1",
			"kind": "DynamoGraphDeploymentRequest",
			"spec": {}
		}`), &bytes.Buffer{}, &bytes.Buffer{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "apply DGD override")
		assert.Contains(t, err.Error(), `kind "DynamoGraphDeploymentRequest"`)
	})
}

func applyRequestReader(t *testing.T, blueprint string, override string) *bytes.Reader {
	t.Helper()
	request, err := json.Marshal(applyRequest{
		Blueprint: json.RawMessage(blueprint),
		Override:  json.RawMessage(override),
	})
	require.NoError(t, err)
	return bytes.NewReader(request)
}

func mustDecodeDGD(t *testing.T, data []byte) *unstructured.Unstructured {
	t.Helper()
	object, err := decodeDGD(data, "test DGD")
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

const betaBlueprintJSON = `
{
  "apiVersion": "nvidia.com/v1beta1",
  "kind": "DynamoGraphDeployment",
  "metadata": {
    "name": "generated",
    "namespace": "default"
  },
  "spec": {
    "backendFramework": "vllm",
    "components": [
      {
        "name": "Frontend",
        "type": "frontend"
      },
      {
        "name": "Worker",
        "type": "worker",
        "podTemplate": {
          "spec": {
            "containers": [
              {
                "name": "main",
                "image": "old-image"
              }
            ]
          }
        }
      }
    ]
  }
}
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
