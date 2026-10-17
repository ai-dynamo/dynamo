// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package golden

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestReadManifestsPreservesHistoricalScalars(t *testing.T) {
	t.Log("Keep NIXL's literal n value and integer fields across unstructured snapshot loading")
	path := filepath.Join(t.TempDir(), "input.yaml")
	require.NoError(t, os.WriteFile(path, []byte("apiVersion: example.com/v1\nkind: Record\nmetadata:\n  name: old\nspec:\n  telemetry: n\n  replicas: 2\n  historical: 'true'\n"), 0600))
	objects, err := ReadManifests(path)
	require.NoError(t, err)
	require.Len(t, objects, 1)
	object := objects[0].DeepCopy()
	value, _, err := unstructured.NestedString(object.Object, "spec", "telemetry")
	require.NoError(t, err)
	require.Equal(t, "n", value)
	replicas, _, err := unstructured.NestedInt64(object.Object, "spec", "replicas")
	require.NoError(t, err)
	require.EqualValues(t, 2, replicas)
	require.NoError(t, WriteManifests(path, objects))
	again, err := ReadManifests(path)
	require.NoError(t, err)
	require.Equal(t, objects, again)
}
