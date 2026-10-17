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

	t.Log("Omit field-manager bookkeeping without mutating the observed object")
	require.NoError(t, unstructured.SetNestedSlice(objects[0].Object, []any{
		map[string]any{"manager": "historical-operator", "operation": "Update"},
	}, "metadata", "managedFields"))
	require.NoError(t, WriteManifests(path, objects))
	_, found, err := unstructured.NestedSlice(objects[0].Object, "metadata", "managedFields")
	require.NoError(t, err)
	require.True(t, found, "export must not mutate its input")
	again, err := ReadManifests(path)
	require.NoError(t, err)
	require.Equal(t, []unstructured.Unstructured{*object}, again, "only managedFields may be omitted")
}
