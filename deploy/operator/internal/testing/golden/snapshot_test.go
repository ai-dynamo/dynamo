// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package golden

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
)

func TestReadManifestsPreservesHistoricalScalars(t *testing.T) {
	t.Log("Keep NIXL's literal n value and integer fields across unstructured snapshot loading")
	path := filepath.Join(t.TempDir(), "input.yaml")
	require.NoError(t, os.WriteFile(path, []byte("apiVersion: example.com/v1\nkind: Record\nmetadata:\n  name: old\nspec:\n  telemetry: n\n  literal: '$var:literal'\n  replicas: 2\n  historical: 'true'\n"), 0600))
	objects, err := ReadManifests(path)
	require.NoError(t, err)
	require.Len(t, objects, 1)
	object := objects[0].DeepCopy()
	value, _, err := unstructured.NestedString(object.Object, "spec", "telemetry")
	require.NoError(t, err)
	require.Equal(t, "n", value)
	literal, _, err := unstructured.NestedString(object.Object, "spec", "literal")
	require.NoError(t, err)
	require.Equal(t, "$var:literal", literal, "historical snapshot reads must not interpret golden directives")
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
	object.SetUID(again[0].GetUID())
	object.SetResourceVersion(again[0].GetResourceVersion())
	object.SetCreationTimestamp(again[0].GetCreationTimestamp())
	require.Equal(t, []unstructured.Unstructured{*object}, again, "only server metadata may be normalized")
}

func TestWriteManifestsNormalizesOnlyServerMetadata(t *testing.T) {
	t.Log("Record different API metadata without changing controller state or the observed object")
	object := unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "example.com/v1", "kind": "Record",
		"metadata": map[string]any{
			"name": "old", "namespace": "history", "generation": int64(7),
			"annotations": map[string]any{"origin": "1.4.0", "worker-hash": "92221e36", "last-applied-generation": "7"},
		},
		"spec":   map[string]any{"image": "runtime:fixed"},
		"status": map[string]any{"observedGeneration": int64(7), "lastTransitionTime": "2026-01-01T00:00:00Z"},
	}}
	original := object.DeepCopy()
	var baseline []byte
	for _, value := range []string{"first", "second"} {
		observed := object.DeepCopy()
		observed.SetUID(types.UID(value))
		observed.SetResourceVersion(value)
		observed.Object["metadata"].(map[string]any)["creationTimestamp"] = value
		observed.SetOwnerReferences([]metav1.OwnerReference{{APIVersion: "example.com/v1", Kind: "Record", Name: "parent", UID: types.UID(value)}})
		require.NoError(t, unstructured.SetNestedSlice(observed.Object, []any{map[string]any{"manager": value}}, "metadata", "managedFields"))
		before := observed.DeepCopy()
		path := filepath.Join(t.TempDir(), "input.yaml")
		require.NoError(t, WriteManifests(path, []unstructured.Unstructured{*observed}))
		require.Equal(t, before, observed, "recording must not mutate the live observations")
		data, err := os.ReadFile(path)
		require.NoError(t, err)
		if baseline != nil {
			require.Equal(t, baseline, data)
		}
		baseline = data

		t.Log("Read and rewrite the normalized input with the same formatting and concrete field types")
		again, err := ReadManifests(path)
		require.NoError(t, err)
		require.Equal(t, types.UID("00000000-0000-0000-0000-000000000000"), again[0].GetUID())
		require.Equal(t, "1", again[0].GetResourceVersion())
		require.Equal(t, "2000-01-01T00:00:00Z", again[0].Object["metadata"].(map[string]any)["creationTimestamp"])
		require.Equal(t, again[0].GetUID(), again[0].GetOwnerReferences()[0].UID)
		semantic := again[0].DeepCopy()
		for _, key := range []string{"uid", "resourceVersion", "creationTimestamp", "ownerReferences"} {
			unstructured.RemoveNestedField(semantic.Object, "metadata", key)
		}
		require.Equal(t, original, semantic, "generation, annotations, spec and status remain exact")
		require.NoError(t, WriteManifests(path, again))
		rewritten, err := os.ReadFile(path)
		require.NoError(t, err)
		require.Equal(t, data, rewritten)
	}
}

// Embed TB so matcher failures can be inspected without failing the containing test.
type capturedMatchReporter struct {
	testing.TB
	mismatches int
}

func (r *capturedMatchReporter) Errorf(string, ...any) { r.mismatches++ }

func TestMatchCapturedManifestsWithVariables(t *testing.T) {
	t.Log("Bind the current operator version exactly while keeping all other lowering fields exact")
	path := filepath.Join(t.TempDir(), "output.yaml")
	require.NoError(t, os.WriteFile(path, []byte(`apiVersion: example.com/v1
kind: Record
metadata:
  name: latest
  annotations:
    nvidia.com/dynamo-operator-origin-version: $var:operator-version
spec:
  image: runtime:fixed
`), 0600))
	object := unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "example.com/v1", "kind": "Record",
		"metadata": map[string]any{"name": "latest", "annotations": map[string]any{"nvidia.com/dynamo-operator-origin-version": "1.7.0"}},
		"spec":     map[string]any{"image": "runtime:fixed"},
	}}
	for _, test := range []struct {
		version, image string
		mismatch       bool
	}{
		{"1.7.0", "runtime:fixed", false},
		{"1.6.0", "runtime:fixed", true},
		{"1.7.0", "runtime:changed", true},
	} {
		reporter := &capturedMatchReporter{TB: t}
		actual := object.DeepCopy()
		require.NoError(t, unstructured.SetNestedField(actual.Object, test.image, "spec", "image"))
		MatchCapturedManifestsWithVariables(reporter, []unstructured.Unstructured{*actual}, path, Variables{"operator-version": test.version})
		require.Equal(t, test.mismatch, reporter.mismatches > 0)
	}
}
