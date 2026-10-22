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
	require.Equal(t, []unstructured.Unstructured{*object}, again, "only managedFields may be omitted")
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
