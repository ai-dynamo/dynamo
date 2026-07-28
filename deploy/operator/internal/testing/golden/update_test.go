/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package golden

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestAdaptNodePreservesCommentsAndStrictness(t *testing.T) {
	t.Log("Build a commented contract with a strict child and a non-strict metadata mapping")
	expected := testNode(t, `
$strict: false
spec:
  # replicas explain the contract
  replicas: 1 # keep this comment
  removed: old
metadata:
  $strict: false
  name: example
  selected: old # selected is intentionally tracked
`)
	actual := testNode(t, `
spec:
  replicas: 2
  added: actual
metadata:
  name: example
`)

	t.Log("Adapt strict fields to reality and mark absent selected non-strict fields explicitly")
	adapted := adaptNode(cloneNode(expected), actual)
	encoded, err := yaml.Marshal(adapted)
	if err != nil {
		t.Fatalf("encode adapted YAML: %v", err)
	}
	text := string(encoded)
	for _, expectedText := range []string{
		"replicas: 2 # keep this comment",
		"# replicas explain the contract",
		"added: actual",
		"selected: $notexists # selected is intentionally tracked",
	} {
		if !strings.Contains(text, expectedText) {
			t.Errorf("adapted YAML does not contain %q:\n%s", expectedText, text)
		}
	}
	if strings.Contains(text, "removed:") {
		t.Errorf("adapted strict mapping retained an absent field:\n%s", text)
	}

	t.Log("Verify the adapted contract now matches the actual object")
	if err := matchNode(adapted, actual, "$"); err != nil {
		t.Fatalf("adapted contract does not match: %v\n%s", err, text)
	}
}

func TestWriteAdaptedWritesEmptyFileWithoutActualObjects(t *testing.T) {
	t.Log("Create a comparison whose expected object has no actual counterpart")
	expected := readTestDocuments(t, "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: absent\n")
	path := filepath.Join(t.TempDir(), "expected.yaml.new")

	t.Log("Write the minimally adapted contract with the absent document removed")
	if err := writeAdapted(path, comparison{expected: expected, actual: map[schema.GroupVersionKind][]actualDocument{}}); err != nil {
		t.Fatalf("writeAdapted(): %v", err)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read adapted file: %v", err)
	}
	if len(contents) != 0 {
		t.Fatalf("adapted file = %q, want empty", contents)
	}
}
