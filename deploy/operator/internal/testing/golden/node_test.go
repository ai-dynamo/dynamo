/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package golden

import (
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"
)

func TestNonStrictMappingDoesNotRelaxSpecifiedChildren(t *testing.T) {
	t.Log("Match unspecified top-level fields through a non-strict mapping")
	expected := testNode(t, `
$strict: false
selected:
  value: kept
`)
	actual := testNode(t, `
selected:
  value: kept
  extra: rejected
topLevelExtra: accepted
`)

	t.Log("Reject an extra field inside the specified child because it is strict again")
	err := matchNode(expected, actual, "$")
	if err == nil || !strings.Contains(err.Error(), "$.selected.extra: unexpected field") {
		t.Fatalf("matchNode() error = %v, want strict child mismatch", err)
	}

	t.Log("Allow the child field only when it has its own non-strict directive")
	expected = testNode(t, `
$strict: false
selected:
  $strict: false
  value: kept
`)
	if err := matchNode(expected, actual, "$"); err != nil {
		t.Fatalf("matchNode() with non-strict child: %v", err)
	}
}

func TestMatchDirectives(t *testing.T) {
	t.Log("Declare scalar, mapping, generated-name, and escaped-field expectations")
	expected := testNode(t, `
$strict: false
ignoredScalar: $ignore
existingScalar: $exists
absentScalar: $notexists
ignoredMapping:
  $ignore: true
absentMapping:
  $notexists: true
globName: $glob:request-*
patternName: $pattern:request-[0-9]+
$$strict: literal
`)
	actual := testNode(t, `
ignoredScalar: anything
existingScalar: 42
ignoredMapping:
  arbitrary: value
globName: request-generated
patternName: request-42
$strict: literal
`)

	t.Log("Match every directive without treating the escaped field as configuration")
	if err := matchNode(expected, actual, "$"); err != nil {
		t.Fatalf("matchNode(): %v", err)
	}
}

func TestStrictDirectiveRequiresBoolean(t *testing.T) {
	t.Log("Use a string where the structural strictness directive requires a boolean")
	expected := testNode(t, "$strict: \"false\"\n")
	actual := testNode(t, "field: value\n")

	t.Log("Report the invalid contract instead of silently changing its semantics")
	err := matchNode(expected, actual, "$")
	if err == nil || !strings.Contains(err.Error(), "$strict must be a boolean") {
		t.Fatalf("matchNode() error = %v, want invalid strict directive", err)
	}
}

func testNode(t *testing.T, manifest string) *yaml.Node {
	t.Helper()
	var document yaml.Node
	if err := yaml.Unmarshal([]byte(manifest), &document); err != nil {
		t.Fatalf("parse test YAML: %v", err)
	}
	return documentRoot(&document)
}
