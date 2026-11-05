/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package aisimulate

import (
	"encoding/json"
	"regexp"
	"testing"

	yamlv3 "go.yaml.in/yaml/v3"
	"sigs.k8s.io/yaml"
)

// pyYAMLFloat is PyYAML's YAML 1.1 implicit float pattern (yaml/resolver.py in PyYAML 6.0.3).
// AISimulate loads input.yaml with yaml.safe_load, so a float that does not match arrives as a string.
var pyYAMLFloat = regexp.MustCompile(`^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?` +
	`|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?` +
	`|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*` +
	`|[-+]?\.(?:inf|Inf|INF)` +
	`|\.(?:nan|NaN|NAN))$`)

// TestMarshalInputWritesFloatsPyYAMLReads checks that PyYAML recognizes emitted floats without changing other values.
func TestMarshalInputWritesFloatsPyYAMLReads(t *testing.T) {
	tests := []struct {
		key     string
		value   any
		wantTag string
		want    any
	}{
		{key: "tiny", value: 0.000001, wantTag: "!!float", want: 1e-6},
		{key: "tinier", value: 1e-7, wantTag: "!!float", want: 1e-7},
		{key: "huge", value: 1e21, wantTag: "!!float", want: 1e21},
		{key: "fractional_mantissa", value: 1.5e-5, wantTag: "!!float", want: 1.5e-5},
		{key: "fixed_point", value: 0.001, wantTag: "!!float", want: 0.001},
		{key: "override_number", value: json.Number("1e-06"), wantTag: "!!float", want: 1e-6},
		{key: "whole_float", value: 2000.0, wantTag: "!!int", want: int64(2000)},
		{key: "large_integer", value: json.Number("9007199254740993"), wantTag: "!!int", want: "9007199254740993"},
		{key: "string", value: "1e-06", wantTag: "!!str", want: "1e-06"},
	}

	t.Log("Encode one input that holds every case")
	input := map[string]any{}
	for _, tt := range tests {
		input[tt.key] = tt.value
	}
	contents, err := MarshalInput(input)
	if err != nil {
		t.Fatalf("MarshalInput: %v", err)
	}
	var doc yamlv3.Node
	if err := yamlv3.Unmarshal(contents, &doc); err != nil {
		t.Fatalf("parse output: %v\n%s", err, contents)
	}
	var decoded map[string]any
	if err := yaml.Unmarshal(contents, &decoded); err != nil {
		t.Fatalf("decode output: %v\n%s", err, contents)
	}
	scalars := map[string]*yamlv3.Node{}
	for i := 0; i+1 < len(doc.Content[0].Content); i += 2 {
		scalars[doc.Content[0].Content[i].Value] = doc.Content[0].Content[i+1]
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			t.Log("The scalar keeps its type, and floats use text PyYAML resolves as a float")
			node := scalars[tt.key]
			if node == nil || node.ShortTag() != tt.wantTag {
				t.Fatalf("%s = %+v, want a %s scalar:\n%s", tt.key, node, tt.wantTag, contents)
			}
			if tt.wantTag == "!!float" && !pyYAMLFloat.MatchString(node.Value) {
				t.Fatalf("%s = %q, which PyYAML reads as a string", tt.key, node.Value)
			}

			t.Log("Go readers decode the same value")
			switch want := tt.want.(type) {
			case string:
				if node.Value != want {
					t.Fatalf("%s text = %q, want %q", tt.key, node.Value, want)
				}
			case int64:
				if decoded[tt.key] != float64(want) {
					t.Fatalf("%s decoded = %v, want %v", tt.key, decoded[tt.key], want)
				}
			default:
				if decoded[tt.key] != want {
					t.Fatalf("%s decoded = %v, want %v", tt.key, decoded[tt.key], want)
				}
			}
		})
	}
}
