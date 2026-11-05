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
	"bytes"
	"strings"

	yamlv3 "go.yaml.in/yaml/v3"
	"sigs.k8s.io/yaml"
)

// MarshalInput encodes a lowered input as the YAML file AISimulate loads with PyYAML's safe_load.
// PyYAML follows YAML 1.1, whose float grammar needs a "." in the mantissa, so it reads Go's
// shortest float text such as 1e-06 or 1e+21 as a string. MarshalInput writes those floats as
// 1.0e-06 and 1.0e+21 and otherwise keeps the sigs.k8s.io/yaml encoding, including integer precision.
func MarshalInput(input map[string]any) ([]byte, error) {
	// Encode with sigs.k8s.io/yaml so key order, json.Number values, and integers stay as before.
	contents, err := yaml.Marshal(input)
	if err != nil {
		return nil, err
	}

	// Rewrite only plain float scalars; keep the original text when none needs a mantissa ".".
	var doc yamlv3.Node
	if err := yamlv3.Unmarshal(contents, &doc); err != nil {
		return nil, err
	}
	if !yaml11Floats(&doc) {
		return contents, nil
	}

	// Re-encode the patched tree with a 2-space indent.
	var out bytes.Buffer
	encoder := yamlv3.NewEncoder(&out)
	encoder.SetIndent(2)
	if err := encoder.Encode(&doc); err != nil {
		return nil, err
	}
	if err := encoder.Close(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// yaml11Floats rewrites plain float scalars whose mantissa has no "." and reports whether it changed any.
func yaml11Floats(node *yamlv3.Node) bool {
	changed := false
	// Quoted scalars are strings even when they look like 1e-06, so only plain floats change.
	if node.Kind == yamlv3.ScalarNode && node.Style == 0 && node.ShortTag() == "!!float" {
		if i := strings.IndexAny(node.Value, "eE"); i > 0 && !strings.Contains(node.Value[:i], ".") {
			node.Value = node.Value[:i] + ".0" + node.Value[i:]
			changed = true
		}
	}
	for _, child := range node.Content {
		if yaml11Floats(child) {
			changed = true
		}
	}
	return changed
}
