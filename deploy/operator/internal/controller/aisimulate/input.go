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
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// BuildInput returns the effective native input: the underlay with the derived defaults added,
// then the parameters applied on top as a JSON Merge Patch (RFC 7386). The underlay is the input
// the caller derived from its own API; it is not modified. parameters is raw JSON: empty, null or an
// object. The only errors are parameters that are not a JSON object; semantic problems are the
// caller's to validate.
func BuildInput(underlay map[string]any, parameters []byte) (map[string]any, error) {
	patch, err := decodeParameters(parameters)
	if err != nil {
		return nil, err
	}
	base, err := normalize(underlay)
	if err != nil {
		return nil, err
	}
	if base == nil {
		base = map[string]any{}
	}
	if err := applyDefaults(base); err != nil {
		return nil, err
	}
	input, _ := mergePatch(base, patch).(map[string]any)
	if input == nil {
		input = map[string]any{}
	}
	return input, nil
}

// DecodeInput decodes a complete native input as it was stored, for example the one frozen into a
// Run. It lowers and defaults nothing: the stored object is the input.
func DecodeInput(raw []byte) (map[string]any, error) {
	if len(raw) == 0 {
		return nil, errors.New("the input is empty")
	}
	var input map[string]any
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("the input must be a JSON object: %w", err)
	}
	if len(input) == 0 {
		return nil, errors.New("the input is empty")
	}
	return input, nil
}

// decodeParameters decodes the parameters as a JSON object, or nil when absent or null.
func decodeParameters(raw []byte) (map[string]any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("parameters: %w", err)
	}
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case map[string]any:
		return typed, nil
	default:
		return nil, fmt.Errorf("parameters: must be a JSON object, got %T", value)
	}
}

// builder writes native input values, remembering the first failure so a list of assignments
// reads straight. Every path is written under a map the caller created and every value is a JSON
// scalar, so a failure means a bad path.
type builder struct {
	root map[string]any
	err  error
}

func (b *builder) set(value any, path []string) {
	if b.err != nil {
		return
	}
	if err := unstructured.SetNestedField(b.root, value, path...); err != nil {
		b.err = fmt.Errorf("set %s: %w", strings.Join(path, "."), err)
	}
}

// normalize round-trips through JSON so every number is a float64, the type the
// parameters decode to. Merge and validation then see one representation.
func normalize(value map[string]any) (map[string]any, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// mergePatch applies patch to target with RFC 7386 semantics. It returns a new value for
// the merged parts and does not retain the patch's maps for null members.
func mergePatch(target, patch any) any {
	patchMap, ok := patch.(map[string]any)
	if !ok {
		return patch
	}
	targetMap, ok := target.(map[string]any)
	if !ok {
		targetMap = map[string]any{}
	}
	for key, value := range patchMap {
		if value == nil {
			delete(targetMap, key)
			continue
		}
		targetMap[key] = mergePatch(targetMap[key], value)
	}
	return targetMap
}
