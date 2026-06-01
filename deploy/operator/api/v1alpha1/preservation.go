/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package v1alpha1

import (
	"encoding/json"
	"fmt"

	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

type preservedSpecEnvelope[T any] struct {
	Version int                `json:"version"`
	Spec    T                  `json:"spec"`
	RawJSON []preservedRawJSON `json:"rawJSON,omitempty"`
}

type preservedRawJSON struct {
	Path string `json:"path"`
	Raw  []byte `json:"raw,omitempty"`
	Nil  bool   `json:"nil,omitempty"`
}

func marshalPreservedSpec[T any](spec T, sanitize func(*T, *[]preservedRawJSON)) ([]byte, error) {
	envelope := preservedSpecEnvelope[T]{
		Version: 1,
		Spec:    spec,
	}
	if sanitize != nil {
		sanitize(&envelope.Spec, &envelope.RawJSON)
	}
	return json.Marshal(envelope)
}

func restorePreservedSpec[T any](raw string, apply func(*T, []preservedRawJSON)) (T, bool) {
	var zero T
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return zero, false
	}
	if _, ok := obj["spec"]; !ok {
		var legacy T
		if err := json.Unmarshal([]byte(raw), &legacy); err != nil {
			return zero, false
		}
		return legacy, true
	}

	var envelope preservedSpecEnvelope[T]
	if err := json.Unmarshal([]byte(raw), &envelope); err != nil {
		return zero, false
	}
	if apply != nil {
		apply(&envelope.Spec, envelope.RawJSON)
	}
	return envelope.Spec, true
}

func preserveEPPPluginParameters(config *apixv1alpha1.EndpointPickerConfig, pathPrefix string, records *[]preservedRawJSON) {
	if config == nil {
		return
	}
	for i := range config.Plugins {
		params := config.Plugins[i].Parameters
		if params == nil {
			*records = append(*records, preservedRawJSON{
				Path: fmt.Sprintf("%s/plugins/%d/parameters", pathPrefix, i),
				Nil:  true,
			})
		} else if !json.Valid(params) {
			*records = append(*records, preservedRawJSON{
				Path: fmt.Sprintf("%s/plugins/%d/parameters", pathPrefix, i),
				Raw:  append([]byte(nil), params...),
			})
			config.Plugins[i].Parameters = nil
		}
	}
}

func restoreEPPPluginParameters(config *apixv1alpha1.EndpointPickerConfig, pathPrefix string, records []preservedRawJSON) {
	if config == nil || len(records) == 0 {
		return
	}
	byPath := make(map[string]preservedRawJSON, len(records))
	for _, record := range records {
		byPath[record.Path] = record
	}
	for i := range config.Plugins {
		record, ok := byPath[fmt.Sprintf("%s/plugins/%d/parameters", pathPrefix, i)]
		if !ok {
			continue
		}
		if record.Nil {
			config.Plugins[i].Parameters = nil
		} else {
			config.Plugins[i].Parameters = append([]byte(nil), record.Raw...)
		}
	}
}
