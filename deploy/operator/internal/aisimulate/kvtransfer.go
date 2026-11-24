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
	"fmt"
	"slices"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// modeDisaggregated is the AISimulate engine mode whose candidates move KV from prefill to decode.
const modeDisaggregated = "disaggregated"

// defaultEngineModes is AISimulate's recommendation default when engine.mode is absent.
var defaultEngineModes = []string{"aggregated", modeDisaggregated}

// KVTransfer describes how an effective recommendation input models the prefill-to-decode KV
// transfer.
type KVTransfer struct {
	// SearchesDisaggregated reports whether disaggregated candidates can be searched.
	SearchesDisaggregated bool
	// Configured reports whether engine.kv_transfer is present.
	Configured bool
	// HasBandwidth reports whether engine.kv_transfer.bandwidth_gb_per_second is set. Without
	// it, AISimulate models the transfer as instantaneous.
	HasBandwidth bool
}

// InspectKVTransfer reads the KV-transfer model of an effective recommendation input, such as
// the result of Lower. input must not be nil. It returns an error when engine.mode is neither a
// string nor a {choices: [...]} object of strings; schema validation reports the same input.
func InspectKVTransfer(input map[string]any) (KVTransfer, error) {
	modes, err := engineModes(input)
	if err != nil {
		return KVTransfer{}, err
	}
	bandwidth, _, _ := unstructured.NestedFieldNoCopy(input, engineKVTransferBW...)
	return KVTransfer{
		SearchesDisaggregated: slices.Contains(modes, modeDisaggregated),
		Configured:            has(input, engineKVTransfer...),
		HasBandwidth:          bandwidth != nil,
	}, nil
}

// engineModes returns the engine modes the input searches, applying AISimulate's default when
// engine.mode is absent or null.
func engineModes(input map[string]any) ([]string, error) {
	value, _, _ := unstructured.NestedFieldNoCopy(input, engineMode...)
	switch mode := value.(type) {
	case nil:
		return defaultEngineModes, nil
	case string:
		return []string{mode}, nil
	case map[string]any:
		choices, found, err := unstructured.NestedStringSlice(mode, "choices")
		if err != nil || !found {
			return nil, fmt.Errorf("engine.mode: choices must be a list of strings")
		}
		return choices, nil
	default:
		return nil, fmt.Errorf("engine.mode: must be a string or a choices object, got %T", value)
	}
}
