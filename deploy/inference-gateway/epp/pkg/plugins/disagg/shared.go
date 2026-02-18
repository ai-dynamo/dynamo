/*
Copyright 2025 NVIDIA Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package disagg implements disaggregated prefill/decode serving plugins for Dynamo EPP.
//
// The disaggregated architecture splits inference into two phases:
//   - Prefill: processes the input prompt (compute-heavy, parallelizable)
//   - Decode: generates tokens autoregressively (memory-bound, sequential)
//
// This package provides three plugins:
//   - DisaggProfileHandler: orchestrates prefill→decode profile execution
//   - DynPrefillScorer: selects prefill workers via Dynamo FFI
//   - DynDecodeScorer: selects decode workers via Dynamo FFI
package disagg

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

const (
	// Profile names
	PrefillProfileName = "prefill"
	DecodeProfileName  = "decode"

	// PrefillEnabledStateKey is used to communicate prefill-enabled status
	// from the DisaggProfileHandler to the scorer plugins via CycleState.
	PrefillEnabledStateKey = plugins.StateKey("disagg-prefill-enabled")
)

// PrefillEnabledState stores whether prefill is enabled for the current scheduling cycle.
// Written by DisaggProfileHandler, read by PrefillScorer and DecodeScorer.
type PrefillEnabledState struct {
	Enabled bool
}

// Clone implements plugins.StateData.
func (s *PrefillEnabledState) Clone() plugins.StateData {
	return &PrefillEnabledState{Enabled: s.Enabled}
}
