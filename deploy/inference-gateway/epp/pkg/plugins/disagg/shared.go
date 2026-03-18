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
	"encoding/json"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	PrefillProfileName = "prefill"
	DecodeProfileName  = "decode"

	// PrefillEnabledStateKey is used to communicate prefill-enabled status
	// from the DisaggProfileHandler to the scorer plugins via CycleState.
	PrefillEnabledStateKey = plugins.StateKey("disagg-prefill-enabled")

	// PrefillWorkerIDStateKey communicates the prefill worker ID selected by
	// DynPrefillScorer to DynDecodeScorer so it can set the x-prefill-instance-id header.
	PrefillWorkerIDStateKey = plugins.StateKey("disagg-prefill-worker-id")
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

// PrefillWorkerIDState stores the prefill worker ID selected by DynPrefillScorer.
// Written by DynPrefillScorer, read by DynDecodeScorer to set the header.
type PrefillWorkerIDState struct {
	WorkerID string
}

// Clone implements plugins.StateData.
func (s *PrefillWorkerIDState) Clone() plugins.StateData {
	return &PrefillWorkerIDState{WorkerID: s.WorkerID}
}

// readPrefillEnabled reads the PrefillEnabledState from CycleState.
// Returns false if the state is not found or not set.
func readPrefillEnabled(cycleState *schedtypes.CycleState) bool {
	state, err := schedtypes.ReadCycleStateKey[*PrefillEnabledState](cycleState, PrefillEnabledStateKey)
	if err == nil && state != nil {
		return state.Enabled
	}
	return false
}

// buildRequestJSON builds an OpenAI-compatible JSON string from a GAIE LLMRequest.
func buildRequestJSON(req *schedtypes.LLMRequest) (string, error) {
	requestBody, err := dynscorer.BuildOpenAIRequest(req)
	if err != nil {
		return "", fmt.Errorf("failed to build OpenAI request: %w", err)
	}
	data, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request JSON: %w", err)
	}
	return string(data), nil
}

// serializePods converts pods to a JSON string for the FFI filter.
// Returns an empty string if serialization fails or pods is empty.
func serializePods(pods []schedtypes.Pod) string {
	if len(pods) == 0 {
		return ""
	}
	pj, err := dynscorer.SerializePodsToJSON(pods)
	if err != nil {
		return ""
	}
	return pj
}

// uniformScores returns a score map with the same score for every pod.
func uniformScores(pods []schedtypes.Pod, score float64) map[schedtypes.Pod]float64 {
	out := make(map[schedtypes.Pod]float64, len(pods))
	for _, p := range pods {
		out[p] = score
	}
	return out
}
