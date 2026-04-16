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
	"os"
	"strings"

	"github.com/go-logr/logr"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	PrefillProfileName = "prefill"
	DecodeProfileName  = "decode"

	// PrefillEnabledStateKey tracks whether this request should use disaggregated routing.
	PrefillEnabledStateKey = plugins.StateKey("disagg-prefill-enabled")
)

// PrefillEnabledState stores whether prefill is enabled for the current scheduling cycle.
type PrefillEnabledState struct {
	Enabled bool
}

// Clone implements plugins.StateData.
func (s *PrefillEnabledState) Clone() plugins.StateData {
	return &PrefillEnabledState{Enabled: s.Enabled}
}

// readPrefillEnabled reads the PrefillEnabledState from CycleState.
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

// serializeEndpoints converts endpoints to a JSON string for the FFI filter.
func serializeEndpoints(endpoints []schedtypes.Endpoint) string {
	if len(endpoints) == 0 {
		return ""
	}
	pj, err := dynscorer.SerializeEndpointsToJSON(endpoints)
	if err != nil {
		return ""
	}
	return pj
}

// uniformScores returns a score map with the same score for every endpoint.
func uniformScores(endpoints []schedtypes.Endpoint, score float64) map[schedtypes.Endpoint]float64 {
	out := make(map[schedtypes.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		out[ep] = score
	}
	return out
}

// setTokenizedPrompt stores pre-computed token IDs on the LLMRequest and
// injects nvext.token_data into the PayloadMap so it is forwarded to the
// worker in the request body.
//
// NOTE: The GAIE ext-proc framework serializes RawBody before scheduling
// plugins run. For the PayloadMap mutation to reach the worker, the framework
// must re-serialize the PayloadMap after scheduling — see
// https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/XXXX
func setTokenizedPrompt(req *schedtypes.LLMRequest, tokens []int64, logger logr.Logger) {
	if req == nil || len(tokens) == 0 {
		logger.V(logutil.DEFAULT).Info("[EPP-INJECT] No tokens to inject (empty token list)")
		return
	}

	tokenIDs := make([]uint32, len(tokens))
	for i, t := range tokens {
		tokenIDs[i] = uint32(t)
	}

	req.TokenizedPrompt = &schedtypes.TokenizedPrompt{
		TokenIDs: tokenIDs,
	}

	// Inject into the PayloadMap so the body includes nvext.token_data.
	payloadInjected := false
	if req.Body != nil {
		if pm, ok := req.Body.Payload.(schedtypes.PayloadMap); ok {
			nvext, _ := pm["nvext"].(map[string]any)
			if nvext == nil {
				nvext = map[string]any{}
			}
			nvext["token_data"] = tokenIDs
			pm["nvext"] = nvext
			payloadInjected = true
		}
	}

	logger.V(logutil.DEFAULT).Info("[EPP-INJECT] Injected pre-computed tokens into request body nvext.token_data",
		"tokenCount", len(tokenIDs),
		"payloadInjected", payloadInjected,
		"requestId", req.RequestId)
}

func getEnvBoolOrDefault(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		switch strings.ToLower(v) {
		case "true", "1", "yes", "on":
			return true
		case "false", "0", "no", "off":
			return false
		}
	}
	return def
}
