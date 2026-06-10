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

package dynamo_kv_scorer

import (
	"testing"

	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// TestBuildOpenAIRequest_ForwardsAgentHintsPriority pins the contract that
// nvext.agent_hints.priority arriving on the original request body is
// preserved in the JSON sent across FFI to the Rust router. Without this,
// the router falls back to priority_jump=0.0 for every request and queue
// ordering silently regresses.
func TestBuildOpenAIRequest_ForwardsAgentHintsPriority(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "user", Content: fwkrh.Content{Raw: "hi"}},
				},
			},
			Payload: fwkrh.PayloadMap{
				"messages": []any{map[string]any{"role": "user", "content": "hi"}},
				"model":    "test-model",
				"nvext":    map[string]any{"agent_hints": map[string]any{"priority": 7}},
			},
		},
	}

	body, err := BuildOpenAIRequest(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequest returned error: %v", err)
	}

	nvext, ok := body["nvext"].(map[string]any)
	if !ok {
		t.Fatalf("expected nvext to be a map, got %T", body["nvext"])
	}
	hints, ok := nvext["agent_hints"].(map[string]any)
	if !ok {
		t.Fatalf("expected agent_hints to be a map, got %T", nvext["agent_hints"])
	}
	if got := hints["priority"]; got != 7 {
		t.Fatalf("expected priority=7 forwarded to FFI body, got %v", got)
	}
}
