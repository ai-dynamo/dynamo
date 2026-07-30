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
	"encoding/json"
	"testing"

	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// ffiBody builds the FFI JSON and parses it back into a map for assertions.
func ffiBody(t *testing.T, req *schedtypes.InferenceRequest) map[string]any {
	t.Helper()
	s, err := BuildOpenAIRequestJSON(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequestJSON returned error: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(s), &body); err != nil {
		t.Fatalf("failed to unmarshal FFI JSON: %v (json=%s)", err, s)
	}
	return body
}

// TestBuildOpenAIRequestJSON_ForwardsAgentHintsPriority pins the contract that
// nvext.agent_hints.priority on the original request body is preserved in the
// JSON sent across FFI to the Rust router. Without it, the router falls back to
// priority_jump=0.0 for every request and queue ordering silently regresses.
func TestBuildOpenAIRequestJSON_ForwardsAgentHintsPriority(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{
				"messages": []any{map[string]any{"role": "user", "content": "hi"}},
				"model":    "test-model",
				"nvext":    map[string]any{"agent_hints": map[string]any{"priority": 7}},
			},
		},
	}

	body := ffiBody(t, req)
	nvext, ok := body["nvext"].(map[string]any)
	if !ok {
		t.Fatalf("expected nvext to be a map, got %T", body["nvext"])
	}
	hints, ok := nvext["agent_hints"].(map[string]any)
	if !ok {
		t.Fatalf("expected agent_hints to be a map, got %T", nvext["agent_hints"])
	}
	if got := hints["priority"]; got != float64(7) { // JSON numbers decode to float64
		t.Fatalf("expected priority=7 forwarded to FFI body, got %v (%T)", got, got)
	}
}

// TestBuildOpenAIRequestJSON_ForwardsLegacyTopLevelCacheSalt verifies a
// top-level cache_salt on the request body is forwarded to the router.
func TestBuildOpenAIRequestJSON_ForwardsLegacyTopLevelCacheSalt(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{
				"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
				"model":      "test-model",
				"cache_salt": "tenant-legacy",
			},
		},
	}

	body := ffiBody(t, req)
	if got := body["cache_salt"]; got != "tenant-legacy" {
		t.Fatalf("expected legacy cache_salt forwarded to FFI body, got %v", got)
	}
}

// TestBuildOpenAIRequestJSON_PreservesToolCallFields pins the contract that a
// multi-turn tool conversation survives intact in the JSON sent across FFI. The
// router parses this body and re-renders the model's chat template to tokenize,
// so it needs the full message structure. If tool_calls / tool_call_id are
// dropped, the router's strict parse fails ("missing field tool_call_id") and
// the request is unroutable (503 no healthy upstream); reasoning/tool parsing is
// also lost when the template renders an incomplete prompt.
func TestBuildOpenAIRequestJSON_PreservesToolCallFields(t *testing.T) {
	toolCall := map[string]any{
		"id":   "call-abc",
		"type": "function",
		"function": map[string]any{
			"name":      "get_current_weather",
			"arguments": `{"location":"Tokyo"}`,
		},
	}
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{
				"model": "alias-model",
				"messages": []any{
					map[string]any{"role": "user", "content": "weather in Tokyo?"},
					map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{toolCall}},
					map[string]any{"role": "tool", "tool_call_id": "call-abc", "content": "18C rain"},
				},
			},
		},
	}

	body := ffiBody(t, req)

	// Target model must override the caller's alias.
	if got := body["model"]; got != "test-model" {
		t.Fatalf("expected model=test-model, got %v", got)
	}

	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %#v", body["messages"])
	}

	// Assistant turn must retain tool_calls.
	assistant, ok := msgs[1].(map[string]any)
	if !ok {
		t.Fatalf("expected assistant message map, got %T", msgs[1])
	}
	if _, ok := assistant["tool_calls"].([]any); !ok {
		t.Fatalf("assistant message lost tool_calls: %#v", assistant)
	}

	// Tool turn must retain tool_call_id (the field whose loss caused the 503).
	tool, ok := msgs[2].(map[string]any)
	if !ok {
		t.Fatalf("expected tool message map, got %T", msgs[2])
	}
	if got := tool["tool_call_id"]; got != "call-abc" {
		t.Fatalf("tool message lost tool_call_id: got %v in %#v", got, tool)
	}
}

// TestBuildOpenAIRequestJSON_ForwardsCompletionsPayload verifies a /v1/completions
// body is forwarded verbatim (the Rust preprocessor handles it via the prompt
// field), with only the model overridden to the resolved target.
func TestBuildOpenAIRequestJSON_ForwardsCompletionsPayload(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{
				"model":  "alias-model",
				"prompt": "hello world",
			},
		},
	}

	body := ffiBody(t, req)
	if got := body["prompt"]; got != "hello world" {
		t.Fatalf("expected prompt forwarded, got %v", got)
	}
	if got := body["model"]; got != "test-model" {
		t.Fatalf("expected model overridden to test-model, got %v", got)
	}
}

// TestBuildOpenAIRequestJSON_MissingPayloadReturnsError verifies that when the
// raw payload is unavailable the request is not KV-routable: an error is
// returned so the scorer falls back to non-KV routing rather than a lossy
// role/content reconstruction (which would drop tool-calling fields).
func TestBuildOpenAIRequestJSON_MissingPayloadReturnsError(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "user", Content: fwkrh.Content{Raw: "hi"}},
				},
			},
			// No Payload set — the typed view cannot carry tool-calling fields.
		},
	}

	if _, err := BuildOpenAIRequestJSON(req); err == nil {
		t.Fatalf("expected an error when the raw payload is unavailable, got nil")
	}
}
