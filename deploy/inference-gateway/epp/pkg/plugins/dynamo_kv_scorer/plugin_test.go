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
	"reflect"
	"testing"

	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestBuildOpenAIRequestJSON_ForwardPayload(t *testing.T) {
	toolCall := map[string]any{
		"id":   "call-abc",
		"type": "function",
		"function": map[string]any{
			"name":      "get_current_weather",
			"arguments": `{"location":"Tokyo"}`,
		},
	}
	payload := fwkrh.PayloadMap{
		"model":      "alias-model",
		"cache_salt": "tenant-legacy",
		"prompt":     "hello world",
		"nvext":      map[string]any{"agent_hints": map[string]any{"priority": 7}},
		"messages": []any{
			map[string]any{"role": "user", "content": "weather in Tokyo?"},
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{toolCall}},
			map[string]any{"role": "tool", "tool_call_id": "call-abc", "content": "18C rain"},
		},
	}
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body:        &fwkrh.InferenceRequestBody{Payload: payload},
	}

	body, err := BuildOpenAIRequestJSON(req)
	if err != nil {
		t.Fatalf("failed to build OpenAI request JSON: %v", err)
	}

	var want map[string]any
	if err := json.Unmarshal([]byte(body), &want); err != nil {
		t.Fatalf("failed to unmarshal body: %v", err)
	}
	if want["model"] != "test-model" {
		t.Fatalf("model = %v, want %v", want["model"], "test-model")
	}
	if !reflect.DeepEqual(payload, want) {
		t.Fatalf("payload not forwarded verbatim:\n got  %#v\n want %#v", body, want)
	}

}

// TestBuildOpenAIRequestJSON_ResolvesModel verifies model resolution: a non-empty
// (non-whitespace) TargetModel overrides the payload's model, otherwise the
// payload's own model is preserved.
func TestBuildOpenAIRequestJSON_ResolvesModel(t *testing.T) {
	tests := []struct {
		name        string
		targetModel string
		wantModel   string
	}{
		{"target overrides payload model", "test-model", "test-model"},
		{"empty target keeps payload model", "", "alias-model"},
		{"whitespace target keeps payload model", "   ", "alias-model"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &schedtypes.InferenceRequest{
				TargetModel: tc.targetModel,
				Body: &fwkrh.InferenceRequestBody{
					Payload: fwkrh.PayloadMap{
						"model":    "alias-model",
						"messages": []any{map[string]any{"role": "user", "content": "hi"}},
					},
				},
			}
			body, err := BuildOpenAIRequestJSON(req)
			if err != nil {
				t.Fatalf("failed to build OpenAI request JSON: %v", err)
			}
			var want map[string]any
			if err := json.Unmarshal([]byte(body), &want); err != nil {
				t.Fatalf("failed to unmarshal body: %v", err)
			}
			if got := want["model"]; got != tc.wantModel {
				t.Fatalf("model = %v, want %v", got, tc.wantModel)
			}
		})
	}
}

func TestBuildOpenAIRequestJSON_MissingPayloadReturnsError(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "user", Content: fwkrh.Content{Raw: "hi"}},
				},
			},
		},
	}

	if _, err := BuildOpenAIRequestJSON(req); err == nil {
		t.Fatalf("expected an error when the raw payload is unavailable, got nil")
	}
}
