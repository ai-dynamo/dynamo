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
	"reflect"
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

func TestBuildOpenAIRequest_ForwardsLegacyTopLevelCacheSalt(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "user", Content: fwkrh.Content{Raw: "hi"}},
				},
			},
			Payload: fwkrh.PayloadMap{
				"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
				"model":      "test-model",
				"cache_salt": "tenant-legacy",
			},
		},
	}

	body, err := BuildOpenAIRequest(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequest returned error: %v", err)
	}
	if got := body["cache_salt"]; got != "tenant-legacy" {
		t.Fatalf("expected legacy cache_salt forwarded to FFI body, got %v", got)
	}
}

func TestBuildOpenAIRequest_CompletionsTokenPromptUsesPromptIDs(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{TokenIDs: []uint32{101, 102, 103}},
			},
		},
	}

	body, err := BuildOpenAIRequest(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequest returned error: %v", err)
	}

	if _, ok := body["messages"]; ok {
		t.Fatalf("did not expect token-id completions to synthesize messages: %v", body["messages"])
	}
	if got := body["prompt"]; !reflect.DeepEqual(got, []uint32{101, 102, 103}) {
		t.Fatalf("expected prompt token IDs, got %#v", got)
	}
	if got := body["model"]; got != "test-model" {
		t.Fatalf("expected model=test-model, got %v", got)
	}
}

func TestBuildOpenAIRequest_CompletionsTextPromptKeepsLegacyMessageShape(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "hello"},
			},
		},
	}

	body, err := BuildOpenAIRequest(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequest returned error: %v", err)
	}

	if _, ok := body["prompt"]; ok {
		t.Fatalf("did not expect text completions to change to prompt field: %v", body["prompt"])
	}
	messages, ok := body["messages"].([]map[string]any)
	if !ok {
		t.Fatalf("expected legacy messages shape, got %#v", body["messages"])
	}
	if len(messages) != 1 || messages[0]["role"] != "user" || messages[0]["content"] != "hello" {
		t.Fatalf("expected single user message with content=hello, got %#v", messages)
	}
}

func TestBuildOpenAIRequest_CompletionsStringArrayPromptKeepsLegacyMessageShape(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Strings: []string{"hello", "world"}},
			},
		},
	}

	body, err := BuildOpenAIRequest(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequest returned error: %v", err)
	}

	if _, ok := body["prompt"]; ok {
		t.Fatalf("did not expect string-array completions to change to prompt field: %v", body["prompt"])
	}
	messages, ok := body["messages"].([]map[string]any)
	if !ok {
		t.Fatalf("expected legacy messages shape, got %#v", body["messages"])
	}
	if len(messages) != 1 || messages[0]["role"] != "user" || messages[0]["content"] != "hello world" {
		t.Fatalf("expected single user message with content='hello world', got %#v", messages)
	}
}

// TestBuildOpenAIRequest_PreservesToolCallFields pins the contract that a
// multi-turn tool conversation survives intact in the JSON sent across FFI to
// the Rust router. The router parses this body and re-renders the model's chat
// template to tokenize, so it needs the full message structure. If tool_calls /
// tool_call_id are dropped, the router's strict parse fails with
// "missing field tool_call_id" and the request is unroutable (503 no healthy
// upstream); reasoning/tool parsing is also lost when the template renders an
// incomplete prompt.
func TestBuildOpenAIRequest_PreservesToolCallFields(t *testing.T) {
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
			// The typed view only carries role+content; the raw payload below is
			// the source of truth and must be forwarded verbatim.
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "user", Content: fwkrh.Content{Raw: "weather in Tokyo?"}},
					{Role: "assistant", Content: fwkrh.Content{Raw: ""}},
					{Role: "tool", Content: fwkrh.Content{Raw: "18C rain"}},
				},
			},
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

	body, err := BuildOpenAIRequest(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequest returned error: %v", err)
	}

	// Target model must override the caller's alias.
	if got := body["model"]; got != "test-model" {
		t.Fatalf("expected model=test-model, got %v", got)
	}

	msgs, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("expected messages []any from raw payload, got %T", body["messages"])
	}
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
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
