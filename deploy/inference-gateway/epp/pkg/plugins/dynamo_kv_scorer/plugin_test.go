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

	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// TestBuildOpenAIRequest_ForwardsAgentHintsPriority pins the contract that
// nvext.agent_hints.priority arriving on the original request body is
// preserved in the JSON sent across FFI to the Rust router. Without this,
// the router falls back to priority_jump=0.0 for every request and queue
// ordering silently regresses.
func TestBuildOpenAIRequest_ForwardsAgentHintsPriority(t *testing.T) {
	req := &schedtypes.LLMRequest{
		TargetModel: "test-model",
		Body: &schedtypes.LLMRequestBody{
			ChatCompletions: &schedtypes.ChatCompletionsRequest{
				Messages: []schedtypes.Message{
					{Role: "user", Content: schedtypes.Content{Raw: "hi"}},
				},
			},
			Payload: schedtypes.PayloadMap{
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
	req := &schedtypes.LLMRequest{
		TargetModel: "test-model",
		Body: &schedtypes.LLMRequestBody{
			ChatCompletions: &schedtypes.ChatCompletionsRequest{
				Messages: []schedtypes.Message{
					{Role: "user", Content: schedtypes.Content{Raw: "hi"}},
				},
			},
			Payload: schedtypes.PayloadMap{
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

func TestBuildOpenAIRequest_CompletionsTextPromptKeepsLegacyMessageShape(t *testing.T) {
	req := &schedtypes.LLMRequest{
		TargetModel: "test-model",
		Body: &schedtypes.LLMRequestBody{
			Completions: &schedtypes.CompletionsRequest{
				Prompt: schedtypes.Prompt{Raw: "hello"},
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

func TestBuildOpenAIRequest_TokenizedPromptUsesCompletionPromptShape(t *testing.T) {
	req := &schedtypes.LLMRequest{
		TargetModel: "test-model",
		TokenizedPrompt: &schedtypes.TokenizedPrompt{
			TokenIDs: []uint32{101, 102, 103},
		},
		Body: &schedtypes.LLMRequestBody{
			Completions: &schedtypes.CompletionsRequest{
				Prompt: schedtypes.Prompt{Raw: "hello"},
			},
		},
	}

	body, err := BuildOpenAIRequest(req)
	if err != nil {
		t.Fatalf("BuildOpenAIRequest returned error: %v", err)
	}

	prompt, ok := body["prompt"].([]uint32)
	if !ok {
		t.Fatalf("expected tokenized prompt shape, got %#v", body["prompt"])
	}
	if len(prompt) != 3 || prompt[0] != 101 || prompt[1] != 102 || prompt[2] != 103 {
		t.Fatalf("expected prompt token IDs [101 102 103], got %#v", prompt)
	}
	if _, ok := body["messages"]; ok {
		t.Fatalf("did not expect tokenized prompt to also include messages: %v", body["messages"])
	}

	req.TokenizedPrompt.TokenIDs[0] = 999
	if prompt[0] != 101 {
		t.Fatalf("expected BuildOpenAIRequest to copy token IDs, got %#v", prompt)
	}
}

func TestBuildOpenAIRequest_CompletionsStringArrayPromptKeepsLegacyMessageShape(t *testing.T) {
	req := &schedtypes.LLMRequest{
		TargetModel: "test-model",
		Body: &schedtypes.LLMRequestBody{
			Completions: &schedtypes.CompletionsRequest{
				Prompt: schedtypes.Prompt{Strings: []string{"hello", "world"}},
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
