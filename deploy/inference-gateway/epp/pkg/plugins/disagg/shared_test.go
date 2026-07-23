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

package disagg

import (
	"testing"

	"github.com/go-logr/logr"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestSetTokenizedPromptStoresGAIEFieldAndNvextTokenData(t *testing.T) {
	req := &schedtypes.LLMRequest{
		RequestId: "req-1",
		Body: &schedtypes.LLMRequestBody{
			Payload: schedtypes.PayloadMap{},
		},
	}

	setTokenizedPrompt(req, []int64{101, 102, 103}, logr.Discard())

	if req.TokenizedPrompt == nil {
		t.Fatal("expected TokenizedPrompt to be set")
	}
	if got := req.TokenizedPrompt.TokenIDs; len(got) != 3 || got[0] != 101 || got[1] != 102 || got[2] != 103 {
		t.Fatalf("expected TokenizedPrompt token IDs [101 102 103], got %#v", got)
	}

	payload, ok := req.Body.Payload.(schedtypes.PayloadMap)
	if !ok {
		t.Fatalf("expected PayloadMap, got %T", req.Body.Payload)
	}
	nvext, ok := payload["nvext"].(map[string]any)
	if !ok {
		t.Fatalf("expected nvext map, got %#v", payload["nvext"])
	}
	tokenData, ok := nvext["token_data"].([]uint32)
	if !ok {
		t.Fatalf("expected nvext.token_data []uint32, got %#v", nvext["token_data"])
	}
	if len(tokenData) != 3 || tokenData[0] != 101 || tokenData[1] != 102 || tokenData[2] != 103 {
		t.Fatalf("expected nvext.token_data [101 102 103], got %#v", tokenData)
	}
}
