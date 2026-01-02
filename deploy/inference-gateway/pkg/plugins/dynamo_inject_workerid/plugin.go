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

// Package dynamo_inject_workerid provides a PreRequest plugin that processes
// and normalizes Dynamo routing headers before forwarding to the backend.
//
// # Header-Only Approach
//
// This plugin works with the KV scorer to enable Dynamo routing via HTTP headers.
// The backend workers must read these headers to extract routing information:
//
//   - x-worker-instance-id: The selected worker ID (decode worker in disagg mode)
//   - x-prefiller-host-port: The prefill worker ID (only in disagg mode)
//   - x-dynamo-token-data: JSON-encoded token IDs for KV cache routing
//   - x-dynamo-routing-mode: "aggregated" or "disaggregated"
//   - x-dynamo-backend-instance-id: Worker ID in aggregated mode
//   - x-dynamo-prefill-worker-id: Prefill worker in disagg mode
//   - x-dynamo-decode-worker-id: Decode worker in disagg mode
//
// The backend should parse these headers and use them for routing decisions
// instead of relying on nvext body fields.
package dynamo_inject_workerid

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	typeString = "dynamo-inject-workerid"
	pluginName = "dynamo-inject-workerid"

	// Headers set by the KV scorer and processed by this plugin
	WorkerIDHeader        = "x-worker-instance-id"
	PrefillWorkerIDHeader = "x-prefiller-host-port"
	TokenDataHeaderKey    = "x-dynamo-token-data"

	// Additional headers set by this plugin for backend consumption
	RoutingModeHeader       = "x-dynamo-routing-mode"
	BackendInstanceIDHeader = "x-dynamo-backend-instance-id"
	PrefillWorkerHeader     = "x-dynamo-prefill-worker-id"
	DecodeWorkerHeader      = "x-dynamo-decode-worker-id"
)

var _ plugins.Plugin = (*InjectWorkerIDPreRequest)(nil)
var _ rc.PreRequest = (*InjectWorkerIDPreRequest)(nil)

type InjectWorkerIDPreRequest struct {
	typedName plugins.TypedName
}

func NewInjectWorkerIDPreRequest() *InjectWorkerIDPreRequest {
	return &InjectWorkerIDPreRequest{
		typedName: plugins.TypedName{Type: typeString, Name: pluginName},
	}
}

func (p *InjectWorkerIDPreRequest) WithName(name string) *InjectWorkerIDPreRequest {
	p.typedName.Name = name
	return p
}

func InjectWorkerIDPreRequestFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewInjectWorkerIDPreRequest().WithName(name), nil
}

func (p *InjectWorkerIDPreRequest) TypedName() plugins.TypedName { return p.typedName }

// PreRequest processes headers set by the KV scorer and adds normalized
// headers that the backend can easily consume.
//
// The backend should read:
//   - x-dynamo-routing-mode: "aggregated" or "disaggregated"
//   - x-dynamo-backend-instance-id: Worker ID (aggregated mode)
//   - x-dynamo-prefill-worker-id: Prefill worker (disagg mode)
//   - x-dynamo-decode-worker-id: Decode worker (disagg mode)
//   - x-dynamo-token-data: JSON array of token IDs
func (p *InjectWorkerIDPreRequest) PreRequest(
	_ context.Context,
	req *schedtypes.LLMRequest,
	_ *schedtypes.SchedulingResult,
) {
	if req == nil {
		return
	}
	if req.Headers == nil {
		req.Headers = map[string]string{}
	}

	// Get worker IDs from scorer (these are set by kv-aware-scorer)
	wid := strings.TrimSpace(req.Headers[WorkerIDHeader])
	prefillWid := strings.TrimSpace(req.Headers[PrefillWorkerIDHeader])

	// Normalize the primary headers
	if wid != "" {
		req.Headers[WorkerIDHeader] = wid
	}
	if prefillWid != "" {
		req.Headers[PrefillWorkerIDHeader] = prefillWid
	}

	// Set routing mode and normalized worker headers for easy backend consumption
	if prefillWid != "" && prefillWid != wid && wid != "" {
		// Disaggregated mode: separate prefill and decode workers
		req.Headers[RoutingModeHeader] = "disaggregated"

		if prefillWidUint, err := strconv.ParseUint(prefillWid, 10, 64); err == nil {
			req.Headers[PrefillWorkerHeader] = strconv.FormatUint(prefillWidUint, 10)
		}
		if widUint, err := strconv.ParseUint(wid, 10, 64); err == nil {
			req.Headers[DecodeWorkerHeader] = strconv.FormatUint(widUint, 10)
		}
	} else if wid != "" {
		// Aggregated mode: single worker handles both prefill and decode
		req.Headers[RoutingModeHeader] = "aggregated"

		if widUint, err := strconv.ParseUint(wid, 10, 64); err == nil {
			req.Headers[BackendInstanceIDHeader] = strconv.FormatUint(widUint, 10)
		}
	}

	// Token data header is already set by the scorer, just pass it through
	// Backend should parse: JSON.parse(headers["x-dynamo-token-data"])
}
