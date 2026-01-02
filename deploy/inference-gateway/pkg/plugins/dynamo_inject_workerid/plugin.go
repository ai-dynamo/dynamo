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
	typeString            = "dynamo-inject-workerid"
	pluginName            = "dynamo-inject-workerid"
	WorkerIDHeader        = "x-worker-instance-id"
	PrefillWorkerIDHeader = "x-prefiller-host-port"
	TokenDataHeaderKey    = "x-dynamo-token-data"
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

// PreRequest is called after scheduling and before the request is sent to the model server.
// In v1.2.1, this interface no longer receives targetPort and no longer has direct body mutation.
// The request body mutation must happen through a different mechanism (e.g., BBR extension or
// custom request transformation).
//
// This plugin now focuses on ensuring headers are properly set for downstream processing.
// The actual body mutation (adding nvext fields) needs to be handled at the proxy/gateway level
// based on these headers.
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

	// Handle worker instance ID - ensure it's trimmed
	wid := strings.TrimSpace(req.Headers[WorkerIDHeader])
	if wid != "" {
		req.Headers[WorkerIDHeader] = wid
	}

	// Handle prefill worker ID - ensure it's trimmed
	prefillWid := strings.TrimSpace(req.Headers[PrefillWorkerIDHeader])
	if prefillWid != "" {
		req.Headers[PrefillWorkerIDHeader] = prefillWid
	}

	// In v1.2.1, we cannot directly mutate the request body from PreRequest plugins.
	// The body mutation for nvext fields (backend_instance_id, prefill_worker_id, decode_worker_id, token_data)
	// must be handled by a separate mechanism such as:
	// 1. A custom BBR (Body-Based Router) extension
	// 2. Gateway-level request transformation
	// 3. An Envoy filter that reads these headers and modifies the body
	//
	// We set special headers that can be read by such mechanisms:
	// - x-worker-instance-id: The selected worker ID (decode worker in disagg mode)
	// - x-prefiller-host-port: The prefill worker ID (only in disagg mode)
	// - x-dynamo-token-data: JSON-encoded token IDs for KV cache routing
	//
	// Additionally, we set headers to indicate the routing mode:
	if prefillWid != "" && prefillWid != wid {
		// Disaggregated mode
		req.Headers["x-dynamo-routing-mode"] = "disaggregated"
		if prefillWidUint, err := strconv.ParseUint(prefillWid, 10, 64); err == nil {
			req.Headers["x-dynamo-prefill-worker-id"] = strconv.FormatUint(prefillWidUint, 10)
		}
		if widUint, err := strconv.ParseUint(wid, 10, 64); err == nil {
			req.Headers["x-dynamo-decode-worker-id"] = strconv.FormatUint(widUint, 10)
		}
	} else if wid != "" {
		// Aggregated mode
		req.Headers["x-dynamo-routing-mode"] = "aggregated"
		if widUint, err := strconv.ParseUint(wid, 10, 64); err == nil {
			req.Headers["x-dynamo-backend-instance-id"] = strconv.FormatUint(widUint, 10)
		}
	}
}

