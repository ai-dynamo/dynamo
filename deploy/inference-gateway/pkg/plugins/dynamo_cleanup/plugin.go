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

package dynamo_cleanup

import (
	"context"
	"encoding/json"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	dynamo "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	PluginName = "dynamo-cleanup"
	PluginType = "dynamo-cleanup"
)

// DynamoCleanupPlugin is a ResponseComplete plugin that cleans up router state
// when a request completes. It calls dynamo_router_free_request to release
// the bookkeeping resources associated with the request.
//
// In v1.2.1, this implements the ResponseComplete interface instead of PostResponse.
type DynamoCleanupPlugin struct {
	typedName plugins.TypedName
}

var _ plugins.Plugin = (*DynamoCleanupPlugin)(nil)
var _ rc.ResponseComplete = (*DynamoCleanupPlugin)(nil)

// NewDynamoCleanupPlugin creates a new DynamoCleanupPlugin instance.
func NewDynamoCleanupPlugin() *DynamoCleanupPlugin {
	return &DynamoCleanupPlugin{
		typedName: plugins.TypedName{Type: PluginType, Name: PluginName},
	}
}

// WithName sets a custom name for the plugin.
func (p *DynamoCleanupPlugin) WithName(name string) *DynamoCleanupPlugin {
	p.typedName.Name = name
	return p
}

// DynamoCleanupPluginFactory creates a DynamoCleanupPlugin from configuration.
func DynamoCleanupPluginFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewDynamoCleanupPlugin().WithName(name), nil
}

// TypedName returns the plugin's type and name.
func (p *DynamoCleanupPlugin) TypedName() plugins.TypedName {
	return p.typedName
}

// ResponseComplete is called after the complete response is sent.
// It cleans up the router bookkeeping state for the completed request.
//
// This replaces the old PostResponse interface in v1.2.1.
func (p *DynamoCleanupPlugin) ResponseComplete(
	ctx context.Context,
	request *schedtypes.LLMRequest,
	response *rc.Response,
	targetPod *backend.Pod,
) {
	logger := log.FromContext(ctx)

	if request == nil {
		logger.V(logutil.DEBUG).Info("DynamoCleanupPlugin: request is nil, skipping cleanup")
		return
	}

	requestID := request.RequestId
	if requestID == "" {
		logger.V(logutil.DEBUG).Info("DynamoCleanupPlugin: no request ID, skipping cleanup")
		return
	}

	// Call the dynamo router to free the request bookkeeping
	if err := dynamo.CallFreeRequest(requestID); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynamoCleanupPlugin: failed to free request",
			"requestID", requestID)
		return
	}

	logger.V(logutil.VERBOSE).Info("DynamoCleanupPlugin: freed request from router",
		"requestID", requestID)
}

