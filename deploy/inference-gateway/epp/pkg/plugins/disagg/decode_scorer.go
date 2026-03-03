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
	"context"
	"encoding/json"
	"fmt"
	"sync"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	// DynDecodeScorerType is the plugin type registered in the plugin registry.
	DynDecodeScorerType = "dyn-decode-scorer"

	WorkerIDHeader        = "x-worker-instance-id"
	PrefillWorkerIDHeader = "x-prefill-instance-id"
	RoutingModeHeader     = "x-dynamo-routing-mode"

	// decodeStateKey is the key used to store routing state in PluginState
	decodeStateKey = "dynamo-decode-routing-state"
)

// compile-time type assertions
var _ framework.Scorer = &DynDecodeScorer{}
var _ plugins.Plugin = &DynDecodeScorer{}
var _ rc.PreRequest = &DynDecodeScorer{}
var _ rc.ResponseStreaming = &DynDecodeScorer{}
var _ rc.ResponseComplete = &DynDecodeScorer{}

// DecodeRoutingState holds routing information passed from Score() to PreRequest().
type DecodeRoutingState struct {
	WorkerID        string
	PrefillWorkerID string
	TokenData       []int64
}

// Clone implements plugins.StateData.
func (s *DecodeRoutingState) Clone() plugins.StateData {
	if s == nil {
		return nil
	}
	clone := &DecodeRoutingState{
		WorkerID:        s.WorkerID,
		PrefillWorkerID: s.PrefillWorkerID,
	}
	if s.TokenData != nil {
		clone.TokenData = make([]int64, len(s.TokenData))
		copy(clone.TokenData, s.TokenData)
	}
	return clone
}

// DynDecodeScorerConfig holds the configuration for the DynDecodeScorer plugin.
type DynDecodeScorerConfig struct{}

// DynDecodeScorerFactory defines the factory function for DynDecodeScorer.
func DynDecodeScorerFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	cfg := DynDecodeScorerConfig{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", DynDecodeScorerType, err)
		}
	}

	// Initialize the shared FFI (idempotent)
	if err := dynscorer.InitFFI(); err != nil {
		return nil, fmt.Errorf("Dynamo FFI init for decode scorer failed: %w", err)
	}

	return NewDynDecodeScorer(handle.Context()).WithName(name), nil
}

// NewDynDecodeScorer initializes a new DynDecodeScorer.
func NewDynDecodeScorer(ctx context.Context) *DynDecodeScorer {
	return &DynDecodeScorer{
		typedName:   plugins.TypedName{Type: DynDecodeScorerType, Name: DynDecodeScorerType},
		pluginState: plugins.NewPluginState(ctx),
	}
}

// DynDecodeScorer is a scorer plugin for the decode scheduling profile.
//
// When Score() is called, it:
//  1. Reads PrefillEnabledState from CycleState (written by DisaggProfileHandler).
//  2. Calls the Dynamo FFI decode router with is_disaggregated flag.
//  3. Sets routing headers on the request.
//  4. Stores routing state for PreRequest to register with router bookkeeping.
//
// It also implements PreRequest, ResponseStreaming, and ResponseComplete lifecycle hooks
// for router bookkeeping (add_request, mark_prefill_complete, free_request).
type DynDecodeScorer struct {
	typedName      plugins.TypedName
	pluginState    *plugins.PluginState
	firstTokenSeen sync.Map
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *DynDecodeScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the name of the scorer.
func (s *DynDecodeScorer) WithName(name string) *DynDecodeScorer {
	s.typedName.Name = name
	return s
}

// Score scores pods for decode suitability.
func (s *DynDecodeScorer) Score(ctx context.Context, cycleState *schedtypes.CycleState, req *schedtypes.LLMRequest, pods []schedtypes.Pod) map[schedtypes.Pod]float64 {
	logger := log.FromContext(ctx)

	isDisaggregated := readPrefillEnabled(cycleState)

	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: failed to build request")
		return uniformScores(pods, 1.0)
	}

	podsJSON := serializePods(pods)
	logger.V(logutil.DEFAULT).Info("DynDecodeScorer: pods received for scoring",
		"podCount", len(pods),
		"podsJSON", string(podsJSON))

	result, err := dynscorer.CallRouteDecodeRequest(requestJSON, podsJSON, isDisaggregated)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: FFI decode routing failed")
		return uniformScores(pods, 1.0)
	}

	workerIDStr := fmt.Sprintf("%d", result.WorkerID)
	logger.V(logutil.DEFAULT).Info("DynDecodeScorer: decode worker selected",
		"decodeWorkerID", workerIDStr,
		"isDisaggregated", isDisaggregated,
		"tokenCount", len(result.TokenData))

	// Set routing headers
	if req.Headers == nil {
		req.Headers = map[string]string{}
	}
	req.Headers[WorkerIDHeader] = workerIDStr

	if isDisaggregated {
		req.Headers[RoutingModeHeader] = "disaggregated"
		// Read the prefill worker ID selected by DynPrefillScorer via CycleState.
		prefillState, err := schedtypes.ReadCycleStateKey[*PrefillWorkerIDState](cycleState, PrefillWorkerIDStateKey)
		if err == nil && prefillState != nil && prefillState.WorkerID != "" {
			req.Headers[PrefillWorkerIDHeader] = prefillState.WorkerID
			logger.V(logutil.DEFAULT).Info("DynDecodeScorer: setting prefill worker header",
				"prefillWorkerID", prefillState.WorkerID)
		}
	} else {
		req.Headers[RoutingModeHeader] = "aggregated"
	}

	// Store routing state for PreRequest bookkeeping
	if req.RequestId != "" {
		routingState := &DecodeRoutingState{
			WorkerID:  workerIDStr,
			TokenData: result.TokenData,
		}
		s.pluginState.Write(req.RequestId, plugins.StateKey(decodeStateKey), routingState)
	}

	// Score: all decode pods get 1.0 since the router's internal selection is authoritative
	// and the worker ID is communicated via headers.
	return uniformScores(pods, 1.0)
}

// PreRequest is called after scheduling is finalized and before the request is sent to the worker.
// This registers the request with the Dynamo router's bookkeeping.
func (s *DynDecodeScorer) PreRequest(ctx context.Context, request *schedtypes.LLMRequest, _ *schedtypes.SchedulingResult) {
	logger := log.FromContext(ctx)

	if request == nil || request.RequestId == "" {
		logger.V(logutil.DEBUG).Info("DynDecodeScorer PreRequest: no request ID, skipping")
		return
	}

	state, err := plugins.ReadPluginStateKey[*DecodeRoutingState](
		s.pluginState, request.RequestId, plugins.StateKey(decodeStateKey),
	)
	s.pluginState.Delete(request.RequestId)

	if err != nil {
		logger.V(logutil.DEBUG).Info("DynDecodeScorer PreRequest: no routing state found",
			"requestID", request.RequestId)
		return
	}

	var workerIDUint uint64
	if _, parseErr := fmt.Sscanf(state.WorkerID, "%d", &workerIDUint); parseErr != nil {
		logger.V(logutil.DEFAULT).Error(parseErr, "DynDecodeScorer PreRequest: invalid worker ID",
			"requestID", request.RequestId, "workerID", state.WorkerID)
		return
	}

	if addErr := dynscorer.CallAddRequest(request.RequestId, state.TokenData, workerIDUint, 0); addErr != nil {
		logger.V(logutil.DEFAULT).Error(addErr, "DynDecodeScorer PreRequest: failed to add request",
			"requestID", request.RequestId)
		return
	}

	logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: registered request",
		"requestID", request.RequestId,
		"workerID", state.WorkerID,
		"tokenCount", len(state.TokenData))
}

// ResponseStreaming is called for each chunk of a streaming response.
// On the first token, it marks prefill as complete in the Dynamo router's bookkeeping.
func (s *DynDecodeScorer) ResponseStreaming(ctx context.Context, request *schedtypes.LLMRequest, _ *rc.Response, _ *backend.Pod) {
	if request == nil || request.RequestId == "" {
		return
	}

	if _, alreadySeen := s.firstTokenSeen.LoadOrStore(request.RequestId, true); !alreadySeen {
		logger := log.FromContext(ctx)
		if err := dynscorer.CallMarkPrefillComplete(request.RequestId); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseStreaming: failed to mark prefill complete",
				"requestID", request.RequestId)
			return
		}
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseStreaming: marked prefill complete",
			"requestID", request.RequestId)
	}
}

// ResponseComplete is called after the complete response is sent to the client.
// It cleans up the router bookkeeping state for the completed request.
func (s *DynDecodeScorer) ResponseComplete(ctx context.Context, request *schedtypes.LLMRequest, _ *rc.Response, _ *backend.Pod) {
	logger := log.FromContext(ctx)

	if request == nil || request.RequestId == "" {
		return
	}

	s.firstTokenSeen.Delete(request.RequestId)

	if err := dynscorer.CallFreeRequest(request.RequestId); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseComplete: failed to free request",
			"requestID", request.RequestId)
		return
	}

	logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseComplete: freed request",
		"requestID", request.RequestId)
}
