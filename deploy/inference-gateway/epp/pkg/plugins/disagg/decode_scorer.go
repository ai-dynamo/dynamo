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
	"strconv"
	"sync"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	// DynDecodeScorerType is the plugin type registered in the plugin registry.
	DynDecodeScorerType = "dyn-decode-scorer"

	WorkerIDHeader        = "x-worker-instance-id"
	PrefillWorkerIDHeader = "x-prefill-instance-id"
	DpRankHeader          = "x-dp-rank"
	PrefillDpRankHeader   = "x-prefill-dp-rank"
	RoutingModeHeader     = "x-dynamo-routing-mode"

	decodeStateKey = "dynamo-decode-routing-state"
)

// compile-time type assertions
var _ schedtypes.Scorer = &DynDecodeScorer{}
var _ plugins.Plugin = &DynDecodeScorer{}
var _ rc.PreRequest = &DynDecodeScorer{}
var _ rc.ResponseBodyProcessor = &DynDecodeScorer{}

// DecodeRoutingState holds routing information passed from Score() to PreRequest().
type DecodeRoutingState struct {
	WorkerID        string
	DpRank          uint32
	PrefillWorkerID string
	TokenData       []int64
	ModifiedBody    string
}

// Clone implements plugins.StateData.
func (s *DecodeRoutingState) Clone() plugins.StateData {
	if s == nil {
		return nil
	}
	clone := &DecodeRoutingState{
		WorkerID:        s.WorkerID,
		DpRank:          s.DpRank,
		PrefillWorkerID: s.PrefillWorkerID,
		ModifiedBody:    s.ModifiedBody,
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

	if err := dynscorer.InitFFI(); err != nil {
		return nil, fmt.Errorf("Dynamo FFI init for decode scorer failed: %w", err)
	}

	enforceDisagg := getEnvBoolOrDefault("DYN_ENFORCE_DISAGG", false)
	return NewDynDecodeScorer(handle.Context(), enforceDisagg).WithName(name), nil
}

// NewDynDecodeScorer initializes a new DynDecodeScorer.
func NewDynDecodeScorer(ctx context.Context, enforceDisagg bool) *DynDecodeScorer {
	return &DynDecodeScorer{
		typedName:     plugins.TypedName{Type: DynDecodeScorerType, Name: DynDecodeScorerType},
		pluginState:   plugins.NewPluginState(ctx),
		enforceDisagg: enforceDisagg,
	}
}

// DynDecodeScorer is a scorer plugin for the decode scheduling profile.
type DynDecodeScorer struct {
	typedName      plugins.TypedName
	pluginState    *plugins.PluginState
	enforceDisagg  bool
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

// Category returns the scorer category.
func (s *DynDecodeScorer) Category() schedtypes.ScorerCategory {
	return schedtypes.Affinity
}

// Score scores endpoints for decode suitability.
func (s *DynDecodeScorer) Score(ctx context.Context, cycleState *schedtypes.CycleState, req *schedtypes.InferenceRequest, endpoints []schedtypes.Endpoint) map[schedtypes.Endpoint]float64 {
	logger := log.FromContext(ctx)

	isDisaggregated := readPrefillEnabled(cycleState)

	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: failed to build request")
		return uniformScores(endpoints, 1.0)
	}

	endpointsJSON := serializeEndpoints(endpoints)
	logger.V(logutil.DEFAULT).Info("DynDecodeScorer: endpoints received for scoring",
		"endpointCount", len(endpoints),
		"endpointsJSON", string(endpointsJSON))

	result, err := dynscorer.CallRouteDecodeRequest(requestJSON, endpointsJSON, isDisaggregated)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: FFI decode routing failed")
		return uniformScores(endpoints, 1.0)
	}

	workerIDStr := fmt.Sprintf("%d", result.WorkerID)
	dpRankStr := strconv.FormatUint(uint64(result.DpRank), 10)
	logger.V(logutil.DEFAULT).Info("[EPP-SCORER] FFI returned tokens from C bindings tokenization",
		"decodeWorkerID", workerIDStr,
		"decodeDpRank", result.DpRank,
		"isDisaggregated", isDisaggregated,
		"tokenCount", len(result.TokenData),
		"hasModifiedBody", result.ModifiedBody != "")

	if req.Headers == nil {
		req.Headers = map[string]string{}
	}
	req.Headers[WorkerIDHeader] = workerIDStr
	req.Headers[DpRankHeader] = dpRankStr

	if isDisaggregated {
		req.Headers[RoutingModeHeader] = "disaggregated"
		if prefillID, ok := req.Headers[PrefillWorkerIDHeader]; ok {
			logger.V(logutil.DEFAULT).Info("DynDecodeScorer: prefill worker header present",
				"prefillWorkerID", prefillID)
		} else if s.enforceDisagg {
			logger.V(logutil.DEFAULT).Error(nil,
				"DynDecodeScorer: prefill worker header missing and enforce_disagg=true")
		} else {
			logger.V(logutil.DEFAULT).Error(nil,
				"DynDecodeScorer: x-prefill-instance-id header missing — DynPrefillScorer did not set it")
		}
	} else {
		req.Headers[RoutingModeHeader] = "aggregated"
	}

	// Store routing state for PreRequest bookkeeping
	if req.RequestId != "" {
		routingState := &DecodeRoutingState{
			WorkerID:     workerIDStr,
			DpRank:       result.DpRank,
			TokenData:    result.TokenData,
			ModifiedBody: result.ModifiedBody,
		}
		s.pluginState.Write(req.RequestId, plugins.StateKey(decodeStateKey), routingState)
	}

	// Inject pre-computed tokens into the request body so the frontend
	// sidecar can skip redundant tokenization.
	setTokenizedPrompt(req, result.TokenData, logger)

	return uniformScores(endpoints, 1.0)
}

// PreRequest registers the request with the Dynamo router's bookkeeping.
func (s *DynDecodeScorer) PreRequest(ctx context.Context, request *schedtypes.InferenceRequest, _ *schedtypes.SchedulingResult) {
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

	if addErr := dynscorer.CallAddRequest(request.RequestId, state.TokenData, workerIDUint, state.DpRank); addErr != nil {
		logger.V(logutil.DEFAULT).Error(addErr, "DynDecodeScorer PreRequest: failed to add request",
			"requestID", request.RequestId)
		return
	}

	logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: registered request",
		"requestID", request.RequestId,
		"workerID", state.WorkerID,
		"dpRank", state.DpRank,
		"tokenCount", len(state.TokenData))
}

// ResponseBody handles streaming chunks and end-of-stream cleanup.
// On the first token it marks prefill as complete; on EndOfStream it frees the request.
func (s *DynDecodeScorer) ResponseBody(ctx context.Context, request *schedtypes.InferenceRequest, response *rc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || request.RequestId == "" {
		return
	}

	logger := log.FromContext(ctx)

	// Mark prefill complete on first token
	if _, alreadySeen := s.firstTokenSeen.LoadOrStore(request.RequestId, true); !alreadySeen {
		if err := dynscorer.CallMarkPrefillComplete(request.RequestId); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to mark prefill complete",
				"requestID", request.RequestId)
			return
		}
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: marked prefill complete",
			"requestID", request.RequestId)
	}

	// Free request on end of stream
	if response != nil && response.EndOfStream {
		s.firstTokenSeen.Delete(request.RequestId)

		if err := dynscorer.CallFreeRequest(request.RequestId); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to free request",
				"requestID", request.RequestId)
			return
		}

		logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: freed request",
			"requestID", request.RequestId)
	}
}
