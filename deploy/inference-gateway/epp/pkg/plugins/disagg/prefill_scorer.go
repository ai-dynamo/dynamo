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
	"sync/atomic"

	"github.com/google/uuid"
	log "sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	// DynPrefillScorerType is the plugin type registered in the plugin registry.
	DynPrefillScorerType = "dyn-prefill-scorer"

	prefillReservationStateKey      = plugins.StateKey("dynamo-prefill-reservation-state")
	prefillReservationCycleStateKey = plugins.StateKey("dynamo-prefill-reservation-cycle-state")
)

// compile-time type assertions
var _ schedtypes.Scorer = &DynPrefillScorer{}
var _ plugins.Plugin = &DynPrefillScorer{}
var _ rc.ResponseBodyProcessor = &DynPrefillScorer{}

// PrefillReservationState tracks the Rust scheduler booking acquired by Score.
type PrefillReservationState struct {
	ReservationID string
	WorkerID      uint64
	DpRank        uint32
	releaseTried  atomic.Bool
}

// Clone implements plugins.StateData.
func (s *PrefillReservationState) Clone() plugins.StateData {
	if s == nil {
		return nil
	}
	clone := &PrefillReservationState{
		ReservationID: s.ReservationID,
		WorkerID:      s.WorkerID,
		DpRank:        s.DpRank,
	}
	clone.releaseTried.Store(s.releaseTried.Load())
	return clone
}

// DynPrefillScorerConfig holds the configuration for the DynPrefillScorer plugin.
type DynPrefillScorerConfig struct{}

// DynPrefillScorerFactory defines the factory function for DynPrefillScorer.
func DynPrefillScorerFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	cfg := DynPrefillScorerConfig{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", DynPrefillScorerType, err)
		}
	}

	if err := dynscorer.InitFFI(); err != nil {
		return nil, fmt.Errorf("Dynamo FFI init for prefill scorer failed: %w", err)
	}

	return newDynPrefillScorer(handle.Context()).WithName(name), nil
}

// NewDynPrefillScorer initializes a new DynPrefillScorer.
func NewDynPrefillScorer() *DynPrefillScorer {
	return newDynPrefillScorer(context.Background())
}

func newDynPrefillScorer(ctx context.Context) *DynPrefillScorer {
	return &DynPrefillScorer{
		typedName:               plugins.TypedName{Type: DynPrefillScorerType, Name: DynPrefillScorerType},
		pluginState:             plugins.NewPluginState(ctx),
		routeAndReservePrefill:  dynscorer.CallRouteAndReservePrefillRequest,
		routePrefillAdvisory:    dynscorer.CallRoutePrefillRequest,
		freePrefillReservation:  dynscorer.CallFreePrefillRequest,
		newPrefillReservationID: func() string { return "epp-prefill-" + uuid.NewString() },
	}
}

// DynPrefillScorer is a scorer plugin for the prefill scheduling profile.
type DynPrefillScorer struct {
	typedName               plugins.TypedName
	pluginState             *plugins.PluginState
	routeAndReservePrefill  func(string, string, string) (*dynscorer.RoutingResult, error)
	routePrefillAdvisory    func(string, string) (*dynscorer.RoutingResult, error)
	freePrefillReservation  func(string, uint64, uint32) error
	newPrefillReservationID func() string
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *DynPrefillScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the name of the scorer.
func (s *DynPrefillScorer) WithName(name string) *DynPrefillScorer {
	s.typedName.Name = name
	return s
}

// Category returns the scorer category.
func (s *DynPrefillScorer) Category() schedtypes.ScorerCategory {
	return schedtypes.Affinity
}

// Score scores endpoints for prefill suitability.
func (s *DynPrefillScorer) Score(ctx context.Context, cycleState *schedtypes.CycleState, req *schedtypes.InferenceRequest, endpoints []schedtypes.Endpoint) map[schedtypes.Endpoint]float64 {
	logger := log.FromContext(ctx)

	if !readPrefillEnabled(cycleState) {
		logger.V(logutil.VERBOSE).Info("DynPrefillScorer: prefill not enabled, returning zero scores")
		return uniformScores(endpoints, 0)
	}

	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: failed to build request")
		return uniformScores(endpoints, 0)
	}

	endpointsJSON := serializeEndpoints(endpoints)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: endpoints received for scoring",
		"endpointCount", len(endpoints),
		"endpointsJSON", string(endpointsJSON))

	var result *dynscorer.RoutingResult
	if req.RequestId == "" {
		logger.V(logutil.DEFAULT).Info("DynPrefillScorer: request has no ID; using advisory routing without a reservation")
		result, err = s.routePrefillAdvisory(requestJSON, endpointsJSON)
	} else {
		if releaseErr := s.releaseReservation(ctx, req.RequestId, true); releaseErr != nil {
			logger.V(logutil.DEFAULT).Error(releaseErr, "DynPrefillScorer: failed to release previous reservation",
				"requestID", req.RequestId)
			cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
			return uniformScores(endpoints, 0)
		}

		reservationID := s.newPrefillReservationID()
		result, err = s.routeAndReservePrefill(reservationID, requestJSON, endpointsJSON)
		if err == nil {
			reservation := &PrefillReservationState{
				ReservationID: reservationID,
				WorkerID:      result.WorkerID,
				DpRank:        result.DpRank,
			}
			s.pluginState.Write(req.RequestId, prefillReservationStateKey, reservation)
			cycleState.Write(prefillReservationCycleStateKey, reservation)
		}
	}
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: FFI prefill routing failed")
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}

	prefillWorkerID := strconv.FormatUint(result.WorkerID, 10)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: prefill worker selected",
		"prefillWorkerID", prefillWorkerID,
		"prefillDpRank", result.DpRank,
		"tokenCount", len(result.TokenData))

	if req.Headers == nil {
		req.Headers = map[string]string{}
	}
	req.Headers[PrefillWorkerIDHeader] = prefillWorkerID
	if result.DpRank != dynscorer.UnsetDpRank {
		req.Headers[PrefillDpRankHeader] = strconv.FormatUint(uint64(result.DpRank), 10)
	} else {
		delete(req.Headers, PrefillDpRankHeader)
	}

	return uniformScores(endpoints, 1.0)
}

// ResponseBody releases prefill load as soon as a decode response is observed.
// At that point the remote prefill stage has completed. The framework also
// invokes this hook with EndOfStream for cancellations after target selection.
func (s *DynPrefillScorer) ResponseBody(ctx context.Context, request *schedtypes.InferenceRequest, response *rc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || request.RequestId == "" {
		return
	}
	// Intermediate chunks skip a failed first attempt; the terminal callback
	// gets one bounded retry without stalling every chunk on synchronous FFI.
	retry := response != nil && response.EndOfStream
	if err := s.releaseReservation(ctx, request.RequestId, retry); err != nil {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(err,
			"DynPrefillScorer ResponseBody: failed to free prefill reservation",
			"requestID", request.RequestId)
	}
}

func (s *DynPrefillScorer) releaseReservation(_ context.Context, requestID string, retry bool) error {
	state, err := plugins.ReadPluginStateKey[*PrefillReservationState](
		s.pluginState, requestID, prefillReservationStateKey,
	)
	if err != nil {
		return nil
	}
	if !retry && state.releaseTried.Swap(true) {
		return nil
	}
	if err := s.freePrefillReservation(state.ReservationID, state.WorkerID, state.DpRank); err != nil {
		return err
	}
	s.pluginState.Delete(requestID)
	return nil
}
