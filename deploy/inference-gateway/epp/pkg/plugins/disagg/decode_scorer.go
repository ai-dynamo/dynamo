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

	WorkerIDHeader        = "x-dynamo-worker-instance-id"
	PrefillWorkerIDHeader = "x-dynamo-prefill-instance-id"
	DpRankHeader          = "x-dynamo-dp-rank"
	PrefillDpRankHeader   = "x-dynamo-prefill-dp-rank"
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
	BookingID      string
	WorkerID       string
	DpRank         uint32
	TokenData      []int64
}

// Clone implements plugins.StateData.
func (s *DecodeRoutingState) Clone() plugins.StateData {
	if s == nil {
		return nil
	}
	clone := &DecodeRoutingState{
		BookingID:      s.BookingID,
		WorkerID:       s.WorkerID,
		DpRank:         s.DpRank,
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

	warnDeprecatedEnforceDisagg(log.Log.WithName(DynDecodeScorerType))
	return NewDynDecodeScorer(handle.Context()).WithName(name), nil
}

// NewDynDecodeScorer initializes a new DynDecodeScorer.
func NewDynDecodeScorer(ctx context.Context) *DynDecodeScorer {
	return &DynDecodeScorer{
		typedName:           plugins.TypedName{Type: DynDecodeScorerType, Name: DynDecodeScorerType},
		pluginState:         plugins.NewPluginState(ctx),
		addRequest:          dynscorer.CallAddRequest,
		markPrefillComplete: dynscorer.CallMarkPrefillComplete,
		freeBooking:         dynscorer.CallFreeRequest,
	}
}

// DynDecodeScorer is a scorer plugin for the decode scheduling profile.
type DynDecodeScorer struct {
	typedName           plugins.TypedName
	pluginState         *plugins.PluginState
	prefillMarkInFlight sync.Map
	addRequest          func(string, []int64, uint64, uint32) error
	markPrefillComplete func(string) error
	freeBooking         func(string) error
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
	if req == nil {
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 1.0)
	}
	booking := ensureBookingState(cycleState)
	attachBookingID(req, booking.ID)

	if err := ctx.Err(); err != nil {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer: scheduling already cancelled", "error", err.Error())
		s.cleanupBooking(ctx, booking.ID, "decode scheduling cancelled before routing")
		return uniformScores(endpoints, 1.0)
	}

	prefillEnabled := readPrefillEnabled(cycleState)
	if booking.PrefillReserved && !prefillEnabled {
		s.rollbackPrefillReservation(ctx, cycleState, req, booking, "prefill scheduling did not complete")
	}
	isDisaggregated := prefillEnabled && booking.PrefillReserved
	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: failed to build request")
		s.rollbackPrefillReservation(ctx, cycleState, req, booking, "decode request serialization failed")
		return uniformScores(endpoints, 1.0)
	}

	endpointsJSON := serializeEndpoints(endpoints)
	logger.V(logutil.DEFAULT).Info("DynDecodeScorer: endpoints received for scoring",
		"endpointCount", len(endpoints),
		"endpointsJSON", string(endpointsJSON))

	result, err := dynscorer.CallRouteDecodeRequest(requestJSON, endpointsJSON, isDisaggregated)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: FFI decode routing failed")
		s.rollbackPrefillReservation(ctx, cycleState, req, booking, "decode routing failed")
		return uniformScores(endpoints, 1.0)
	}
	if err := ctx.Err(); err != nil {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer: scheduling cancelled after decode routing", "error", err.Error())
		s.cleanupBooking(ctx, booking.ID, "decode scheduling cancelled after routing")
		return uniformScores(endpoints, 1.0)
	}

	workerIDStr := fmt.Sprintf("%d", result.WorkerID)
	dpRankStr := strconv.FormatUint(uint64(result.DpRank), 10)
	logger.V(logutil.DEFAULT).Info("[EPP-SCORER] FFI returned tokens from C bindings tokenization",
		"bookingID", booking.ID,
		"decodeWorkerID", workerIDStr,
		"decodeDpRank", result.DpRank,
		"isDisaggregated", isDisaggregated,
		"tokenCount", len(result.TokenData))

	req.Headers[WorkerIDHeader] = workerIDStr
	req.Headers[DpRankHeader] = dpRankStr
	if isDisaggregated {
		req.Headers[RoutingModeHeader] = "disaggregated"
	} else {
		req.Headers[RoutingModeHeader] = "aggregated"
		delete(req.Headers, PrefillWorkerIDHeader)
		delete(req.Headers, PrefillDpRankHeader)
	}

	routingState := &DecodeRoutingState{
		BookingID:      booking.ID,
		WorkerID:       workerIDStr,
		DpRank:         result.DpRank,
		TokenData:      result.TokenData,
	}
	s.pluginState.Write(booking.ID, plugins.StateKey(decodeStateKey), routingState)

	// Inject pre-computed tokens into the request body so the frontend
	// sidecar can skip redundant tokenization.
	setTokenizedPrompt(req, result.TokenData, logger)

	return uniformScores(endpoints, 1.0)
}

func (s *DynDecodeScorer) cleanupBooking(ctx context.Context, bookingID, reason string) bool {
	if bookingID == "" {
		return true
	}
	if err := s.freeBooking(bookingID); err != nil {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(err, "DynDecodeScorer: booking cleanup failed",
			"bookingID", bookingID, "reason", reason)
		return false
	}
	log.FromContext(ctx).V(logutil.VERBOSE).Info("DynDecodeScorer: booking cleaned up",
		"bookingID", bookingID, "reason", reason)
	return true
}

func (s *DynDecodeScorer) rollbackPrefillReservation(
	ctx context.Context,
	cycleState *schedtypes.CycleState,
	request *schedtypes.InferenceRequest,
	booking *BookingState,
	reason string,
) {
	if booking != nil && booking.PrefillReserved && s.cleanupBooking(ctx, booking.ID, reason) {
		booking.PrefillReserved = false
		cycleState.Write(BookingStateKey, booking)
	}
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
	if request != nil {
		if request.Headers == nil {
			request.Headers = map[string]string{}
		}
		request.Headers[RoutingModeHeader] = "aggregated"
		delete(request.Headers, PrefillWorkerIDHeader)
		delete(request.Headers, PrefillDpRankHeader)
	}
}

// PreRequest registers the request with the Dynamo router's bookkeeping.
func (s *DynDecodeScorer) PreRequest(ctx context.Context, request *schedtypes.InferenceRequest, _ *schedtypes.SchedulingResult) {
	logger := log.FromContext(ctx)
	bookingID := bookingIDFromRequest(request)
	if bookingID == "" {
		logger.V(logutil.DEBUG).Info("DynDecodeScorer PreRequest: no controller booking ID, skipping")
		return
	}
	if err := ctx.Err(); err != nil {
		s.cleanupBooking(ctx, bookingID, "request cancelled before decode booking")
		return
	}

	state, err := plugins.ReadPluginStateKey[*DecodeRoutingState](
		s.pluginState, bookingID, plugins.StateKey(decodeStateKey),
	)
	s.pluginState.Delete(bookingID)
	if err != nil || state == nil || state.BookingID != bookingID {
		logger.V(logutil.DEBUG).Info("DynDecodeScorer PreRequest: no routing state found",
			"bookingID", bookingID)
		s.cleanupBooking(ctx, bookingID, "decode routing state missing")
		return
	}

	var workerIDUint uint64
	if _, parseErr := fmt.Sscanf(state.WorkerID, "%d", &workerIDUint); parseErr != nil {
		logger.V(logutil.DEFAULT).Error(parseErr, "DynDecodeScorer PreRequest: invalid worker ID",
			"bookingID", bookingID, "workerID", state.WorkerID)
		s.cleanupBooking(ctx, bookingID, "decode worker ID invalid")
		return
	}

	if addErr := s.addRequest(
		bookingID,
		state.TokenData,
		workerIDUint,
		state.DpRank,
	); addErr != nil {
		logger.V(logutil.DEFAULT).Error(addErr, "DynDecodeScorer PreRequest: failed to add request",
			"bookingID", bookingID)
		s.cleanupBooking(ctx, bookingID, "decode booking failed")
		return
	}

	logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: registered request",
		"bookingID", bookingID,
		"workerID", state.WorkerID,
		"dpRank", state.DpRank,
		"tokenCount", len(state.TokenData))

	go func() {
		<-ctx.Done()
		s.cleanupBooking(ctx, bookingID, "request context cancelled")
	}()
}

// ResponseBody handles streaming chunks and end-of-stream cleanup.
// On the first token it marks prefill as complete; on EndOfStream it frees the request.
func (s *DynDecodeScorer) ResponseBody(ctx context.Context, request *schedtypes.InferenceRequest, response *rc.Response, _ *fwkdl.EndpointMetadata) {
	bookingID := bookingIDFromRequest(request)
	if bookingID == "" || response == nil {
		return
	}

	logger := log.FromContext(ctx)

	// Terminal cleanup takes precedence over first-token bookkeeping. This also
	// covers empty/error responses where no output token was observed.
	if response.EndOfStream {
		s.prefillMarkInFlight.Delete(bookingID)
		if err := s.freeBooking(bookingID); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to free request",
				"bookingID", bookingID, "requestID", request.RequestId)
		} else {
			logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: freed request",
				"bookingID", bookingID, "requestID", request.RequestId)
		}
		return
	}

	// Keep the marker while the FFI call is in flight. On failure, remove it so
	// the next response chunk retries instead of leaking the prefill load.
	if _, alreadyInFlight := s.prefillMarkInFlight.LoadOrStore(bookingID, struct{}{}); alreadyInFlight {
		return
	}
	if err := s.markPrefillComplete(bookingID); err != nil {
		s.prefillMarkInFlight.Delete(bookingID)
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to mark prefill complete",
			"bookingID", bookingID, "requestID", request.RequestId)
	} else {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: marked prefill complete",
			"bookingID", bookingID, "requestID", request.RequestId)
	}
}
