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
)

// compile-time type assertions
var _ schedtypes.Scorer = &DynDecodeScorer{}
var _ plugins.Plugin = &DynDecodeScorer{}
var _ rc.ResponseBodyProcessor = &DynDecodeScorer{}

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
func NewDynDecodeScorer(_ context.Context) *DynDecodeScorer {
	return &DynDecodeScorer{
		typedName:           plugins.TypedName{Type: DynDecodeScorerType, Name: DynDecodeScorerType},
		routeDecode:         dynscorer.CallRouteDecodeRequest,
		addRequest:          dynscorer.CallAddRequest,
		markPrefillComplete: dynscorer.CallMarkPrefillComplete,
		freeBooking:         dynscorer.CallFreeRequest,
	}
}

// DynDecodeScorer is a scorer plugin for the decode scheduling profile.
type DynDecodeScorer struct {
	typedName           plugins.TypedName
	routeDecode         func(string, string, bool) (*dynscorer.RoutingResult, error)
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

	result, err := s.routeDecode(requestJSON, endpointsJSON, isDisaggregated)
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
	lifecycle := registerBookingLifecycle(booking.ID, s.freeBooking)
	lifecycle.armCancellation(ctx)
	if !lifecycle.startDecodeRegistration() {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer: request cancelled before decode booking registration",
			"bookingID", booking.ID)
		s.rollbackPrefillReservation(ctx, cycleState, req, booking, "decode booking registration cancelled")
		return uniformScores(endpoints, 1.0)
	}
	addErr := s.addRequest(booking.ID, result.TokenData, result.WorkerID, result.DpRank)
	lifecycle.finishDecodeRegistration()
	if addErr != nil {
		logger.V(logutil.DEFAULT).Error(addErr, "DynDecodeScorer: failed to add decode booking",
			"bookingID", booking.ID)
		s.rollbackPrefillReservation(ctx, cycleState, req, booking, "decode booking failed")
		return uniformScores(endpoints, 1.0)
	}
	if err := ctx.Err(); err != nil {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer: scheduling cancelled during decode booking registration", "error", err.Error())
		s.rollbackPrefillReservation(ctx, cycleState, req, booking, "decode scheduling cancelled during booking registration")
		return uniformScores(endpoints, 1.0)
	}

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

	// Inject pre-computed tokens into the request body so the frontend
	// sidecar can skip redundant tokenization.
	setTokenizedPrompt(req, result.TokenData, logger)

	return uniformScores(endpoints, 1.0)
}

func (s *DynDecodeScorer) cleanupBooking(ctx context.Context, bookingID, reason string) bool {
	if bookingID == "" {
		return true
	}
	if lifecycle := findBookingLifecycle(bookingID); lifecycle != nil {
		return lifecycle.cleanup(ctx, reason)
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
	if booking != nil {
		if findBookingLifecycle(booking.ID) != nil || booking.PrefillReserved {
			s.cleanupBooking(ctx, booking.ID, reason)
		}
		booking.PrefillReserved = false
		cycleState.Write(BookingStateKey, booking)
	}
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
	if request != nil {
		if request.Headers == nil {
			request.Headers = map[string]string{}
		}
		request.Headers[RoutingModeHeader] = "aggregated"
		delete(request.Headers, WorkerIDHeader)
		delete(request.Headers, DpRankHeader)
		delete(request.Headers, PrefillWorkerIDHeader)
		delete(request.Headers, PrefillDpRankHeader)
	}
}

// ResponseBody handles streaming chunks and end-of-stream cleanup.
// On the first token it marks prefill as complete; on EndOfStream it frees the request.
func (s *DynDecodeScorer) ResponseBody(ctx context.Context, request *schedtypes.InferenceRequest, response *rc.Response, _ *fwkdl.EndpointMetadata) {
	bookingID := bookingIDFromRequest(request)
	if bookingID == "" || response == nil {
		return
	}

	lifecycle := findBookingLifecycle(bookingID)
	if lifecycle == nil {
		// Only Score creates controller-owned booking lifecycles. Do not let an
		// inbound header create router bookkeeping on a response-only path.
		return
	}
	if response.EndOfStream {
		lifecycle.cleanup(ctx, "response end of stream")
		return
	}
	if request.Headers[RoutingModeHeader] != "disaggregated" {
		return
	}

	lifecycle.startPrefillMarker(s.markPrefillComplete, log.FromContext(ctx), request.RequestId)
}
