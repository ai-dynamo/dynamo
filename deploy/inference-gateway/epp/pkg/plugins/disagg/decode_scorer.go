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
	"time"

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

	WorkerIDHeader             = "x-dynamo-worker-instance-id"
	PrefillWorkerIDHeader      = "x-dynamo-prefill-instance-id"
	DpRankHeader               = "x-dynamo-dp-rank"
	PrefillDpRankHeader        = "x-dynamo-prefill-dp-rank"
	RoutingModeHeader          = "x-dynamo-routing-mode"
	PrefillReservationIDHeader = "x-dynamo-epp-prefill-reservation-id"

	decodeStateKey = "dynamo-decode-routing-state"

	prefillCleanupRetryAttempts = 3
	prefillCleanupRetryDelay    = 100 * time.Millisecond
	prefillCleanupRetryWorkers  = 8
)

var (
	prefillCleanupRetries sync.Map
	prefillCleanupSlots   = make(chan struct{}, prefillCleanupRetryWorkers)
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
	CacheNamespace string
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
		CacheNamespace: s.CacheNamespace,
	}
	if s.TokenData != nil {
		clone.TokenData = make([]int64, len(s.TokenData))
		copy(clone.TokenData, s.TokenData)
	}
	return clone
}

func prefillReservationID(request *schedtypes.InferenceRequest) string {
	if request == nil || request.Headers == nil {
		return ""
	}
	reservationID := request.Headers[PrefillReservationIDHeader]
	if !isPrefillReservationID(reservationID) {
		return ""
	}
	return reservationID
}

func routerBookingID(request *schedtypes.InferenceRequest) string {
	if request == nil {
		return ""
	}
	if reservationID := prefillReservationID(request); reservationID != "" {
		return reservationID
	}
	return request.RequestId
}

func forceAggregatedRouting(request *schedtypes.InferenceRequest) {
	if request == nil {
		return
	}
	if request.Headers == nil {
		request.Headers = map[string]string{}
	}
	request.Headers[RoutingModeHeader] = "aggregated"
	delete(request.Headers, PrefillWorkerIDHeader)
	delete(request.Headers, PrefillDpRankHeader)
}

func cleanupRouterBooking(ctx context.Context, request *schedtypes.InferenceRequest, bookingID string, reason string) bool {
	if bookingID == "" {
		return true
	}
	logger := log.FromContext(ctx)
	if err := dynscorer.CallFreeRequest(bookingID); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Dynamo EPP: failed to roll back router booking",
			"bookingID", bookingID, "reason", reason)
		return false
	}
	if request != nil && prefillReservationID(request) == bookingID {
		delete(request.Headers, PrefillReservationIDHeader)
		forceAggregatedRouting(request)
	}
	return true
}

func retryPrefillReservationCleanup(bookingID string, reason string) {
	if bookingID == "" {
		return
	}
	if _, alreadyQueued := prefillCleanupRetries.LoadOrStore(bookingID, struct{}{}); alreadyQueued {
		return
	}
	select {
	case prefillCleanupSlots <- struct{}{}:
	default:
		prefillCleanupRetries.Delete(bookingID)
		log.Log.WithName(DynDecodeScorerType).V(logutil.VERBOSE).Info(
			"Dynamo EPP: router cleanup retry queue is saturated; relying on stale-request expiry",
			"bookingID", bookingID, "reason", reason)
		return
	}
	go func() {
		defer func() {
			<-prefillCleanupSlots
			prefillCleanupRetries.Delete(bookingID)
		}()
		logger := log.Log.WithName(DynDecodeScorerType)
		delay := prefillCleanupRetryDelay
		var lastErr error
		for attempt := 1; attempt <= prefillCleanupRetryAttempts; attempt++ {
			time.Sleep(delay)
			if err := dynscorer.CallFreeRequest(bookingID); err == nil {
				logger.V(logutil.VERBOSE).Info("Dynamo EPP: router cleanup retry succeeded",
					"bookingID", bookingID, "reason", reason, "attempt", attempt)
				return
			} else {
				lastErr = err
			}
			delay *= 2
		}
		logger.V(logutil.DEFAULT).Error(lastErr, "Dynamo EPP: router cleanup retries exhausted",
			"bookingID", bookingID, "reason", reason)
	}()
}

func cleanupCancelledRouterBooking(ctx context.Context, request *schedtypes.InferenceRequest, bookingID string, reason string) {
	if cleanupRouterBooking(ctx, request, bookingID, reason) {
		return
	}
	if prefillReservationID(request) == bookingID {
		retryPrefillReservationCleanup(bookingID, reason)
	}
}

func finalizePrefillRollback(cycleState *schedtypes.CycleState, request *schedtypes.InferenceRequest, cleanupSucceeded bool) bool {
	if cleanupSucceeded {
		clearPrefillReservation(cycleState)
		return true
	}
	// Keep the reservation state and ID available for another cleanup attempt,
	// but strip the selected prefill worker so a fallback cannot be forwarded as
	// a disaggregated request with stale routing metadata.
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
	forceAggregatedRouting(request)
	return false
}

func rollbackPrefillReservation(ctx context.Context, cycleState *schedtypes.CycleState, request *schedtypes.InferenceRequest, reason string) bool {
	reservation := readPrefillReservation(cycleState)
	bookingID := ""
	if reservation != nil {
		bookingID = reservation.ID
	}
	return finalizePrefillRollback(cycleState, request, cleanupRouterBooking(ctx, request, bookingID, reason))
}

func rollbackPrefillReservationAtTerminal(ctx context.Context, cycleState *schedtypes.CycleState, request *schedtypes.InferenceRequest, reason string) {
	reservation := readPrefillReservation(cycleState)
	bookingID := ""
	if reservation != nil {
		bookingID = reservation.ID
	}
	if !rollbackPrefillReservation(ctx, cycleState, request, reason) {
		retryPrefillReservationCleanup(bookingID, reason)
	}
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
		markPrefillComplete: dynscorer.CallMarkPrefillComplete,
		freeRequest:         dynscorer.CallFreeRequest,
	}
}

type prefillMarkAttemptState struct {
	stopContextCleanup func() bool
}

// DynDecodeScorer is a scorer plugin for the decode scheduling profile.
type DynDecodeScorer struct {
	typedName            plugins.TypedName
	pluginState          *plugins.PluginState
	prefillMarkAttempted sync.Map
	markPrefillComplete  func(string) error
	freeRequest          func(string) error
}

func (s *DynDecodeScorer) beginPrefillMarkAttempt(ctx context.Context, bookingID string) bool {
	if ctx.Err() != nil {
		return false
	}

	state := &prefillMarkAttemptState{}
	state.stopContextCleanup = context.AfterFunc(ctx, func() {
		// CompareAndDelete prevents an old cancelled callback from deleting a
		// newer lifecycle state if a booking ID is ever reused.
		s.prefillMarkAttempted.CompareAndDelete(bookingID, state)
	})

	if _, alreadyAttempted := s.prefillMarkAttempted.LoadOrStore(bookingID, state); alreadyAttempted {
		state.stopContextCleanup()
		return false
	}

	// Cancellation can race with registering the callback and publishing the
	// state. Remove the exact state synchronously so a cancelled queued chunk
	// cannot leave a marker behind or initiate a new FFI call.
	if ctx.Err() != nil {
		state.stopContextCleanup()
		s.prefillMarkAttempted.CompareAndDelete(bookingID, state)
		return false
	}

	return true
}

func (s *DynDecodeScorer) clearPrefillMarkAttempt(bookingID string) {
	stateValue, ok := s.prefillMarkAttempted.LoadAndDelete(bookingID)
	if !ok {
		return
	}
	state := stateValue.(*prefillMarkAttemptState)
	state.stopContextCleanup()
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
	reservation := readPrefillReservation(cycleState)
	if reservation != nil && reservation.ID != "" {
		if req != nil {
			if req.Headers == nil {
				req.Headers = map[string]string{}
			}
			// Restore only a controller-owned ID recorded in CycleState. This
			// also preserves ownership when a failed rollback falls back to
			// aggregated serving and needs a later lifecycle callback to retry.
			req.Headers[PrefillReservationIDHeader] = reservation.ID
		}
	} else if req != nil && req.Headers != nil {
		// The incoming header is not trusted unless the current scheduling
		// cycle recorded the matching reservation.
		delete(req.Headers, PrefillReservationIDHeader)
		delete(req.Headers, PrefillWorkerIDHeader)
		delete(req.Headers, PrefillDpRankHeader)
	}

	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: failed to build request")
		rollbackPrefillReservation(ctx, cycleState, req, "decode request serialization failed")
		return uniformScores(endpoints, 1.0)
	}

	endpointsJSON := serializeEndpoints(endpoints)
	logger.V(logutil.DEFAULT).Info("DynDecodeScorer: endpoints received for scoring",
		"endpointCount", len(endpoints),
		"endpointsJSON", string(endpointsJSON))

	result, err := dynscorer.CallRouteDecodeRequest(requestJSON, endpointsJSON, isDisaggregated)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer: FFI decode routing failed")
		rollbackPrefillReservation(ctx, cycleState, req, "decode routing failed")
		return uniformScores(endpoints, 1.0)
	}
	if err := ctx.Err(); err != nil {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer: scheduling cancelled after decode routing",
			"error", err.Error())
		rollbackPrefillReservationAtTerminal(ctx, cycleState, req, "scheduling cancelled after decode routing")
		return uniformScores(endpoints, 1.0)
	}

	workerIDStr := fmt.Sprintf("%d", result.WorkerID)
	dpRankStr := strconv.FormatUint(uint64(result.DpRank), 10)
	logger.V(logutil.DEFAULT).Info("[EPP-SCORER] FFI returned tokens from C bindings tokenization",
		"decodeWorkerID", workerIDStr,
		"decodeDpRank", result.DpRank,
		"isDisaggregated", isDisaggregated,
		"tokenCount", len(result.TokenData))

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
		} else {
			logger.V(logutil.DEFAULT).Error(nil,
				"DynDecodeScorer: x-dynamo-prefill-instance-id header missing — DynPrefillScorer did not set it")
		}
	} else {
		req.Headers[RoutingModeHeader] = "aggregated"
		delete(req.Headers, PrefillWorkerIDHeader)
		delete(req.Headers, PrefillDpRankHeader)
	}

	// Store routing state for PreRequest bookkeeping. Disaggregated requests use
	// the EPP-minted reservation ID so reused client request IDs cannot collide.
	bookingID := routerBookingID(req)
	if bookingID != "" {
		routingState := &DecodeRoutingState{
			BookingID:      bookingID,
			WorkerID:       workerIDStr,
			DpRank:         result.DpRank,
			TokenData:      result.TokenData,
			CacheNamespace: result.CacheNamespace,
		}
		s.pluginState.Write(bookingID, plugins.StateKey(decodeStateKey), routingState)
	}

	// Inject pre-computed tokens into the request body so the frontend
	// sidecar can skip redundant tokenization.
	setTokenizedPrompt(req, result.TokenData, logger)

	return uniformScores(endpoints, 1.0)
}

// PreRequest registers the request with the Dynamo router's bookkeeping.
func (s *DynDecodeScorer) PreRequest(ctx context.Context, request *schedtypes.InferenceRequest, _ *schedtypes.SchedulingResult) {
	logger := log.FromContext(ctx)

	if request == nil {
		return
	}
	bookingID := routerBookingID(request)
	if bookingID == "" {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: no booking ID, skipping")
		return
	}

	state, err := plugins.ReadPluginStateKey[*DecodeRoutingState](
		s.pluginState, bookingID, plugins.StateKey(decodeStateKey),
	)
	s.pluginState.Delete(bookingID)

	if err != nil {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: no routing state found",
			"requestID", request.RequestId)
		cleanupRouterBooking(ctx, request, prefillReservationID(request), "decode routing state missing")
		return
	}

	var workerIDUint uint64
	if _, parseErr := fmt.Sscanf(state.WorkerID, "%d", &workerIDUint); parseErr != nil {
		logger.V(logutil.DEFAULT).Error(parseErr, "DynDecodeScorer PreRequest: invalid worker ID",
			"requestID", request.RequestId, "workerID", state.WorkerID)
		cleanupRouterBooking(ctx, request, state.BookingID, "decode worker ID invalid")
		return
	}
	if err := ctx.Err(); err != nil {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: request cancelled before decode booking",
			"requestID", request.RequestId)
		cleanupCancelledRouterBooking(ctx, request, state.BookingID, "request cancelled before decode booking")
		return
	}

	if addErr := dynscorer.CallAddRequest(
		state.BookingID,
		state.TokenData,
		workerIDUint,
		state.DpRank,
		state.CacheNamespace,
	); addErr != nil {
		logger.V(logutil.DEFAULT).Error(addErr, "DynDecodeScorer PreRequest: failed to add request",
			"requestID", request.RequestId)
		cleanupRouterBooking(ctx, request, state.BookingID, "decode booking failed")
		return
	}
	if err := ctx.Err(); err != nil {
		logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: request cancelled after decode booking",
			"requestID", request.RequestId)
		cleanupCancelledRouterBooking(ctx, request, state.BookingID, "request cancelled after decode booking")
		return
	}

	logger.V(logutil.VERBOSE).Info("DynDecodeScorer PreRequest: registered request",
		"requestID", request.RequestId,
		"workerID", state.WorkerID,
		"dpRank", state.DpRank,
		"hasCacheNamespace", state.CacheNamespace != "",
		"tokenCount", len(state.TokenData))
}

// ResponseBody handles response callbacks and end-of-stream cleanup.
// On the first non-terminal body callback it releases prefill load; on EndOfStream it frees the request.
func (s *DynDecodeScorer) ResponseBody(ctx context.Context, request *schedtypes.InferenceRequest, response *rc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	bookingID := routerBookingID(request)
	if bookingID == "" {
		return
	}

	logger := log.FromContext(ctx)

	// A single terminal callback needs only free: free is idempotent and also
	// releases any remaining prefill load.
	if response.EndOfStream {
		s.clearPrefillMarkAttempt(bookingID)
		if err := s.freeRequest(bookingID); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to free request",
				"requestID", request.RequestId)
			if prefillReservationID(request) == bookingID {
				retryPrefillReservationCleanup(bookingID, "end-of-stream cleanup failed")
			}
		} else {
			logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: freed request",
				"requestID", request.RequestId)
		}
		return
	}

	// The framework does not expose token contents here. Treat the first
	// non-terminal body callback as the earliest observable prefill completion.
	// The marker records an attempt, not success: a failed synchronous FFI call
	// must not be retried on every streamed chunk. End-of-stream free releases
	// any remaining prefill load.
	if s.beginPrefillMarkAttempt(ctx, bookingID) {
		if err := s.markPrefillComplete(bookingID); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to mark prefill complete",
				"requestID", request.RequestId)
		} else {
			logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: marked prefill complete",
				"requestID", request.RequestId)
		}
	}
}
