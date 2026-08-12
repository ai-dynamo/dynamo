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

// Package disagg implements disaggregated prefill/decode serving plugins for Dynamo EPP.
//
// The disaggregated architecture splits inference into two phases:
//   - Prefill: processes the input prompt (compute-heavy, parallelizable)
//   - Decode: generates tokens autoregressively (memory-bound, sequential)
//
// This package provides three plugins:
//   - DisaggProfileHandler: orchestrates prefill→decode profile execution
//   - DynPrefillScorer: selects prefill workers via Dynamo FFI
//   - DynDecodeScorer: selects decode workers via Dynamo FFI
package disagg

import (
	"context"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	log "sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	PrefillProfileName = "prefill"
	DecodeProfileName  = "decode"

	// PrefillEnabledStateKey tracks whether this request should use disaggregated routing.
	PrefillEnabledStateKey = plugins.StateKey("disagg-prefill-enabled")
	BookingStateKey        = plugins.StateKey("dynamo-epp-booking")
	BookingIDHeader        = "x-dynamo-epp-booking-id"
)

// PrefillEnabledState stores whether prefill is enabled for the current scheduling cycle.
type PrefillEnabledState struct {
	Enabled bool
}

// Clone implements plugins.StateData.
func (s *PrefillEnabledState) Clone() plugins.StateData {
	return &PrefillEnabledState{Enabled: s.Enabled}
}

// readPrefillEnabled reads the PrefillEnabledState from CycleState.
func readPrefillEnabled(cycleState *schedtypes.CycleState) bool {
	state, err := schedtypes.ReadCycleStateKey[*PrefillEnabledState](cycleState, PrefillEnabledStateKey)
	if err == nil && state != nil {
		return state.Enabled
	}
	return false
}

// BookingState owns the controller-generated ID used for router bookkeeping.
type BookingState struct {
	ID              string
	PrefillReserved bool
}

// Clone implements plugins.StateData.
func (s *BookingState) Clone() plugins.StateData {
	if s == nil {
		return &BookingState{}
	}
	return &BookingState{ID: s.ID, PrefillReserved: s.PrefillReserved}
}

func readBookingState(cycleState *schedtypes.CycleState) (*BookingState, bool) {
	state, err := schedtypes.ReadCycleStateKey[*BookingState](cycleState, BookingStateKey)
	if err != nil || state == nil {
		return nil, false
	}
	if _, err := uuid.Parse(state.ID); err != nil {
		return nil, false
	}
	return state, true
}

func ensureBookingState(cycleState *schedtypes.CycleState) *BookingState {
	if state, ok := readBookingState(cycleState); ok {
		return state
	}
	state := &BookingState{ID: uuid.NewString()}
	cycleState.Write(BookingStateKey, state)
	return state
}

func attachBookingID(request *schedtypes.InferenceRequest, bookingID string) {
	if request == nil {
		return
	}
	if request.Headers == nil {
		request.Headers = map[string]string{}
	}
	request.Headers[BookingIDHeader] = bookingID
}

func bookingIDFromRequest(request *schedtypes.InferenceRequest) string {
	if request == nil || request.Headers == nil {
		return ""
	}
	bookingID := request.Headers[BookingIDHeader]
	if _, err := uuid.Parse(bookingID); err != nil {
		return ""
	}
	return bookingID
}

// buildRequestJSON builds an OpenAI-compatible JSON string from a GAIE LLMRequest.
func buildRequestJSON(req *schedtypes.InferenceRequest) (string, error) {
	return dynscorer.BuildOpenAIRequestJSON(req)
}

// serializeEndpoints converts endpoints to a JSON string for the FFI filter.
func serializeEndpoints(endpoints []schedtypes.Endpoint) string {
	if len(endpoints) == 0 {
		return ""
	}
	pj, err := dynscorer.SerializeEndpointsToJSON(endpoints)
	if err != nil {
		return ""
	}
	return pj
}

// uniformScores returns a score map with the same score for every endpoint.
func uniformScores(endpoints []schedtypes.Endpoint, score float64) map[schedtypes.Endpoint]float64 {
	out := make(map[schedtypes.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		out[ep] = score
	}
	return out
}

// setTokenizedPrompt stores pre-computed token IDs on the LLMRequest and
// injects nvext.token_data into the PayloadMap so it is forwarded to the
// worker in the request body.
//
// The GAIE framework re-serializes the PayloadMap after scheduling/PreRequest
// plugins run (PR #2854), so this mutation is included in the forwarded body.
func setTokenizedPrompt(req *schedtypes.InferenceRequest, tokens []int64, logger logr.Logger) {
	if req == nil || len(tokens) == 0 {
		logger.V(logutil.DEFAULT).Info("[EPP-INJECT] No tokens to inject (empty token list)")
		return
	}

	tokenIDs := make([]uint32, len(tokens))
	for i, t := range tokens {
		tokenIDs[i] = uint32(t)
	}

	if req.Body != nil {
		req.Body.TokenizedPrompt = &schedtypes.TokenizedPrompt{
			TokenIDs: tokenIDs,
		}
	}

	// Inject into the PayloadMap so the body includes nvext.token_data.
	payloadInjected := false
	if req.Body != nil {
		if pm, ok := req.Body.Payload.(fwkrh.PayloadMap); ok {
			nvext, _ := pm["nvext"].(map[string]any)
			if nvext == nil {
				nvext = map[string]any{}
			}
			nvext["token_data"] = tokenIDs
			pm["nvext"] = nvext
			payloadInjected = true
		}
	}

	if payloadInjected {
		logger.V(logutil.DEFAULT).Info("[EPP-INJECT] Injected pre-computed tokens into request body nvext.token_data",
			"tokenCount", len(tokenIDs),
			"requestId", req.RequestId)
	} else {
		logger.V(logutil.DEFAULT).Error(nil, "[EPP-INJECT] Failed to inject nvext.token_data: Payload is not a PayloadMap — sidecar will re-tokenize",
			"tokenCount", len(tokenIDs),
			"requestId", req.RequestId)
	}
}

func getEnvBoolOrDefault(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		switch strings.ToLower(v) {
		case "true", "1", "yes", "on":
			return true
		case "false", "0", "no", "off":
			return false
		}
	}
	return def
}

var enforceDisaggDeprecationOnce sync.Once

const (
	prefillMarkMaxAttempts       = 3
	prefillMarkRetryBackoff      = 100 * time.Millisecond
	cleanupMaxAttempts           = 3
	cleanupRetryBackoff          = 100 * time.Millisecond
	maxConcurrentBookingCleanups = 32
)

var (
	bookingLifecycles   sync.Map
	bookingCleanupSlots = make(chan struct{}, maxConcurrentBookingCleanups)
)

// bookingLifecycle owns cleanup for one EPP booking across the prefill scorer,
// decode scorer, and response callbacks. A booking has exactly one cleanup owner
// even when EOS, decode registration, and context cancellation race.
type bookingLifecycle struct {
	bookingID   string
	freeBooking func(string) error

	mu                     sync.Mutex
	cleanupStarted         bool
	cleanupSucceeded       bool
	cleanupExhausted       bool
	decodeRegistrationDone chan struct{}
	decodeRegistrationOpen bool

	stopCancellation func() bool
	stopMarker       context.CancelFunc
	markerDone       chan struct{}
	cleanupDone      chan struct{}
}

func registerBookingLifecycle(bookingID string, freeBooking func(string) error) *bookingLifecycle {
	lifecycle := &bookingLifecycle{
		bookingID:   bookingID,
		freeBooking: freeBooking,
	}
	actual, loaded := bookingLifecycles.LoadOrStore(bookingID, lifecycle)
	if loaded {
		return actual.(*bookingLifecycle)
	}
	return lifecycle
}

func findBookingLifecycle(bookingID string) *bookingLifecycle {
	lifecycle, ok := bookingLifecycles.Load(bookingID)
	if !ok {
		return nil
	}
	return lifecycle.(*bookingLifecycle)
}

// startDecodeRegistration installs a barrier before adding the decode booking.
// Cleanup waits on this barrier so cancellation cannot free the booking before
// add_request has either installed it or reported failure.
func (l *bookingLifecycle) startDecodeRegistration() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.cleanupStarted || l.decodeRegistrationDone != nil {
		return false
	}
	l.decodeRegistrationDone = make(chan struct{})
	l.decodeRegistrationOpen = true
	return true
}

func (l *bookingLifecycle) finishDecodeRegistration() {
	l.mu.Lock()
	if !l.decodeRegistrationOpen {
		l.mu.Unlock()
		return
	}
	l.decodeRegistrationOpen = false
	done := l.decodeRegistrationDone
	l.mu.Unlock()
	close(done)
}

func (l *bookingLifecycle) armCancellation(ctx context.Context) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.cleanupStarted || l.stopCancellation != nil {
		return
	}
	l.stopCancellation = context.AfterFunc(ctx, func() {
		l.cleanup(ctx, "request context cancelled")
	})
}

// startPrefillMarker makes first-token bookkeeping bounded and independent of
// the response callback. EOS or request cancellation stops any pending retry.
func (l *bookingLifecycle) startPrefillMarker(markPrefillComplete func(string) error, logger logr.Logger, requestID string) {
	l.mu.Lock()
	if l.cleanupStarted || l.markerDone != nil {
		l.mu.Unlock()
		return
	}
	markerCtx, stopMarker := context.WithCancel(context.Background())
	markerDone := make(chan struct{})
	l.stopMarker = stopMarker
	l.markerDone = markerDone
	l.mu.Unlock()

	go func() {
		defer close(markerDone)
		for attempt := 1; attempt <= prefillMarkMaxAttempts; attempt++ {
			if markerCtx.Err() != nil {
				return
			}
			if err := markPrefillComplete(l.bookingID); err == nil {
				logger.V(logutil.VERBOSE).Info("DynDecodeScorer ResponseBody: marked prefill complete",
					"bookingID", l.bookingID, "requestID", requestID, "attempt", attempt)
				return
			} else {
				logger.V(logutil.DEFAULT).Error(err, "DynDecodeScorer ResponseBody: failed to mark prefill complete",
					"bookingID", l.bookingID, "requestID", requestID, "attempt", attempt)
			}
			if attempt == prefillMarkMaxAttempts {
				return
			}

			timer := time.NewTimer(prefillMarkRetryBackoff * time.Duration(attempt))
			select {
			case <-markerCtx.Done():
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				return
			case <-timer.C:
			}
		}
	}()
}

// cleanup stops cancellation and marker ownership, waits for in-flight work,
// and retries free_request on a bounded executor. A terminal failure remains in
// bookingLifecycles as a tombstone so a duplicate cleanup cannot hide it.
func (l *bookingLifecycle) cleanup(ctx context.Context, reason string) bool {
	l.mu.Lock()
	if l.cleanupStarted {
		l.mu.Unlock()
		return false
	}
	l.cleanupStarted = true
	stopCancellation := l.stopCancellation
	stopMarker := l.stopMarker
	markerDone := l.markerDone
	decodeRegistrationDone := l.decodeRegistrationDone
	cleanupDone := make(chan struct{})
	l.cleanupDone = cleanupDone
	l.mu.Unlock()

	if stopCancellation != nil {
		stopCancellation()
	}
	if stopMarker != nil {
		stopMarker()
	}

	go func() {
		if markerDone != nil {
			<-markerDone
		}
		if decodeRegistrationDone != nil {
			<-decodeRegistrationDone
		}

		logger := log.FromContext(ctx)
		for attempt := 1; attempt <= cleanupMaxAttempts; attempt++ {
			bookingCleanupSlots <- struct{}{}
			err := l.freeBooking(l.bookingID)
			<-bookingCleanupSlots
			if err == nil {
				l.mu.Lock()
				l.cleanupSucceeded = true
				l.mu.Unlock()
				logger.V(logutil.VERBOSE).Info("Dynamo EPP booking cleaned up",
					"bookingID", l.bookingID, "reason", reason, "attempt", attempt)
				close(cleanupDone)
				bookingLifecycles.Delete(l.bookingID)
				return
			}

			logger.V(logutil.DEFAULT).Error(err, "Dynamo EPP booking cleanup failed",
				"bookingID", l.bookingID, "reason", reason, "attempt", attempt)
			if attempt == cleanupMaxAttempts {
				l.mu.Lock()
				l.cleanupExhausted = true
				l.mu.Unlock()
				logger.V(logutil.DEFAULT).Error(err, "Dynamo EPP booking cleanup exhausted retries; retaining tombstone",
					"bookingID", l.bookingID, "reason", reason, "attempts", cleanupMaxAttempts)
				close(cleanupDone)
				return
			}
			time.Sleep(cleanupRetryBackoff * time.Duration(attempt))
		}
	}()
	return true
}

func (l *bookingLifecycle) markerComplete() <-chan struct{} {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.markerDone
}

func (l *bookingLifecycle) cleanupComplete() <-chan struct{} {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.cleanupDone
}

func warnDeprecatedEnforceDisagg(logger logr.Logger) {
	if getEnvBoolOrDefault("DYN_ENFORCE_DISAGG", false) {
		enforceDisaggDeprecationOnce.Do(func() {
			logger.Info("DYN_ENFORCE_DISAGG is deprecated and ignored; routing topology and readiness come from registered worker types")
		})
	}
}
