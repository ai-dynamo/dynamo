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
	"time"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	// DynPrefillScorerType is the plugin type registered in the plugin registry.
	DynPrefillScorerType = "dyn-prefill-scorer"

	defaultPrefillReservationAdmissionTimeout = 60 * time.Second
	defaultMaxPrefillReservations             = 64
)

// compile-time type assertion
var _ schedtypes.Scorer = &DynPrefillScorer{}

// DynPrefillScorerConfig holds the configuration for the DynPrefillScorer plugin.
type DynPrefillScorerConfig struct {
	// ReservationTimeoutSeconds bounds waiting for scheduler admission. The
	// request context deadline still wins when it is shorter.
	ReservationTimeoutSeconds int `json:"reservationTimeoutSeconds"`
	// MaxConcurrentReservations bounds the number of blocking CGO admissions.
	MaxConcurrentReservations int `json:"maxConcurrentReservations"`
}

// DynPrefillScorerFactory defines the factory function for DynPrefillScorer.
func DynPrefillScorerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	cfg := DynPrefillScorerConfig{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", DynPrefillScorerType, err)
		}
	}
	if cfg.ReservationTimeoutSeconds < 0 || cfg.MaxConcurrentReservations < 0 {
		return nil, fmt.Errorf("%s reservation timeout and concurrency must not be negative", DynPrefillScorerType)
	}

	if err := dynscorer.InitFFI(); err != nil {
		return nil, fmt.Errorf("Dynamo FFI init for prefill scorer failed: %w", err)
	}

	return newDynPrefillScorer(cfg).WithName(name), nil
}

// NewDynPrefillScorer initializes a new DynPrefillScorer.
func NewDynPrefillScorer() *DynPrefillScorer {
	return newDynPrefillScorer(DynPrefillScorerConfig{})
}

func newDynPrefillScorer(cfg DynPrefillScorerConfig) *DynPrefillScorer {
	reservationTimeout := defaultPrefillReservationAdmissionTimeout
	if cfg.ReservationTimeoutSeconds > 0 {
		reservationTimeout = time.Duration(cfg.ReservationTimeoutSeconds) * time.Second
	}
	maxReservations := defaultMaxPrefillReservations
	if cfg.MaxConcurrentReservations > 0 {
		maxReservations = cfg.MaxConcurrentReservations
	}
	return &DynPrefillScorer{
		typedName:                   plugins.TypedName{Type: DynPrefillScorerType, Name: DynPrefillScorerType},
		beginPrefill:                dynscorer.CallBeginPrefillReservation,
		reservePrefill:              dynscorer.CallRoutePrefillRequestWithReservation,
		cancelPrefill:               dynscorer.CallCancelPrefillReservation,
		releasePrefill:              dynscorer.CallReleasePrefillReservation,
		freeBooking:                 dynscorer.CallFreeRequest,
		reservationAdmissionTimeout: reservationTimeout,
		reservationSlots:            make(chan struct{}, maxReservations),
	}
}

// DynPrefillScorer is a scorer plugin for the prefill scheduling profile.
type DynPrefillScorer struct {
	typedName                   plugins.TypedName
	beginPrefill                func(string) error
	reservePrefill              func(string, string, string) (*dynscorer.RoutingResult, error)
	cancelPrefill               func(string) error
	releasePrefill              func(string) error
	freeBooking                 func(string) error
	reservationAdmissionTimeout time.Duration
	reservationSlots            chan struct{}
}

type prefillReservationResult struct {
	result *dynscorer.RoutingResult
	err    error
}

func (s *DynPrefillScorer) admissionTimeout() time.Duration {
	if s.reservationAdmissionTimeout <= 0 {
		return defaultPrefillReservationAdmissionTimeout
	}
	return s.reservationAdmissionTimeout
}

func (s *DynPrefillScorer) acquireReservationSlot(ctx context.Context) (func(), bool) {
	if s.reservationSlots == nil {
		return func() {}, true
	}
	select {
	case s.reservationSlots <- struct{}{}:
		return func() { <-s.reservationSlots }, true
	case <-ctx.Done():
		return nil, false
	}
}

func (s *DynPrefillScorer) beginReservation(bookingID string) error {
	if s.beginPrefill == nil {
		return nil
	}
	return s.beginPrefill(bookingID)
}

func (s *DynPrefillScorer) releaseLatePrefillReservation(bookingID string) error {
	if s.releasePrefill == nil {
		return fmt.Errorf("prefill reservation release is not configured")
	}
	return s.releasePrefill(bookingID)
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
	if req == nil {
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}

	if err := ctx.Err(); err != nil {
		logger.V(logutil.VERBOSE).Info("DynPrefillScorer: scheduling already cancelled", "error", err.Error())
		return uniformScores(endpoints, 0)
	}
	if !readPrefillEnabled(cycleState) {
		logger.V(logutil.VERBOSE).Info("DynPrefillScorer: prefill not enabled, returning zero scores")
		return uniformScores(endpoints, 0)
	}

	booking := ensureBookingState(cycleState)
	attachBookingID(req, booking.ID)
	delete(req.Headers, PrefillWorkerIDHeader)
	delete(req.Headers, PrefillDpRankHeader)

	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: failed to build request")
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}

	endpointsJSON := serializeEndpoints(endpoints)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: endpoints received for scoring",
		"endpointCount", len(endpoints),
		"endpointsJSON", string(endpointsJSON))

	bookingID := booking.ID
	admissionCtx, cancelAdmission := context.WithTimeout(ctx, s.admissionTimeout())
	defer cancelAdmission()
	releaseSlot, acquired := s.acquireReservationSlot(admissionCtx)
	if !acquired {
		logger.V(logutil.DEFAULT).Info("DynPrefillScorer: prefill reservation admission budget exhausted",
			"bookingID", bookingID, "error", admissionCtx.Err().Error())
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}
	if err := s.beginReservation(bookingID); err != nil {
		releaseSlot()
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: failed to begin prefill reservation",
			"bookingID", bookingID)
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}

	resultCh := make(chan prefillReservationResult, 1)
	go func() {
		defer releaseSlot()
		result, err := s.reservePrefill(bookingID, requestJSON, endpointsJSON)
		resultCh <- prefillReservationResult{result: result, err: err}
	}()

	var result *dynscorer.RoutingResult
	select {
	case <-admissionCtx.Done():
		logger.V(logutil.VERBOSE).Info("DynPrefillScorer: scheduling cancelled during prefill reservation",
			"error", admissionCtx.Err().Error())
		if cancelErr := s.cancelPrefill(bookingID); cancelErr != nil {
			logger.V(logutil.DEFAULT).Error(cancelErr, "DynPrefillScorer: failed to cancel pending prefill reservation",
				"bookingID", bookingID)
		}
		go func(bookingID string) {
			// Rust can retain an active reservation after returning an error when
			// its first cleanup attempt fails. Release is idempotent for every
			// other late outcome.
			<-resultCh
			if cleanupErr := s.releaseLatePrefillReservation(bookingID); cleanupErr != nil {
				logger.V(logutil.DEFAULT).Error(cleanupErr, "DynPrefillScorer: failed to release late prefill reservation",
					"bookingID", bookingID)
			}
		}(bookingID)
		booking.PrefillReserved = false
		cycleState.Write(BookingStateKey, booking)
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	case outcome := <-resultCh:
		result = outcome.result
		err = outcome.err
	}
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: FFI prefill reservation failed")
		booking.PrefillReserved = false
		cycleState.Write(BookingStateKey, booking)
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}

	booking.PrefillReserved = true
	cycleState.Write(BookingStateKey, booking)
	prefillWorkerID := strconv.FormatUint(result.WorkerID, 10)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: prefill worker reserved",
		"bookingID", booking.ID,
		"prefillWorkerID", prefillWorkerID,
		"prefillDpRank", result.DpRank,
		"tokenCount", len(result.TokenData))

	req.Headers[PrefillWorkerIDHeader] = prefillWorkerID
	if result.DpRank != dynscorer.UnsetDpRank {
		req.Headers[PrefillDpRankHeader] = strconv.FormatUint(uint64(result.DpRank), 10)
	} else {
		delete(req.Headers, PrefillDpRankHeader)
	}

	lifecycle := registerBookingLifecycle(booking.ID, s.freeBooking)
	lifecycle.armCancellation(ctx)
	if err := ctx.Err(); err != nil {
		lifecycle.cleanup(ctx, "prefill scheduling cancelled after reservation")
		booking.PrefillReserved = false
		cycleState.Write(BookingStateKey, booking)
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}

	return uniformScores(endpoints, 1.0)
}
