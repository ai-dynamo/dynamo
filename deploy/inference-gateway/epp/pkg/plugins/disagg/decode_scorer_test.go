/*
Copyright 2026 NVIDIA Corporation.

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
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestResponseBodyMarkFailureIsAttemptedOnceAcrossChunks(t *testing.T) {
	request := responseBodyTestRequest(t)
	bookingID := routerBookingID(request)
	scorer := NewDynDecodeScorer(context.Background())

	var markCalls atomic.Int32
	var freeCalls atomic.Int32
	scorer.markPrefillComplete = func(gotBookingID string) error {
		if gotBookingID != bookingID {
			t.Fatalf("mark used booking ID %q, want %q", gotBookingID, bookingID)
		}
		markCalls.Add(1)
		return errors.New("injected mark failure")
	}
	scorer.freeRequest = func(gotBookingID string) error {
		if gotBookingID != bookingID {
			t.Fatalf("free used booking ID %q, want %q", gotBookingID, bookingID)
		}
		freeCalls.Add(1)
		return nil
	}

	for range 3 {
		scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	}

	if got := markCalls.Load(); got != 1 {
		t.Fatalf("mark call count = %d, want 1", got)
	}
	if _, ok := scorer.prefillMarkAttempted.Load(bookingID); !ok {
		t.Fatal("expected failed mark attempt to remain recorded until terminal cleanup")
	}

	scorer.ResponseBody(context.Background(), request, &rc.Response{EndOfStream: true}, nil)

	if got := freeCalls.Load(); got != 1 {
		t.Fatalf("free call count = %d, want 1", got)
	}
	if _, ok := scorer.prefillMarkAttempted.Load(bookingID); ok {
		t.Fatal("expected end-of-stream to remove mark attempt state")
	}
}

func TestResponseBodyContextCancellationClearsAttemptAndDoesNotRecreate(t *testing.T) {
	request := responseBodyTestRequest(t)
	bookingID := routerBookingID(request)
	scorer := NewDynDecodeScorer(context.Background())

	var markCalls atomic.Int32
	scorer.markPrefillComplete = func(string) error {
		markCalls.Add(1)
		return nil
	}
	scorer.freeRequest = func(string) error { return nil }

	ctx, cancel := context.WithCancel(context.Background())
	scorer.ResponseBody(ctx, request, &rc.Response{}, nil)
	if _, ok := scorer.prefillMarkAttempted.Load(bookingID); !ok {
		t.Fatal("expected first non-terminal callback to record a mark attempt")
	}

	cancel()
	waitForMarkAttemptState(t, scorer, bookingID, false)

	for range 3 {
		scorer.ResponseBody(ctx, request, &rc.Response{}, nil)
	}
	if got := markCalls.Load(); got != 1 {
		t.Fatalf("mark call count after cancellation = %d, want 1", got)
	}
	if _, ok := scorer.prefillMarkAttempted.Load(bookingID); ok {
		t.Fatal("expected cancelled callbacks not to recreate mark attempt state")
	}
}

func TestResponseBodyConcurrentChunksAttemptMarkOnceAndEOSCleansUp(t *testing.T) {
	request := responseBodyTestRequest(t)
	bookingID := routerBookingID(request)
	scorer := NewDynDecodeScorer(context.Background())

	var markCalls atomic.Int32
	var freeCalls atomic.Int32
	scorer.markPrefillComplete = func(string) error {
		markCalls.Add(1)
		return nil
	}
	scorer.freeRequest = func(string) error {
		freeCalls.Add(1)
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	const chunkCount = 32
	start := make(chan struct{})
	var workers sync.WaitGroup
	workers.Add(chunkCount)
	for range chunkCount {
		go func() {
			defer workers.Done()
			<-start
			scorer.ResponseBody(ctx, request, &rc.Response{}, nil)
		}()
	}
	close(start)
	workers.Wait()

	if got := markCalls.Load(); got != 1 {
		t.Fatalf("concurrent mark call count = %d, want 1", got)
	}
	if _, ok := scorer.prefillMarkAttempted.Load(bookingID); !ok {
		t.Fatal("expected concurrent callbacks to retain one mark attempt state")
	}

	scorer.ResponseBody(ctx, request, &rc.Response{EndOfStream: true}, nil)
	if got := freeCalls.Load(); got != 1 {
		t.Fatalf("free call count = %d, want 1", got)
	}
	if _, ok := scorer.prefillMarkAttempted.Load(bookingID); ok {
		t.Fatal("expected end-of-stream to remove concurrent mark attempt state")
	}

	// Cancelling after EOS exercises the stopped context callback; it must not
	// disturb terminal cleanup or recreate state.
	cancel()
	if _, ok := scorer.prefillMarkAttempted.Load(bookingID); ok {
		t.Fatal("expected cancelled context after EOS not to recreate mark attempt state")
	}
}

func TestFailedPrefillRollbackForcesAggregatedRoutingAndKeepsReservation(t *testing.T) {
	reservationID := mustPrefillReservationID(t)
	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	cycleState.Write(PrefillReservationStateKey, &PrefillReservationState{
		ID:        reservationID,
		WorkerID:  "7",
		DpRank:    3,
		HasDpRank: true,
	})
	request := &schedtypes.InferenceRequest{
		RequestId: "client-request-id",
		Headers: map[string]string{
			PrefillReservationIDHeader: reservationID,
			PrefillWorkerIDHeader:      "7",
			PrefillDpRankHeader:        "3",
			RoutingModeHeader:          "disaggregated",
		},
	}

	if rollbackPrefillReservation(context.Background(), cycleState, request, "test uninitialized router cleanup") {
		t.Fatal("expected failed cleanup to report an incomplete rollback")
	}

	if readPrefillEnabled(cycleState) {
		t.Fatal("expected failed cleanup to disable disaggregated fallback")
	}
	reservation := readPrefillReservation(cycleState)
	if reservation == nil || reservation.ID != reservationID {
		t.Fatalf("reservation state = %#v, want ID %q retained", reservation, reservationID)
	}
	if got := request.Headers[PrefillReservationIDHeader]; got != reservationID {
		t.Fatalf("reservation ID = %q, want %q", got, reservationID)
	}
	if got := routerBookingID(request); got != reservationID {
		t.Fatalf("router booking ID = %q, want retained reservation ID %q", got, reservationID)
	}
	if got := request.Headers[RoutingModeHeader]; got != "aggregated" {
		t.Fatalf("routing mode = %q, want aggregated", got)
	}
	if _, ok := request.Headers[PrefillWorkerIDHeader]; ok {
		t.Fatal("expected aggregated fallback to remove prefill worker header")
	}
	if _, ok := request.Headers[PrefillDpRankHeader]; ok {
		t.Fatal("expected aggregated fallback to remove prefill DP-rank header")
	}
}

func responseBodyTestRequest(t *testing.T) *schedtypes.InferenceRequest {
	t.Helper()
	return &schedtypes.InferenceRequest{
		RequestId: "client-request-id",
		Headers: map[string]string{
			PrefillReservationIDHeader: mustPrefillReservationID(t),
			RoutingModeHeader:          "disaggregated",
		},
	}
}

func waitForMarkAttemptState(t *testing.T, scorer *DynDecodeScorer, bookingID string, wantPresent bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		_, present := scorer.prefillMarkAttempted.Load(bookingID)
		if present == wantPresent {
			return
		}
		time.Sleep(time.Millisecond)
	}
	_, present := scorer.prefillMarkAttempted.Load(bookingID)
	t.Fatalf("mark attempt state present = %t, want %t", present, wantPresent)
}
