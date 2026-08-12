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
	"testing"
	"time"

	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

func requestWithBooking(externalRequestID, bookingID string) *schedtypes.InferenceRequest {
	return &schedtypes.InferenceRequest{
		RequestId: externalRequestID,
		Headers:   map[string]string{BookingIDHeader: bookingID},
	}
}

func TestBookingStateDoesNotTrustExternalRequestID(t *testing.T) {
	externalRequestID := "shared-client-request-id"
	requestA := &schedtypes.InferenceRequest{
		RequestId: externalRequestID,
		Headers:   map[string]string{BookingIDHeader: "caller-controlled"},
	}
	requestB := &schedtypes.InferenceRequest{RequestId: externalRequestID}
	stateA := schedtypes.NewCycleState()
	stateB := schedtypes.NewCycleState()

	bookingA := ensureBookingState(stateA)
	bookingB := ensureBookingState(stateB)
	attachBookingID(requestA, bookingA.ID)
	attachBookingID(requestB, bookingB.ID)

	if bookingA.ID == externalRequestID || bookingB.ID == externalRequestID {
		t.Fatal("controller booking ID reused an external request ID")
	}
	if bookingA.ID == bookingB.ID {
		t.Fatal("independent scheduling cycles received the same booking ID")
	}
	if got := bookingIDFromRequest(requestA); got != bookingA.ID {
		t.Fatalf("request A booking header = %q, want %q", got, bookingA.ID)
	}
	if got := bookingIDFromRequest(requestB); got != bookingB.ID {
		t.Fatalf("request B booking header = %q, want %q", got, bookingB.ID)
	}
}

func TestResponseBodyRetriesPrefillMarkAndFreesTerminalResponse(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	request := requestWithBooking("external-request", bookingID)
	request.Headers[RoutingModeHeader] = "disaggregated"
	markCalls := 0
	freeCalls := 0
	scorer := &DynDecodeScorer{
		markPrefillComplete: func(got string) error {
			if got != bookingID {
				t.Fatalf("mark booking ID = %q, want %q", got, bookingID)
			}
			markCalls++
			if markCalls == 1 {
				return errors.New("transient mark failure")
			}
			return nil
		},
		freeBooking: func(got string) error {
			if got != bookingID {
				t.Fatalf("free booking ID = %q, want %q", got, bookingID)
			}
			freeCalls++
			return nil
		},
	}
	registerBookingLifecycle(bookingID, scorer.freeBooking)

	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)

	lifecycle := findBookingLifecycle(bookingID)
	if lifecycle == nil {
		t.Fatal("expected booking lifecycle after first response chunk")
	}
	select {
	case <-lifecycle.markerComplete():
	case <-time.After(time.Second):
		t.Fatal("prefill marker did not finish")
	}
	scorer.ResponseBody(context.Background(), request, &rc.Response{EndOfStream: true}, nil)
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("terminal booking cleanup did not finish")
	}

	if markCalls != 2 {
		t.Fatalf("mark calls = %d, want one retry then success", markCalls)
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestResponseBodyFreesWithoutMarkingEmptyTerminalResponse(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	markCalls := 0
	freeCalls := 0
	scorer := &DynDecodeScorer{
		markPrefillComplete: func(string) error {
			markCalls++
			return nil
		},
		freeBooking: func(string) error {
			freeCalls++
			return nil
		},
	}
	registerBookingLifecycle(bookingID, scorer.freeBooking)

	scorer.ResponseBody(
		context.Background(),
		requestWithBooking("external-request", bookingID),
		&rc.Response{EndOfStream: true},
		nil,
	)
	lifecycle := findBookingLifecycle(bookingID)
	if lifecycle == nil {
		t.Fatal("expected booking lifecycle for terminal response")
	}
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("terminal booking cleanup did not finish")
	}

	if markCalls != 0 {
		t.Fatalf("mark calls = %d, want 0", markCalls)
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestPrefillScoreCancelsPendingReservation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	reserveStarted := make(chan struct{})
	allowReserveReturn := make(chan struct{})
	cancelCalls := make(chan string, 1)
	releaseCalls := make(chan string, 1)
	combinedFreeCalls := 0
	scorer := &DynPrefillScorer{
		reservePrefill: func(string, string, string) (*dynscorer.RoutingResult, error) {
			close(reserveStarted)
			<-allowReserveReturn
			return &dynscorer.RoutingResult{WorkerID: 7}, nil
		},
		cancelPrefill: func(bookingID string) error {
			cancelCalls <- bookingID
			close(allowReserveReturn)
			return nil
		},
		releasePrefill: func(bookingID string) error {
			releaseCalls <- bookingID
			return nil
		},
		freeBooking: func(string) error {
			combinedFreeCalls++
			return nil
		},
	}
	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	req := &schedtypes.InferenceRequest{
		TargetModel: "model",
		Headers:     map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{"model": "model", "prompt": "hello"},
		},
	}

	scoresCh := make(chan map[schedtypes.Endpoint]float64, 1)
	go func() {
		scoresCh <- scorer.Score(ctx, cycleState, req, nil)
	}()
	<-reserveStarted
	cancel()

	if scores := <-scoresCh; len(scores) != 0 {
		t.Fatalf("scores = %v, want aggregate fallback with no prefill endpoints", scores)
	}
	bookingID := bookingIDFromRequest(req)
	if got := <-cancelCalls; got != bookingID {
		t.Fatalf("cancel booking ID = %q, want %q", got, bookingID)
	}
	select {
	case got := <-releaseCalls:
		if got != bookingID {
			t.Fatalf("late release booking ID = %q, want %q", got, bookingID)
		}
	case <-time.After(time.Second):
		t.Fatal("late successful reservation was not released")
	}
	if combinedFreeCalls != 0 {
		t.Fatalf("combined free_request calls = %d, want 0", combinedFreeCalls)
	}
	if readPrefillEnabled(cycleState) {
		t.Fatal("prefill remained enabled after reservation cancellation")
	}
}

func TestResponseBodyBoundsPersistentPrefillMarkRetries(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	markCalls := 0
	freeCalls := 0
	scorer := &DynDecodeScorer{
		markPrefillComplete: func(string) error {
			markCalls++
			return errors.New("persistent mark failure")
		},
		freeBooking: func(string) error {
			freeCalls++
			return nil
		},
	}
	request := requestWithBooking("external-request", bookingID)
	request.Headers[RoutingModeHeader] = "disaggregated"
	registerBookingLifecycle(bookingID, scorer.freeBooking)

	for range 10 {
		scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	}
	lifecycle := findBookingLifecycle(bookingID)
	if lifecycle == nil {
		t.Fatal("expected booking lifecycle after first response chunk")
	}
	select {
	case <-lifecycle.markerComplete():
	case <-time.After(2 * time.Second):
		t.Fatal("bounded marker retries did not finish")
	}
	if markCalls != prefillMarkMaxAttempts {
		t.Fatalf("mark calls = %d, want %d", markCalls, prefillMarkMaxAttempts)
	}

	scorer.ResponseBody(context.Background(), request, &rc.Response{EndOfStream: true}, nil)
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("terminal booking cleanup did not finish")
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestResponseBodyEOSDoesNotWaitForInFlightPrefillMark(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	markStarted := make(chan struct{})
	allowMarkReturn := make(chan struct{})
	freeCalls := 0
	scorer := &DynDecodeScorer{
		markPrefillComplete: func(string) error {
			close(markStarted)
			<-allowMarkReturn
			return nil
		},
		freeBooking: func(string) error {
			freeCalls++
			return nil
		},
	}
	request := requestWithBooking("external-request", bookingID)
	request.Headers[RoutingModeHeader] = "disaggregated"
	registerBookingLifecycle(bookingID, scorer.freeBooking)
	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	<-markStarted
	lifecycle := findBookingLifecycle(bookingID)
	if lifecycle == nil {
		t.Fatal("expected booking lifecycle after first response chunk")
	}

	responseDone := make(chan struct{})
	go func() {
		scorer.ResponseBody(context.Background(), request, &rc.Response{EndOfStream: true}, nil)
		close(responseDone)
	}()
	select {
	case <-responseDone:
	case <-time.After(time.Second):
		close(allowMarkReturn)
		t.Fatal("EOS callback blocked waiting on prefill mark")
	}
	close(allowMarkReturn)
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("terminal booking cleanup did not finish")
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestBookingLifecycleCleansUpOnceForEOSAndCancellation(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	freeCalls := 0
	lifecycle := registerBookingLifecycle(bookingID, func(string) error {
		freeCalls++
		return nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	lifecycle.armCancellation(ctx)
	lifecycle.cleanup(ctx, "response end of stream")
	cancel()
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("booking cleanup did not finish")
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestPrefillScoreBoundsConcurrentReservations(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reserveStarted := make(chan struct{})
	allowReserveReturn := make(chan struct{})
	reserveCalls := 0
	scorer := &DynPrefillScorer{
		reservePrefill: func(string, string, string) (*dynscorer.RoutingResult, error) {
			reserveCalls++
			close(reserveStarted)
			<-allowReserveReturn
			return &dynscorer.RoutingResult{WorkerID: 7}, nil
		},
		freeBooking:                 func(string) error { return nil },
		reservationAdmissionTimeout: 20 * time.Millisecond,
		reservationSlots:            make(chan struct{}, 1),
	}
	newRequest := func() (*schedtypes.CycleState, *schedtypes.InferenceRequest) {
		cycleState := schedtypes.NewCycleState()
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
		return cycleState, &schedtypes.InferenceRequest{
			TargetModel: "model",
			Headers:     map[string]string{},
			Body: &fwkrh.InferenceRequestBody{
				Payload: fwkrh.PayloadMap{"model": "model", "prompt": "hello"},
			},
		}
	}
	firstState, firstRequest := newRequest()
	firstScores := make(chan map[schedtypes.Endpoint]float64, 1)
	go func() {
		firstScores <- scorer.Score(ctx, firstState, firstRequest, nil)
	}()
	<-reserveStarted

	secondState, secondRequest := newRequest()
	if scores := scorer.Score(context.Background(), secondState, secondRequest, nil); len(scores) != 0 {
		t.Fatalf("scores = %v, want aggregate fallback with no prefill endpoints", scores)
	}
	if reserveCalls != 1 {
		t.Fatalf("reserve calls = %d, want 1 bounded in-flight admission", reserveCalls)
	}

	close(allowReserveReturn)
	<-firstScores
	cancel()
	lifecycle := findBookingLifecycle(bookingIDFromRequest(firstRequest))
	if lifecycle == nil {
		t.Fatal("expected lifecycle for successful first reservation")
	}
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("first reservation cleanup did not finish")
	}
}

func newDecodeRequest() *schedtypes.InferenceRequest {
	return &schedtypes.InferenceRequest{
		TargetModel: "model",
		Headers:     map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{"model": "model", "prompt": "hello"},
		},
	}
}

func TestDecodeScoreCancellationWaitsForRegistration(t *testing.T) {
	cycleState := schedtypes.NewCycleState()
	booking := ensureBookingState(cycleState)
	request := newDecodeRequest()
	addStarted := make(chan struct{})
	allowAddReturn := make(chan struct{})
	freeCalls := make(chan string, 1)
	scorer := &DynDecodeScorer{
		routeDecode: func(string, string, bool) (*dynscorer.RoutingResult, error) {
			return &dynscorer.RoutingResult{WorkerID: 7, DpRank: 1, TokenData: []int64{1}}, nil
		},
		addRequest: func(string, []int64, uint64, uint32, string) error {
			close(addStarted)
			<-allowAddReturn
			return nil
		},
		freeBooking: func(bookingID string) error {
			freeCalls <- bookingID
			return nil
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	scoresDone := make(chan map[schedtypes.Endpoint]float64, 1)
	go func() {
		scoresDone <- scorer.Score(ctx, cycleState, request, nil)
	}()
	<-addStarted
	lifecycle := findBookingLifecycle(booking.ID)
	if lifecycle == nil {
		t.Fatal("expected lifecycle while decode registration is in flight")
	}
	cancel()
	select {
	case got := <-freeCalls:
		t.Fatalf("booking %q was freed before decode registration returned", got)
	case <-time.After(20 * time.Millisecond):
	}
	close(allowAddReturn)
	<-scoresDone
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("booking cleanup did not finish after registration returned")
	}
	select {
	case got := <-freeCalls:
		if got != booking.ID {
			t.Fatalf("freed booking ID = %q, want %q", got, booking.ID)
		}
	case <-time.After(time.Second):
		t.Fatal("booking was not cleaned up after cancellation")
	}
}

func TestDecodeScoreRegistrationFailureRedirectsAggregate(t *testing.T) {
	cycleState := schedtypes.NewCycleState()
	booking := ensureBookingState(cycleState)
	request := newDecodeRequest()
	freeStarted := make(chan struct{})
	allowFreeReturn := make(chan struct{})
	scorer := &DynDecodeScorer{
		routeDecode: func(string, string, bool) (*dynscorer.RoutingResult, error) {
			return &dynscorer.RoutingResult{WorkerID: 7, DpRank: 1, TokenData: []int64{1}}, nil
		},
		addRequest: func(string, []int64, uint64, uint32, string) error {
			return errors.New("decode booking failed")
		},
		freeBooking: func(string) error {
			close(freeStarted)
			<-allowFreeReturn
			return nil
		},
	}

	scorer.Score(context.Background(), cycleState, request, nil)
	lifecycle := findBookingLifecycle(booking.ID)
	if lifecycle == nil {
		t.Fatal("expected lifecycle after decode booking failure")
	}
	select {
	case <-freeStarted:
	case <-time.After(time.Second):
		t.Fatal("failed decode booking was not cleaned up")
	}
	if got := request.Headers[RoutingModeHeader]; got != "aggregated" {
		t.Fatalf("routing mode = %q, want aggregated fallback", got)
	}
	for _, header := range []string{WorkerIDHeader, DpRankHeader, PrefillWorkerIDHeader, PrefillDpRankHeader} {
		if _, ok := request.Headers[header]; ok {
			t.Fatalf("redirect retained %s header", header)
		}
	}
	close(allowFreeReturn)
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("failed decode booking cleanup did not finish")
	}
}

func TestBookingLifecycleRetriesCleanup(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	calls := 0
	lifecycle := registerBookingLifecycle(bookingID, func(string) error {
		calls++
		if calls == 1 {
			return errors.New("transient cleanup failure")
		}
		return nil
	})
	if !lifecycle.cleanup(context.Background(), "test retry") {
		t.Fatal("initial cleanup did not start")
	}
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("cleanup retry did not finish")
	}
	if calls != 2 {
		t.Fatalf("cleanup calls = %d, want 2", calls)
	}
	if findBookingLifecycle(bookingID) != nil {
		t.Fatal("successful cleanup retained a lifecycle")
	}
}

func TestBookingLifecycleRetainsTombstoneAfterCleanupExhaustion(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	defer bookingLifecycles.Delete(bookingID)
	calls := 0
	lifecycle := registerBookingLifecycle(bookingID, func(string) error {
		calls++
		return errors.New("persistent cleanup failure")
	})
	if !lifecycle.cleanup(context.Background(), "test exhaustion") {
		t.Fatal("initial cleanup did not start")
	}
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("cleanup retries did not finish")
	}
	if calls != cleanupMaxAttempts {
		t.Fatalf("cleanup calls = %d, want %d", calls, cleanupMaxAttempts)
	}
	if got := findBookingLifecycle(bookingID); got != lifecycle {
		t.Fatal("exhausted cleanup did not retain its lifecycle tombstone")
	}
	if lifecycle.cleanup(context.Background(), "duplicate cleanup") {
		t.Fatal("exhausted cleanup started a second owner")
	}
}

func TestResponseBodySkipsMarkForAggregatedRequest(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	markCalled := make(chan struct{}, 1)
	freeCalls := 0
	scorer := &DynDecodeScorer{
		markPrefillComplete: func(string) error {
			markCalled <- struct{}{}
			return nil
		},
		freeBooking: func(string) error {
			freeCalls++
			return nil
		},
	}
	request := requestWithBooking("external-request", bookingID)
	request.Headers[RoutingModeHeader] = "aggregated"
	lifecycle := registerBookingLifecycle(bookingID, scorer.freeBooking)

	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	select {
	case <-markCalled:
		t.Fatal("aggregated response marked prefill complete")
	case <-time.After(100 * time.Millisecond):
	}
	scorer.ResponseBody(context.Background(), request, &rc.Response{EndOfStream: true}, nil)
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("aggregated response cleanup did not finish")
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestResponseBodyIgnoresUntrackedBookingHeader(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	markCalled := make(chan struct{}, 1)
	freeCalled := make(chan struct{}, 1)
	scorer := &DynDecodeScorer{
		markPrefillComplete: func(string) error {
			markCalled <- struct{}{}
			return nil
		},
		freeBooking: func(string) error {
			freeCalled <- struct{}{}
			return nil
		},
	}
	request := requestWithBooking("client-request", bookingID)
	request.Headers[RoutingModeHeader] = "disaggregated"

	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	scorer.ResponseBody(context.Background(), request, &rc.Response{EndOfStream: true}, nil)
	select {
	case <-markCalled:
		t.Fatal("untracked booking header marked prefill complete")
	case <-freeCalled:
		t.Fatal("untracked booking header freed router bookkeeping")
	case <-time.After(100 * time.Millisecond):
	}
}
