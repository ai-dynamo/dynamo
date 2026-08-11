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

	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

func TestDynPrefillScorerReservesAndReleasesOnce(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scorer := newDynPrefillScorer(ctx)
	scorer.newPrefillReservationID = func() string { return "prefill-reservation-1" }
	reserveCalls := 0
	scorer.routeAndReservePrefill = func(reservationID, _, _ string) (*dynscorer.RoutingResult, error) {
		reserveCalls++
		if reservationID != "prefill-reservation-1" {
			t.Fatalf("unexpected reservation ID: %q", reservationID)
		}
		return &dynscorer.RoutingResult{WorkerID: 42, DpRank: 3}, nil
	}
	freeCalls := 0
	scorer.freePrefillReservation = func(reservationID string, workerID uint64, dpRank uint32) error {
		freeCalls++
		if reservationID != "prefill-reservation-1" || workerID != 42 || dpRank != 3 {
			t.Fatalf("unexpected cleanup: reservation=%q worker=%d rank=%d", reservationID, workerID, dpRank)
		}
		return nil
	}

	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	request := testPrefillRequest("request-1")
	scorer.Score(ctx, cycleState, request, nil)

	if reserveCalls != 1 {
		t.Fatalf("reserve calls = %d, want 1", reserveCalls)
	}
	if got := request.Headers[PrefillWorkerIDHeader]; got != "42" {
		t.Fatalf("prefill worker header = %q, want 42", got)
	}
	if got := request.Headers[PrefillDpRankHeader]; got != "3" {
		t.Fatalf("prefill DP rank header = %q, want 3", got)
	}

	state, err := plugins.ReadPluginStateKey[*PrefillReservationState](
		scorer.pluginState, request.RequestId, plugins.StateKey(prefillReservationStateKey),
	)
	if err != nil || state.ReservationID != "prefill-reservation-1" {
		t.Fatalf("reservation state = %#v, err = %v", state, err)
	}

	scorer.ResponseBody(ctx, request, &rc.Response{StartOfStream: true}, nil)
	scorer.ResponseBody(ctx, request, &rc.Response{EndOfStream: true}, nil)
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestDynPrefillScorerUsesAdvisoryRoutingWithoutRequestID(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scorer := newDynPrefillScorer(ctx)
	reserveCalls := 0
	scorer.routeAndReservePrefill = func(_, _, _ string) (*dynscorer.RoutingResult, error) {
		reserveCalls++
		return nil, nil
	}
	advisoryCalls := 0
	scorer.routePrefillAdvisory = func(_, _ string) (*dynscorer.RoutingResult, error) {
		advisoryCalls++
		return &dynscorer.RoutingResult{WorkerID: 7, DpRank: dynscorer.UnsetDpRank}, nil
	}

	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	request := testPrefillRequest("")
	scorer.Score(ctx, cycleState, request, nil)

	if reserveCalls != 0 || advisoryCalls != 1 {
		t.Fatalf("reserve calls = %d, advisory calls = %d; want 0 and 1", reserveCalls, advisoryCalls)
	}
	if got := request.Headers[PrefillWorkerIDHeader]; got != "7" {
		t.Fatalf("prefill worker header = %q, want 7", got)
	}
}

func TestDynPrefillScorerRetriesFailedCleanup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scorer := newDynPrefillScorer(ctx)
	scorer.pluginState.Write("request-1", plugins.StateKey(prefillReservationStateKey), &PrefillReservationState{
		ReservationID: "prefill-reservation-1",
		WorkerID:      42,
		DpRank:        3,
	})

	freeCalls := 0
	scorer.freePrefillReservation = func(_ string, _ uint64, _ uint32) error {
		freeCalls++
		if freeCalls == 1 {
			return errors.New("temporary cleanup failure")
		}
		return nil
	}

	request := testPrefillRequest("request-1")
	scorer.ResponseBody(ctx, request, &rc.Response{StartOfStream: true}, nil)
	scorer.ResponseBody(ctx, request, &rc.Response{}, nil)
	if freeCalls != 1 {
		t.Fatalf("free calls before terminal retry = %d, want 1", freeCalls)
	}
	if _, err := plugins.ReadPluginStateKey[*PrefillReservationState](
		scorer.pluginState, request.RequestId, plugins.StateKey(prefillReservationStateKey),
	); err != nil {
		t.Fatalf("reservation state removed after failed cleanup: %v", err)
	}

	scorer.ResponseBody(ctx, request, &rc.Response{EndOfStream: true}, nil)
	if freeCalls != 2 {
		t.Fatalf("free calls = %d, want 2", freeCalls)
	}
	if _, err := plugins.ReadPluginStateKey[*PrefillReservationState](
		scorer.pluginState, request.RequestId, plugins.StateKey(prefillReservationStateKey),
	); err == nil {
		t.Fatal("reservation state retained after successful cleanup retry")
	}
}

func TestDynPrefillScorerReleasesPriorReservationBeforeRescoring(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scorer := newDynPrefillScorer(ctx)
	reservationIDs := []string{"prefill-reservation-1", "prefill-reservation-2"}
	scorer.newPrefillReservationID = func() string {
		id := reservationIDs[0]
		reservationIDs = reservationIDs[1:]
		return id
	}
	events := make([]string, 0, 3)
	scorer.routeAndReservePrefill = func(reservationID, _, _ string) (*dynscorer.RoutingResult, error) {
		events = append(events, "reserve:"+reservationID)
		return &dynscorer.RoutingResult{WorkerID: 42, DpRank: 3}, nil
	}
	scorer.freePrefillReservation = func(reservationID string, _ uint64, _ uint32) error {
		events = append(events, "free:"+reservationID)
		return nil
	}

	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	request := testPrefillRequest("request-1")
	scorer.Score(ctx, cycleState, request, nil)
	scorer.Score(ctx, cycleState, request, nil)

	want := []string{
		"reserve:prefill-reservation-1",
		"free:prefill-reservation-1",
		"reserve:prefill-reservation-2",
	}
	if len(events) != len(want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
	for i := range want {
		if events[i] != want[i] {
			t.Fatalf("events = %v, want %v", events, want)
		}
	}

	state, err := plugins.ReadPluginStateKey[*PrefillReservationState](
		scorer.pluginState, request.RequestId, plugins.StateKey(prefillReservationStateKey),
	)
	if err != nil || state.ReservationID != "prefill-reservation-2" {
		t.Fatalf("reservation state = %#v, err = %v", state, err)
	}
}

func TestDynDecodeScorerRollsBackPrefillWhenDecodeRoutingFails(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scorer := NewDynDecodeScorer(ctx)
	scorer.routeDecode = func(_, _ string, _ bool) (*dynscorer.RoutingResult, error) {
		return nil, errors.New("decode routing failed")
	}
	freeCalls := 0
	scorer.freePrefillReservation = func(reservationID string, workerID uint64, dpRank uint32) error {
		freeCalls++
		if reservationID != "prefill-reservation-1" || workerID != 42 || dpRank != 3 {
			t.Fatalf("unexpected cleanup: reservation=%q worker=%d rank=%d", reservationID, workerID, dpRank)
		}
		return nil
	}

	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	cycleState.Write(prefillReservationCycleStateKey, &PrefillReservationState{
		ReservationID: "prefill-reservation-1",
		WorkerID:      42,
		DpRank:        3,
	})

	scorer.Score(ctx, cycleState, testPrefillRequest("request-1"), nil)

	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
	if readPrefillEnabled(cycleState) {
		t.Fatal("prefill remained enabled after decode routing rollback")
	}
	if reservation := readCyclePrefillReservation(cycleState); reservation != nil {
		t.Fatalf("cycle reservation not cleared: %#v", reservation)
	}
}

func TestDynDecodeScorerRollsBackPrefillWhenCancelledAfterRouting(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scorer := NewDynDecodeScorer(ctx)
	scorer.routeDecode = func(_, _ string, _ bool) (*dynscorer.RoutingResult, error) {
		cancel()
		return &dynscorer.RoutingResult{WorkerID: 7}, nil
	}
	freeCalls := 0
	scorer.freePrefillReservation = func(reservationID string, workerID uint64, dpRank uint32) error {
		freeCalls++
		if reservationID != "prefill-reservation-1" || workerID != 42 || dpRank != 3 {
			t.Fatalf("unexpected cleanup: reservation=%q worker=%d rank=%d", reservationID, workerID, dpRank)
		}
		return nil
	}

	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	cycleState.Write(prefillReservationCycleStateKey, &PrefillReservationState{
		ReservationID: "prefill-reservation-1",
		WorkerID:      42,
		DpRank:        3,
	})

	scorer.Score(ctx, cycleState, testPrefillRequest("request-1"), nil)

	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
	if readPrefillEnabled(cycleState) {
		t.Fatal("prefill remained enabled after cancellation rollback")
	}
}

func TestDynDecodeScorerRollsBackPrefillWhenDecodeBookingFails(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scorer := NewDynDecodeScorer(ctx)
	scorer.addDecodeRequest = func(string, []int64, uint64, uint32, string) error {
		return errors.New("decode booking failed")
	}
	freeCalls := 0
	scorer.freePrefillReservation = func(reservationID string, workerID uint64, dpRank uint32) error {
		freeCalls++
		if reservationID != "prefill-reservation-1" || workerID != 42 || dpRank != 3 {
			t.Fatalf("unexpected cleanup: reservation=%q worker=%d rank=%d", reservationID, workerID, dpRank)
		}
		return nil
	}
	scorer.pluginState.Write("request-1", plugins.StateKey(decodeStateKey), &DecodeRoutingState{
		WorkerID: "7",
		PrefillReservation: &PrefillReservationState{
			ReservationID: "prefill-reservation-1",
			WorkerID:      42,
			DpRank:        3,
		},
	})

	scorer.PreRequest(ctx, testPrefillRequest("request-1"), nil)

	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestDisaggProfileHandlerRollsBackPrefillOnTerminalFailure(t *testing.T) {
	ctx := context.Background()
	handler := NewDisaggProfileHandler()
	freeCalls := 0
	handler.freePrefillReservation = func(reservationID string, workerID uint64, dpRank uint32) error {
		freeCalls++
		if reservationID != "prefill-reservation-1" || workerID != 42 || dpRank != 3 {
			t.Fatalf("unexpected cleanup: reservation=%q worker=%d rank=%d", reservationID, workerID, dpRank)
		}
		return nil
	}
	cycleState := schedtypes.NewCycleState()
	cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: true})
	cycleState.Write(prefillReservationCycleStateKey, &PrefillReservationState{
		ReservationID: "prefill-reservation-1",
		WorkerID:      42,
		DpRank:        3,
	})

	if _, err := handler.ProcessResults(ctx, cycleState, nil, map[string]*schedtypes.ProfileRunResult{}); err == nil {
		t.Fatal("expected empty profile results to fail")
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
	if readPrefillEnabled(cycleState) {
		t.Fatal("prefill remained enabled after terminal profile failure")
	}
}

func testPrefillRequest(requestID string) *schedtypes.InferenceRequest {
	return &schedtypes.InferenceRequest{
		RequestId: requestID,
		Body: &fwkrh.InferenceRequestBody{
			Payload: fwkrh.PayloadMap{
				"model":  "test-model",
				"prompt": "hello",
			},
		},
	}
}
