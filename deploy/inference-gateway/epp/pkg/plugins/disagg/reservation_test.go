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

	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
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

	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	scorer.ResponseBody(context.Background(), request, &rc.Response{}, nil)
	scorer.ResponseBody(context.Background(), request, &rc.Response{EndOfStream: true}, nil)

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

	scorer.ResponseBody(
		context.Background(),
		requestWithBooking("external-request", bookingID),
		&rc.Response{EndOfStream: true},
		nil,
	)

	if markCalls != 0 {
		t.Fatalf("mark calls = %d, want 0", markCalls)
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestPreRequestCancellationCleansBooking(t *testing.T) {
	bookingID := ensureBookingState(schedtypes.NewCycleState()).ID
	freeCalls := 0
	scorer := &DynDecodeScorer{
		freeBooking: func(got string) error {
			if got != bookingID {
				t.Fatalf("free booking ID = %q, want %q", got, bookingID)
			}
			freeCalls++
			return nil
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	scorer.PreRequest(ctx, requestWithBooking("external-request", bookingID), nil)

	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}
