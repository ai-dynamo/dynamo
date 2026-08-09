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
	"testing"

	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestNewPrefillReservationIDIsUniqueAndOpaque(t *testing.T) {
	first, err := newPrefillReservationID()
	if err != nil {
		t.Fatalf("newPrefillReservationID() returned error: %v", err)
	}
	second, err := newPrefillReservationID()
	if err != nil {
		t.Fatalf("newPrefillReservationID() returned error: %v", err)
	}

	if len(first) != 64 || len(second) != 64 {
		t.Fatalf("expected authenticated 256-bit hex IDs, got lengths %d and %d", len(first), len(second))
	}
	if first == second {
		t.Fatalf("expected unique reservation IDs, both were %q", first)
	}
	if !isPrefillReservationID(first) || !isPrefillReservationID(second) {
		t.Fatal("expected generated reservation IDs to authenticate")
	}
	if isPrefillReservationID("client-controlled-booking-id") {
		t.Fatal("expected a client-controlled booking ID to be rejected")
	}
	tampered := first[:len(first)-1] + "0"
	if tampered == first {
		tampered = first[:len(first)-1] + "1"
	}
	if isPrefillReservationID(tampered) {
		t.Fatal("expected a modified reservation ID to be rejected")
	}
}

func TestRouterBookingIDUsesInternalReservationWhenPresent(t *testing.T) {
	req := &schedtypes.InferenceRequest{
		RequestId: "client-request-id",
		Headers: map[string]string{
			PrefillReservationIDHeader: mustPrefillReservationID(t),
			RoutingModeHeader:          "disaggregated",
		},
	}
	reservationID := req.Headers[PrefillReservationIDHeader]

	if got := routerBookingID(req); got != reservationID {
		t.Fatalf("expected internal reservation ID, got %q", got)
	}

	// A failed cleanup can fall back to aggregated dispatch while keeping the
	// internal ID for a later lifecycle callback to retry both routers.
	req.Headers[RoutingModeHeader] = "aggregated"
	if got := routerBookingID(req); got != reservationID {
		t.Fatalf("expected cleanup fallback to retain internal reservation ID, got %q", got)
	}

	delete(req.Headers, PrefillReservationIDHeader)
	if got := routerBookingID(req); got != "client-request-id" {
		t.Fatalf("expected request ID without an internal reservation, got %q", got)
	}
}

func TestDuplicateClientRequestIDsDoNotShareDisaggregatedBookings(t *testing.T) {
	request := func(reservationID string) *schedtypes.InferenceRequest {
		return &schedtypes.InferenceRequest{
			RequestId: "reused-client-id",
			Headers: map[string]string{
				PrefillReservationIDHeader: reservationID,
				RoutingModeHeader:          "disaggregated",
			},
		}
	}

	first := routerBookingID(request(mustPrefillReservationID(t)))
	second := routerBookingID(request(mustPrefillReservationID(t)))
	if first == second {
		t.Fatalf("expected independent bookings, both used %q", first)
	}
}

func mustPrefillReservationID(t *testing.T) string {
	t.Helper()
	reservationID, err := newPrefillReservationID()
	if err != nil {
		t.Fatalf("newPrefillReservationID() returned error: %v", err)
	}
	return reservationID
}
