/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"errors"
	"testing"
)

func TestQualifiedBounds(t *testing.T) {
	index := QualificationIndex{
		"a": {MinWatts: 200, DefaultWatts: 400, MaxWatts: 700},
		"b": {MinWatts: 250, DefaultWatts: 500, MaxWatts: 800},
	}

	t.Log("Clamp a below-minimum request for the one exact eligible product")
	bounds, err := index.QualifiedBounds(150, []string{"a"})
	if err != nil {
		t.Fatalf("QualifiedBounds() error = %v", err)
	}
	if bounds.InGateWatts != 200 {
		t.Errorf("B_c = %d, want 200", bounds.InGateWatts)
	}
	if bounds.UnenforcedWatts != 700 {
		t.Errorf("U_c = %d, want 700", bounds.UnenforcedWatts)
	}
	if bounds.QualifiedMinWatts != 200 || bounds.QualifiedMaxWatts != 700 {
		t.Errorf("qualified range = [%d,%d], want [200,700]", bounds.QualifiedMinWatts, bounds.QualifiedMaxWatts)
	}

	t.Log("Use the immutable request, not a safe-default cap, when the request is in range")
	bounds, err = index.QualifiedBounds(350, []string{"a"})
	if err != nil {
		t.Fatalf("QualifiedBounds() error = %v", err)
	}
	if bounds.InGateWatts != 350 {
		t.Errorf("B_c = %d, want requested cap 350", bounds.InGateWatts)
	}
	if bounds.UnenforcedWatts != 700 {
		t.Errorf("U_c = %d, want 700", bounds.UnenforcedWatts)
	}

	t.Log("Fail closed when the exact product is unknown")
	if _, err := index.QualifiedBounds(350, []string{"unknown"}); !errors.Is(err, ErrUnqualifiedHardware) {
		t.Fatalf("QualifiedBounds() error = %v, want ErrUnqualifiedHardware", err)
	}

	t.Log("Fail closed instead of accepting multi-product eligibility")
	if _, err := index.QualifiedBounds(350, []string{"a", "b"}); !errors.Is(err, ErrUnqualifiedHardware) {
		t.Fatalf("multi-product QualifiedBounds() error = %v, want ErrUnqualifiedHardware", err)
	}

	t.Log("Reject every assigned live-range drift, not only a higher maximum")
	for _, live := range [][2]int64{{199, 700}, {200, 699}, {200, 701}} {
		if err := bounds.ValidateLiveRange(live[0], live[1]); !errors.Is(err, ErrUnqualifiedHardware) {
			t.Fatalf("ValidateLiveRange(%v) error = %v, want ErrUnqualifiedHardware", live, err)
		}
	}
	if err := bounds.ValidateLiveRange(200, 700); err != nil {
		t.Fatalf("ValidateLiveRange() error = %v at qualified range", err)
	}
}
