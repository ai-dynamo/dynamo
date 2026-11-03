// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package provideroverride

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGroveAvailabilityDecodingIsCaseSensitive(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		raw                   string
		minimum, budget       int32
		hasMinimum, hasBudget bool
	}{
		{name: "standalone", raw: `{"spec":{"minAvailable":2},"rollingUpdate":{"maxUnavailable":3}}`, minimum: 2, budget: 3, hasMinimum: true, hasBudget: true},
		{name: "scaling group", raw: `{"minAvailable":2,"rollingUpdate":{"maxUnavailable":3}}`, minimum: 2, budget: 3, hasMinimum: true, hasBudget: true},
		{name: "rolling subtree capitalization", raw: `{"RollingUpdate":{"maxUnavailable":3}}`},
		{name: "budget capitalization", raw: `{"rollingUpdate":{"MaxUnavailable":3}}`},
		{name: "minimum capitalization", raw: `{"MinAvailable":2}`},
		{name: "spec capitalization", raw: `{"Spec":{"minAvailable":2}}`},
		{name: "nested minimum capitalization", raw: `{"spec":{"MinAvailable":2}}`},
		{name: "null budget", raw: `{"rollingUpdate":{"maxUnavailable":null}}`},
		{name: "fractional budget", raw: `{"rollingUpdate":{"maxUnavailable":1.5}}`},
		{name: "overflow budget", raw: `{"rollingUpdate":{"maxUnavailable":2147483648}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Decode the persisted fragment with Kubernetes field-name semantics")
			minimum, hasMinimum := GroveMinAvailable([]byte(tc.raw))
			budget, hasBudget := GroveMaxUnavailable([]byte(tc.raw))
			require.Equal(t, tc.minimum, minimum)
			require.Equal(t, tc.hasMinimum, hasMinimum)
			require.Equal(t, tc.budget, budget)
			require.Equal(t, tc.hasBudget, hasBudget)
		})
	}
}
