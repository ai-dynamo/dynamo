/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func TestGateEnabled(t *testing.T) {
	t.Log("define a feature introduced by Dynamo runtime 1.4.0")
	gate := Gate{
		Name:              "TestFeature",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	}

	t.Log("define versions below, at, and above the feature threshold")
	tests := []struct {
		name    string
		version *runtimeversion.Version
		want    bool
	}{
		{
			name: "unknown runtime",
		},
		{
			name:    "older runtime",
			version: &runtimeversion.Version{Major: 1, Minor: 3, Patch: 9},
		},
		{
			name:    "minimum supported runtime",
			version: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			want:    true,
		},
		{
			name:    "newer runtime",
			version: &runtimeversion.Version{Major: 2, Minor: 0, Patch: 0},
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("evaluate the feature for the resolved runtime version")
			got := gate.Enabled(tt.version)

			t.Log("compare the gate decision")
			if got != tt.want {
				t.Fatalf("Enabled(%v) = %t, want %t", tt.version, got, tt.want)
			}
		})
	}
}

func TestCanaryHealthChecksThreshold(t *testing.T) {
	t.Log("inspect the central canary health-check feature gate")
	got := CanaryHealthChecks.MinRuntimeVersion.String()

	t.Log("verify canary health checks are introduced by runtime 1.5.0")
	if got != "1.5.0" {
		t.Fatalf("MinRuntimeVersion = %s, want 1.5.0", got)
	}
}
