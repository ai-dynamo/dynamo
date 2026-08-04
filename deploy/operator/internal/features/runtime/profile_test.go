/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtime

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func TestProfileForVersion(t *testing.T) {
	t.Log("define profiles below, at, and above the canary health-check threshold")
	tests := []struct {
		name    string
		version *runtimeversion.Version
		want    RuntimeProfile
	}{
		{name: "unknown runtime"},
		{
			name:    "runtime below threshold",
			version: &runtimeversion.Version{Major: 1, Minor: 3, Patch: 9},
		},
		{
			name:    "runtime at threshold",
			version: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			want:    RuntimeProfile{CanaryHealthChecks: true},
		},
		{
			name:    "newer runtime with the same gates",
			version: &runtimeversion.Version{Major: 2, Minor: 0, Patch: 0},
			want:    RuntimeProfile{CanaryHealthChecks: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("evaluate all runtime gates")
			got := ProfileForVersion(tt.version)

			t.Log("verify the effective profile and its empty state")
			if got != tt.want {
				t.Fatalf("ProfileForVersion(%v) = %+v, want %+v", tt.version, got, tt.want)
			}
			if got.IsEmpty() != (tt.want == RuntimeProfile{}) {
				t.Fatalf("IsEmpty() = %t for profile %+v", got.IsEmpty(), got)
			}
		})
	}
}
