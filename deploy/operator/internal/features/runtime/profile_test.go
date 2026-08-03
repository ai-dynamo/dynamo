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
	tests := []struct {
		name    string
		version *runtimeversion.Version
		want    RuntimeProfile
	}{
		{
			name: "unknown runtime keeps the legacy profile",
		},
		{
			name:    "runtime 1.4 keeps the legacy profile",
			version: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
		},
		{
			name:    "runtime 1.5 enables the first versioned profile",
			version: &runtimeversion.Version{Major: 1, Minor: 5, Patch: 0},
			want:    RuntimeProfile{CanaryHealthChecks: true},
		},
		{
			name:    "newer runtime with the same gates has the same profile",
			version: &runtimeversion.Version{Major: 2, Minor: 0, Patch: 0},
			want:    RuntimeProfile{CanaryHealthChecks: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ProfileForVersion(tt.version)
			if got != tt.want {
				t.Fatalf("ProfileForVersion(%v) = %+v, want %+v", tt.version, got, tt.want)
			}
			if got.IsEmpty() != (tt.want == RuntimeProfile{}) {
				t.Fatalf("IsEmpty() = %t for profile %+v", got.IsEmpty(), got)
			}
		})
	}
}
