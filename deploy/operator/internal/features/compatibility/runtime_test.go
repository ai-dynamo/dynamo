/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package compatibility

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func TestRuntimeGateEnabled(t *testing.T) {
	gate := RuntimeGate{
		Name:              "TestFeature",
		MinRuntimeVersion: runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	}

	tests := []struct {
		name    string
		version *runtimeversion.Version
		want    bool
	}{
		{
			name:    "unknown runtime version is disabled",
			version: nil,
			want:    false,
		},
		{
			name:    "runtime version below threshold is disabled",
			version: &runtimeversion.Version{Major: 1, Minor: 3, Patch: 9},
			want:    false,
		},
		{
			name:    "runtime version at threshold is enabled",
			version: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			want:    true,
		},
		{
			name:    "runtime version above threshold is enabled",
			version: &runtimeversion.Version{Major: 2, Minor: 0, Patch: 0},
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := gate.Enabled(tt.version); got != tt.want {
				t.Fatalf("Enabled(%v) = %t, want %t", tt.version, got, tt.want)
			}
		})
	}
}
