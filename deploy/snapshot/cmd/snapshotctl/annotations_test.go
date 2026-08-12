//go:build linux

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestReconcileTargetContainers covers the normalized 1..1 target-container resolution used by the
// capture flow: the flag and manifest annotation must agree when both are present, and the result is
// returned as a []string that createPodSnapshot sets on PodReference.Containers.
func TestReconcileTargetContainers(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		flagValue   string
		want        []string
		wantErr     string
	}{
		{
			name:      "flag only",
			flagValue: "main",
			want:      []string{"main"},
		},
		{
			name:        "manifest only",
			annotations: map[string]string{"nvidia.com/snapshot-target-containers": "engine"},
			flagValue:   "",
			want:        []string{"engine"},
		},
		{
			name:        "flag and manifest agree",
			annotations: map[string]string{"nvidia.com/snapshot-target-containers": "main"},
			flagValue:   "main",
			want:        []string{"main"},
		},
		{
			name:        "flag and manifest mismatch",
			annotations: map[string]string{"nvidia.com/snapshot-target-containers": "b"},
			flagValue:   "a",
			wantErr:     "does not match manifest",
		},
		{
			name:      "neither flag nor manifest",
			flagValue: "",
			wantErr:   "target containers are required",
		},
		{
			name:      "flag exceeds max",
			flagValue: "a,b",
			wantErr:   "at most 1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := reconcileTargetContainers(tt.annotations, tt.flagValue, 1, 1)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
