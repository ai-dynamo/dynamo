/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"github.com/stretchr/testify/require"
)

func TestWorkerDefaultsCanaryHealthCheckVersionGate(t *testing.T) {
	tests := []struct {
		name                     string
		runtimeVersion           *runtimeversion.Version
		wantEnabled              string
		wantLivenessFailureCount int32
	}{
		{
			name:                     "unknown legacy runtime",
			wantEnabled:              "false",
			wantLivenessFailureCount: 1,
		},
		{
			name:                     "older runtime",
			runtimeVersion:           &runtimeversion.Version{Major: 1, Minor: 4, Patch: 9},
			wantEnabled:              "false",
			wantLivenessFailureCount: 1,
		},
		{
			name:                     "minimum supported runtime",
			runtimeVersion:           &runtimeversion.Version{Major: 1, Minor: 5, Patch: 0},
			wantEnabled:              "true",
			wantLivenessFailureCount: 3,
		},
		{
			name:                     "newer runtime",
			runtimeVersion:           &runtimeversion.Version{Major: 2, Minor: 0, Patch: 0},
			wantEnabled:              "true",
			wantLivenessFailureCount: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container, err := NewWorkerDefaults().GetBaseContainer(ComponentContext{
				RuntimeVersion: tt.runtimeVersion,
			})
			require.NoError(t, err)
			require.NotNil(t, container.LivenessProbe)
			require.Equal(t, tt.wantLivenessFailureCount, container.LivenessProbe.FailureThreshold)

			for _, env := range container.Env {
				if env.Name == "DYN_HEALTH_CHECK_ENABLED" {
					require.Equal(t, tt.wantEnabled, env.Value)
					return
				}
			}
			t.Fatal("DYN_HEALTH_CHECK_ENABLED was not rendered")
		})
	}
}
