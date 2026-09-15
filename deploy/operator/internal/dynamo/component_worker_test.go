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
		name           string
		runtimeVersion *runtimeversion.Version
		wantEnabled    string
	}{
		{
			name:        "unknown legacy runtime",
			wantEnabled: "false",
		},
		{
			name:           "older runtime",
			runtimeVersion: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 9},
			wantEnabled:    "false",
		},
		{
			name:           "minimum supported runtime",
			runtimeVersion: &runtimeversion.Version{Major: 1, Minor: 5, Patch: 0},
			wantEnabled:    "true",
		},
		{
			name:           "newer runtime",
			runtimeVersion: &runtimeversion.Version{Major: 2, Minor: 0, Patch: 0},
			wantEnabled:    "true",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("render the worker defaults for the resolved runtime version")
			container, err := NewWorkerDefaults().GetBaseContainer(ComponentContext{
				RuntimeVersion: tt.runtimeVersion,
			})
			require.NoError(t, err)

			t.Log("verify the canary health-check environment default")
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

func TestWorkerDefaultsLivenessFailureThresholdVersionGate(t *testing.T) {
	tests := []struct {
		name             string
		runtimeVersion   *runtimeversion.Version
		wantFailureCount int32
	}{
		{
			name:             "unknown legacy runtime",
			wantFailureCount: 1,
		},
		{
			name:             "older runtime",
			runtimeVersion:   &runtimeversion.Version{Major: 1, Minor: 4, Patch: 9},
			wantFailureCount: 1,
		},
		{
			name:             "minimum supported runtime",
			runtimeVersion:   &runtimeversion.Version{Major: 1, Minor: 5, Patch: 0},
			wantFailureCount: 3,
		},
		{
			name:             "newer runtime",
			runtimeVersion:   &runtimeversion.Version{Major: 2, Minor: 0, Patch: 0},
			wantFailureCount: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("render the worker defaults for the resolved runtime version")
			container, err := NewWorkerDefaults().GetBaseContainer(ComponentContext{
				RuntimeVersion: tt.runtimeVersion,
			})
			require.NoError(t, err)

			t.Log("verify the version-gated liveness failure threshold")
			require.NotNil(t, container.LivenessProbe)
			require.Equal(t, tt.wantFailureCount, container.LivenessProbe.FailureThreshold)
		})
	}
}
