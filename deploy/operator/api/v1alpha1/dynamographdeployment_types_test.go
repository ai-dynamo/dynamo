/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtraPodSpecMergeStrategy_RoundTripAndValidation(t *testing.T) {
	dgd := DynamoGraphDeployment{
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType:             "frontend",
					ExtraPodSpecMergeStrategy: ExtraPodSpecMergeStrategyStrategic,
				},
			},
		},
	}

	data, err := json.Marshal(dgd)
	require.NoError(t, err)

	var restored DynamoGraphDeployment
	require.NoError(t, json.Unmarshal(data, &restored))

	assert.Equal(
		t,
		ExtraPodSpecMergeStrategyStrategic,
		restored.Spec.Services["frontend"].ExtraPodSpecMergeStrategy,
	)
	assert.True(t, ExtraPodSpecMergeStrategyOverride.IsValid())
	assert.True(t, ExtraPodSpecMergeStrategyStrategic.IsValid())
	assert.False(t, ExtraPodSpecMergeStrategy("invalid").IsValid())
}

func TestResolveExtraPodSpecMergeStrategy_DefaultsToOverride(t *testing.T) {
	got, err := ResolveExtraPodSpecMergeStrategy("", "")
	require.NoError(t, err)
	assert.Equal(t, ExtraPodSpecMergeStrategyOverride, got)
}
