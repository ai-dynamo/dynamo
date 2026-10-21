/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"
)

func TestApplyAndClearComponentGPUShapes(t *testing.T) {
	t.Log("Apply provider-resolved shapes to observed component statuses")
	statuses := map[string]nvidiacomv1beta1.ComponentReplicaStatus{
		"worker":   {},
		"frontend": {},
	}
	applyComponentGPUShapes(statuses, map[string]dynamo.GPUShape{
		"worker":   {GPUsPerEngine: 4, GPUsPerReplica: 5},
		"frontend": {},
	})
	require.NotNil(t, statuses["worker"].GPUsPerEngine)
	require.NotNil(t, statuses["worker"].GPUsPerReplica)
	assert.Equal(t, int64(4), *statuses["worker"].GPUsPerEngine)
	assert.Equal(t, int64(5), *statuses["worker"].GPUsPerReplica)
	require.NotNil(t, statuses["frontend"].GPUsPerEngine)
	require.NotNil(t, statuses["frontend"].GPUsPerReplica)
	assert.Equal(t, int64(0), *statuses["frontend"].GPUsPerEngine)
	assert.Equal(t, int64(0), *statuses["frontend"].GPUsPerReplica)

	t.Log("Clear both fields before a later provider render can fail")
	statuses["unobserved"] = nvidiacomv1beta1.ComponentReplicaStatus{
		GPUsPerEngine:  ptr.To(int64(2)),
		GPUsPerReplica: ptr.To(int64(2)),
	}
	clearComponentGPUShapes(statuses)
	assert.Nil(t, statuses["worker"].GPUsPerEngine)
	assert.Nil(t, statuses["worker"].GPUsPerReplica)
	assert.Nil(t, statuses["frontend"].GPUsPerEngine)
	assert.Nil(t, statuses["frontend"].GPUsPerReplica)
	assert.Nil(t, statuses["unobserved"].GPUsPerEngine)
	assert.Nil(t, statuses["unobserved"].GPUsPerReplica)
}

func TestApplyAndClearComponentRuntimeStatuses(t *testing.T) {
	t.Log("Apply provider-resolved serving-role facts to observed component statuses")
	statuses := map[string]nvidiacomv1beta1.ComponentReplicaStatus{
		"decode": {},
	}
	applyComponentRuntimeStatuses(statuses, map[string]dynamo.ComponentRuntimeStatus{
		"decode": {
			ServedModelName:      "Qwen/Qwen3-8B",
			RuntimeComponentName: "custom-decode",
			GPUPowerLimitWatts:   ptr.To(int64(300)),
		},
		"unobserved": {ServedModelName: "ignored"},
	})
	assert.Equal(t, "Qwen/Qwen3-8B", statuses["decode"].ServedModelName)
	assert.Equal(t, "custom-decode", statuses["decode"].RuntimeComponentName)
	require.NotNil(t, statuses["decode"].GPUPowerLimitWatts)
	assert.Equal(t, int64(300), *statuses["decode"].GPUPowerLimitWatts)
	assert.NotContains(t, statuses, "unobserved")

	t.Log("Clear all runtime facts before a later provider render can fail")
	clearComponentRuntimeStatuses(statuses)
	assert.Empty(t, statuses["decode"].ServedModelName)
	assert.Empty(t, statuses["decode"].RuntimeComponentName)
	assert.Nil(t, statuses["decode"].GPUPowerLimitWatts)
}
