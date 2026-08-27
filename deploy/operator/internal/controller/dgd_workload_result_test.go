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
		"worker": {},
	}
	applyComponentGPUShapes(statuses, map[string]dynamo.GPUShape{
		"worker": {GPUsPerEngine: 4, GPUsPerReplica: 5},
	})
	require.NotNil(t, statuses["worker"].GPUsPerEngine)
	require.NotNil(t, statuses["worker"].GPUsPerReplica)
	assert.Equal(t, int64(4), *statuses["worker"].GPUsPerEngine)
	assert.Equal(t, int64(5), *statuses["worker"].GPUsPerReplica)

	t.Log("Clear both fields before a later provider render can fail")
	statuses["unobserved"] = nvidiacomv1beta1.ComponentReplicaStatus{
		GPUsPerEngine:  ptr.To(int64(2)),
		GPUsPerReplica: ptr.To(int64(2)),
	}
	clearComponentGPUShapes(statuses)
	assert.Nil(t, statuses["worker"].GPUsPerEngine)
	assert.Nil(t, statuses["worker"].GPUsPerReplica)
	assert.Nil(t, statuses["unobserved"].GPUsPerEngine)
	assert.Nil(t, statuses["unobserved"].GPUsPerReplica)
}
