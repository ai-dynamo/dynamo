//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEngineGroupReleaseModeDoesNotGrantExactReleaseOrChangeServingHealth(t *testing.T) {
	tests := []struct {
		name   string
		mode   domain.CapacityReleaseMode
		status metav1.ConditionStatus
		reason string
	}{
		{name: "unknown", status: metav1.ConditionUnknown, reason: "ReleaseCapabilityNotReported"},
		{name: "unsupported", mode: domain.CapacityReleaseUnsupported, status: metav1.ConditionFalse, reason: "ReleaseUnsupported"},
		{name: "readiness steering is not exact", mode: domain.CapacityReleaseReadinessSteered, status: metav1.ConditionFalse, reason: "ReadinessSteeredRelease"},
		{name: "exact UID authorization", mode: domain.CapacityReleaseExactUID, status: metav1.ConditionTrue, reason: "ExactUIDRelease"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("project the capacity provider's release mechanism onto an already healthy world")
			group := &api.DynamoGraphDeploymentEngineGroup{Generation: 1}
			setEngineGroupCondition(group, engineGroupConditionAvailable, metav1.ConditionTrue, "ServingAvailable", "Serving")
			projectEngineGroupReleaseMode(group, test.mode)

			t.Log("report only proven release guarantees without changing serving health")
			condition := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionExactReleaseSupported)
			require.NotNil(t, condition)
			assert.Equal(t, test.status, condition.Status)
			assert.Equal(t, test.reason, condition.Reason)
			assert.True(t, meta.IsStatusConditionTrue(group.Status.Conditions, engineGroupConditionAvailable))
		})
	}
}
