//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEngineGroupReleaseModeDoesNotGrantExactReleaseOrChangeServingHealth(t *testing.T) {
	t.Log("start with a healthy serving world using readiness-steered Grove release")
	group := &api.DynamoGraphDeploymentEngineGroup{Generation: 1}
	setEngineGroupCondition(group, engineGroupConditionAvailable, metav1.ConditionTrue, "ServingAvailable", "Serving")
	projectEngineGroupReleaseMode(group, Runtime{ReadinessSteeredRelease: true})

	t.Log("make the experimental release limitation visible without changing availability")
	condition := meta.FindStatusCondition(group.Status.Conditions, "ExactReleaseSupported")
	require.NotNil(t, condition)
	assert.Equal(t, metav1.ConditionFalse, condition.Status)
	assert.Equal(t, "ReadinessSteeredRelease", condition.Reason)
	assert.True(t, meta.IsStatusConditionTrue(group.Status.Conditions, engineGroupConditionAvailable))

	t.Log("an unspecified capability cannot accidentally become exact-release support")
	projectEngineGroupReleaseMode(group, Runtime{})
	condition = meta.FindStatusCondition(group.Status.Conditions, "ExactReleaseSupported")
	assert.Equal(t, metav1.ConditionUnknown, condition.Status)
}
