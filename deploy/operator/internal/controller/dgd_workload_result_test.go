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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
		GPUsPerEngine:  new(int64(2)),
		GPUsPerReplica: new(int64(2)),
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
			GPUPowerLimitWatts:   new(int64(300)),
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

// crashLoopTestDCD describes DCD name of component at observedGeneration whose
// Available condition carries reason and message.
func crashLoopTestDCD(
	name string,
	component string,
	observedGeneration int64,
	reason string,
	message string,
) *nvidiacomv1beta1.DynamoComponentDeployment {
	return &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Generation: 2},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: component,
			},
		},
		Status: nvidiacomv1beta1.DynamoComponentDeploymentStatus{
			ObservedGeneration: observedGeneration,
			Conditions: []metav1.Condition{{
				Type:    nvidiacomv1beta1.DynamoComponentDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  reason,
				Message: message,
			}},
		},
	}
}

func TestApplyComponentCrashLoops(t *testing.T) {
	pending := ReconcileResult{
		State:   nvidiacomv1beta1.DGDStatePending,
		Reason:  "some_resources_are_not_ready",
		Message: "Resources not ready",
	}

	tests := []struct {
		name string
		dcds []*nvidiacomv1beta1.DynamoComponentDeployment
		want ReconcileResult
	}{
		{
			name: "crash-looping components fail the graph",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				crashLoopTestDCD("graph-worker", "worker", 2, dcdReasonContainerCrashLooping, "pod worker-a container main restarted 2 times, last exit code 1 (Error)"),
				crashLoopTestDCD("graph-frontend", "frontend", 2, "DeploymentNotReady", "Deployment is not ready"),
				crashLoopTestDCD("graph-prefill", "prefill", 2, dcdReasonContainerCrashLooping, "pod prefill-a container main restarted 4 times, last exit code 137"),
			},
			want: ReconcileResult{
				State:  nvidiacomv1beta1.DGDStateFailed,
				Reason: reasonComponentCrashLooping,
				Message: "Component prefill is crash-looping: pod prefill-a container main restarted 4 times, last exit code 137; " +
					"Component worker is crash-looping: pod worker-a container main restarted 2 times, last exit code 1 (Error)",
			},
		},
		{
			name: "crash loop reported for an older DCD generation",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				crashLoopTestDCD("graph-worker", "worker", 1, dcdReasonContainerCrashLooping, "pod worker-a container main restarted 2 times, last exit code 1"),
			},
			want: pending,
		},
		{
			name: "unready components without a crash loop",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				crashLoopTestDCD("graph-worker", "worker", 2, "DeploymentNotReady", "Deployment is not ready"),
			},
			want: pending,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Project DCD crash loops onto the pending workload observation")
			got := applyComponentCrashLoops(pending, tt.dcds)

			t.Log("Verify the graph-level state, reason and message")
			assert.Equal(t, tt.want, got)
		})
	}
}
