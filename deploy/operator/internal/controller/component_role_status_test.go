// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"strings"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

func TestComponentRoleStatusGenerationAndRollout(t *testing.T) {
	t.Log("A target snapshot advances with global generation even while a rollout is pending")
	dgd := &v1beta1.DynamoGraphDeployment{Status: v1beta1.DynamoGraphDeploymentStatus{
		ObservedGeneration: 6,
		Components:         map[string]v1beta1.ComponentReplicaStatus{"worker": {Roles: []v1beta1.ComponentRoleStatus{{Name: "old-target"}}}},
		RollingUpdate:      &v1beta1.RollingUpdateStatus{Phase: v1beta1.RollingUpdatePhaseInProgress},
	}}
	workloads := ReconcileResult{State: v1beta1.DGDStatePending,
		ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{"worker": {Roles: []v1beta1.ComponentRoleStatus{{Name: "new-target"}}}},
	}
	result := newWorkloadProgramResult(dgd)
	result.applyReconcileResult(7, workloads)
	require.EqualValues(t, 7, result.Status.ObservedGeneration)
	require.Equal(t, "new-target", result.Status.Components["worker"].Roles[0].Name)

	t.Log("Retaining partial observations after error must not mix target generations")
	preserveComponentRoleStatuses(dgd.Status.Components, workloads.ComponentStatus)
	require.Equal(t, "old-target", workloads.ComponentStatus["worker"].Roles[0].Name)
	require.EqualValues(t, 6, dgd.Status.ObservedGeneration)

	t.Log("Old-serving identity cannot replace the new target roles during rollout")
	old := map[string]v1beta1.ComponentReplicaStatus{"worker": {RuntimeNamespace: "old-runtime", Roles: []v1beta1.ComponentRoleStatus{{Name: "old-target"}}}}
	current := map[string]v1beta1.ComponentReplicaStatus{"worker": {RuntimeNamespace: "new-runtime", Roles: []v1beta1.ComponentRoleStatus{{Name: "new-target"}}}}
	mergeWorkerComponentStatuses(current, old)
	require.Equal(t, "old-runtime", current["worker"].RuntimeNamespace)
	require.Equal(t, "new-target", current["worker"].Roles[0].Name)
	missing := map[string]v1beta1.ComponentReplicaStatus{}
	mergeWorkerComponentStatuses(missing, old)
	require.Nil(t, missing["worker"].Roles)
}

func TestTargetRoleStatusWholeObjectHeadroom(t *testing.T) {
	t.Log("An existing large spec keeps its generation acknowledgement without a new oversized export")
	dgd := &v1beta1.DynamoGraphDeployment{Generation: 7, Spec: v1beta1.DynamoGraphDeploymentSpec{
		Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "frontend", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Args: []string{strings.Repeat("x", maxTargetStatusObjectBytes)}}},
		}}}},
	}}
	original := dgd.DeepCopy()
	result := newWorkloadProgramResult(dgd)
	result.applyReconcileResult(7, ReconcileResult{State: v1beta1.DGDStatePending, ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
		"frontend": {Roles: []v1beta1.ComponentRoleStatus{{Name: "main"}}, Replicas: 1},
	}})
	result.boundTargetRoleStatus(dgd)
	require.EqualValues(t, 7, result.Status.ObservedGeneration)
	require.Nil(t, result.Status.Components["frontend"].Roles)
	require.EqualValues(t, 1, result.Status.Components["frontend"].Replicas)
	require.Equal(t, original, dgd)
}

func TestDisaggregatedSetTargetRoleStatus(t *testing.T) {
	t.Log("Use the persisted DisaggregatedSet target before any child LWS or Pod is ready")
	templates := leaderworkersetv1.LeaderWorkerSetSpec{LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
		LeaderTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "leader:new"}}}},
		WorkerTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "worker:new"}}}},
	}}
	raw, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&templates)
	require.NoError(t, err)
	ds := &unstructured.Unstructured{Object: map[string]any{"spec": map[string]any{
		"roles": []any{map[string]any{"name": "prefill", "spec": raw}},
	}}}
	original := ds.DeepCopy()
	statuses := map[string]v1beta1.ComponentReplicaStatus{"Prefill": {}}
	err = applyDisaggregatedSetRoleStatuses(ds, disaggregatedSetSelection{componentToRole: map[string]string{"Prefill": "prefill"}}, statuses)
	require.NoError(t, err)
	require.Len(t, statuses["Prefill"].Roles, 2)
	require.Equal(t, "leader:new", statuses["Prefill"].Roles[0].Containers[0].Image)
	require.Equal(t, "worker:new", statuses["Prefill"].Roles[1].Containers[0].Image)
	require.Equal(t, original, ds)
}
