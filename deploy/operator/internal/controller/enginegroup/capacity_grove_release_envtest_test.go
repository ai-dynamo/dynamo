//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"
	"slices"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestGroveReadinessReleaseUsesRealStatusAndScaleSubresources(t *testing.T) {
	t.Log("create a two-allocation clique with an API-assigned UID for every participant")
	env := sharedEnv.ForTest(t)
	ctx := t.Context()
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "release-world", Namespace: env.Namespace()},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	require.NoError(t, env.Client().Create(ctx, group))
	gate := corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)
	clique := &grove.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: "release-members", Namespace: env.Namespace(),
			Labels: map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name},
			OwnerReferences: []metav1.OwnerReference{{APIVersion: grove.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup",
				Name: "world", UID: "world-uid", Controller: ptr.To(true)}},
		},
		Spec: grove.PodCliqueSpec{Replicas: 2, MinAvailable: ptr.To(int32(1)), PodSpec: corev1.PodSpec{
			RestartPolicy:  corev1.RestartPolicyNever,
			ReadinessGates: []corev1.PodReadinessGate{{ConditionType: gate}},
			Containers:     []corev1.Container{{Name: "main", Image: "engine:test"}},
		}},
	}
	require.NoError(t, env.Client().Create(ctx, clique))
	pods := make([]*corev1.Pod, 2)
	for rank := range int32(2) {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("release-member-%d", rank), Namespace: clique.Namespace,
				Labels: map[string]string{
					consts.KubeLabelDynamoEngineGroup:         group.Name,
					consts.KubeLabelDynamoScaleRepresentative: consts.KubeLabelDynamoScaleRepresentativeYes,
					grovecommon.LabelPodClique:                clique.Name,
					grovecommon.LabelPodCliquePodIndex:        fmt.Sprint(rank),
				},
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(clique, grove.SchemeGroupVersion.WithKind("PodClique"))},
			},
			Spec: *clique.Spec.PodSpec.DeepCopy(),
		}
		pod.Spec.NodeName = "test-node"
		require.NoError(t, env.Client().Create(ctx, pod))
		pod.Status.Phase = corev1.PodRunning
		pod.Status.Conditions = []corev1.PodCondition{
			{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			{Type: corev1.ContainersReady, Status: corev1.ConditionTrue},
			{Type: corev1.PodScheduled, Status: corev1.ConditionTrue},
		}
		require.NoError(t, env.Client().Status().Update(ctx, pod))
		pods[rank] = pod
	}
	adapter := &grovecapacity.Adapter{
		Client: env.Client(), Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID, GroupName: group.Name,
		Journal: kubejournal.NewStore(env.Client(), group.Namespace, group.Name, group.UID, "grove-capacity"), ReadinessScaleIn: true,
	}
	require.NoError(t, adapter.InitializeRetention(ctx))
	observation, err := adapter.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	retained, victim := groveCapacityAssertion(observation.Allocations[0]), groveCapacityAssertion(observation.Allocations[1])
	target := domain.CapacityTarget{
		ControlRevision: 1, TransitionID: "retire", ProfileFingerprint: "profile", ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator,
		Replicas: []domain.CapacityReplicaTarget{{ReplicaID: retained.ReplicaID, SlotID: retained.SlotID, Incarnation: &retained}},
		ReleaseFences: []domain.ReleaseFence{{ReplicaID: victim.ReplicaID, SlotID: victim.SlotID,
			CapacityRefs: victim.CapacityRefs, TransitionID: "retire", AuthorizingTopologyGeneration: 2}},
	}

	t.Log("strategic status merge closes the retention gate without replacing kubelet-owned conditions")
	result, err := adapter.Apply(ctx, domain.GroupID(group.UID), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(pods[1]), pods[1]))
	require.Len(t, pods[1].Status.Conditions, 4)
	require.True(t, slices.ContainsFunc(pods[1].Status.Conditions, func(condition corev1.PodCondition) bool {
		return condition.Type == gate && condition.Status == corev1.ConditionFalse
	}))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(clique), clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("simulate the kubelet's NotReady observation, then exercise the real Scale endpoint")
	for i := range pods[1].Status.Conditions {
		if pods[1].Status.Conditions[i].Type == corev1.PodReady {
			pods[1].Status.Conditions[i].Status = corev1.ConditionFalse
		}
	}
	require.NoError(t, env.Client().Status().Update(ctx, pods[1]))
	result, err = adapter.Apply(ctx, domain.GroupID(group.UID), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(clique), clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)

	t.Log("simulate Grove deleting the selected UID; receipt publication is read-only and idempotent")
	require.NoError(t, env.Client().Delete(ctx, pods[1], client.GracePeriodSeconds(0), client.Preconditions{UID: &pods[1].UID}))
	observation, err = adapter.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	require.Equal(t, target.ReleaseFences, observation.ReleaseFences)
	result, err = adapter.Apply(ctx, domain.GroupID(group.UID), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
}
