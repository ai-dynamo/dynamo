//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestGroveCapacityUsesRealScaleSubresource(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := t.Context()

	t.Log("create a fresh Engine Group and bind the exact generated member-clique UID")
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "group", Namespace: env.Namespace()},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	require.NoError(t, env.Client().Create(ctx, group))
	primary := testSGLangPrimaryPod()
	primary.Namespace = env.Namespace()
	primary.UID = ""
	primary.Status = corev1.PodStatus{}
	primary.Spec.Containers[0].Image = "test-engine:latest"
	clique := testSGLangMemberClique(primary)
	clique.UID = ""
	require.NoError(t, env.Client().Create(ctx, clique))
	primary.OwnerReferences[0].UID = clique.UID
	require.NoError(t, env.Client().Create(ctx, primary))
	primary.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
	require.NoError(t, env.Client().Status().Update(ctx, primary))
	adapter := &grovecapacity.Adapter{
		Client: env.Client(), Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID,
		GroupName: group.Name,
		Journal:   kubejournal.NewStore(env.Client(), group.Namespace, group.Name, group.UID, "grove-capacity"),
	}

	t.Log("observe the exact primary and persist one absolute grow target before writing /scale")
	observed, err := adapter.Observe(ctx, engineGroupID(group))
	require.NoError(t, err)
	require.Len(t, observed.Allocations, 1)
	primaryIncarnation := observed.Allocations[0].Incarnation
	target := enginegroup.CapacityTarget{
		ControlRevision: 1, TransitionID: "grow", ProfileFingerprint: "profile",
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		Replicas: []enginegroup.CapacityReplicaTarget{
			{ReplicaID: "replica-0", SlotID: "slot-0", Incarnation: &primaryIncarnation},
			{ReplicaID: "replica-1", SlotID: "slot-1", Bootstrap: &enginegroup.CapacityBootstrap{
				Mode: enginegroup.BootstrapModeJoin, BaseTopologyGeneration: 1,
				NativeMembers: []enginegroup.NativeMemberID{"dp-1"},
			}},
		},
	}
	result, err := adapter.Apply(ctx, engineGroupID(group), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)

	t.Log("read native Scale metadata and confirm that only the member count changed")
	scale := &autoscalingv1.Scale{}
	require.NoError(t, env.Client().SubResource("scale").Get(ctx, clique, scale))
	assert.Equal(t, clique.UID, scale.UID)
	assert.Equal(t, int32(2), scale.Spec.Replicas)
	assert.NotEmpty(t, scale.ResourceVersion)
	stored := &grovev1alpha1.PodClique{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(clique), stored))
	assert.Equal(t, clique.Spec.PodSpec, stored.Spec.PodSpec)
	assert.Equal(t, clique.OwnerReferences, stored.OwnerReferences)
	pods := &corev1.PodList{}
	require.NoError(t, env.Client().List(ctx, pods, client.InNamespace(env.Namespace())))
	assert.Len(t, pods.Items, 1, "the capacity adapter must not create joiner Pods itself")

	t.Log("restart the adapter and replay without inventing an allocation or changing the template")
	adapter = &grovecapacity.Adapter{
		Client: env.Client(), Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID,
		GroupName: group.Name,
		Journal:   kubejournal.NewStore(env.Client(), group.Namespace, group.Name, group.UID, "grove-capacity"),
	}
	result, err = adapter.Apply(ctx, engineGroupID(group), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	observed, err = adapter.Observe(ctx, engineGroupID(group))
	require.NoError(t, err)
	assert.Equal(t, int64(1), observed.AppliedRevision)
	assert.Len(t, observed.Allocations, 1, "acceptance is not capacity convergence")
}
