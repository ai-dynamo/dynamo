/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package podcapacity

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestAdapterPersistsBeforeCreatingJoiningPod(t *testing.T) {
	ctx := context.Background()
	primary := testPrimaryPod()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(primary).Build()
	adapter := NewAdapter(kubeClient, "test", "group", types.UID("group-uid"), testPodBuilder{})
	target := enginegroup.CapacityTarget{
		ControlRevision:       1,
		TransitionID:          "grow",
		ProfileFingerprint:    "profile",
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		Replicas: []enginegroup.CapacityReplicaTarget{
			{
				ReplicaID: "replica-0",
				SlotID:    "slot-0",
				Incarnation: &enginegroup.ReplicaIncarnation{
					ReplicaID:          "replica-0",
					SlotID:             "slot-0",
					RuntimeIncarnation: "primary-uid",
					CapacityRefs:       []enginegroup.CapacityRef{{Name: "primary", UID: "primary-uid"}},
				},
			},
			{
				ReplicaID: "replica-1",
				SlotID:    "slot-1",
				Bootstrap: &enginegroup.CapacityBootstrap{
					Mode:          enginegroup.BootstrapModeJoin,
					NativeMembers: []enginegroup.NativeMemberID{"dp-1"},
				},
			},
		},
	}

	t.Log("accept the absolute capacity target durably and create only missing capacity")
	result, err := adapter.Apply(ctx, "group-uid", target)
	require.NoError(t, err)
	assert.Nil(t, result.Rejection)

	state := journalState{}
	found, err := adapter.Journal.Load(ctx, &state)
	require.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, int64(1), state.AppliedRevision)

	pods := &corev1.PodList{}
	require.NoError(t, kubeClient.List(ctx, pods))
	require.Len(t, pods.Items, 2)
	var joiner *corev1.Pod
	for i := range pods.Items {
		if pods.Items[i].Name == "joiner" {
			joiner = &pods.Items[i]
		}
	}
	require.NotNil(t, joiner)
	assert.Equal(t, "replica-1", joiner.Labels[consts.KubeLabelDynamoEngineGroupReplica])
	assert.Equal(t, consts.KubeLabelDynamoScaleRepresentativeYes, joiner.Labels[consts.KubeLabelDynamoScaleRepresentative])
	assert.Equal(t, "group-uid", string(joiner.OwnerReferences[0].UID))

	t.Log("reject conflicting content at the already accepted revision")
	target.ProfileFingerprint = "other-profile"
	result, err = adapter.Apply(ctx, "group-uid", target)
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "ConflictingCapacityRevision", result.Rejection.Reason)
}

type testPodBuilder struct{}

func (testPodBuilder) Build(_ *corev1.Pod, _ enginegroup.CapacityReplicaTarget) (*corev1.Pod, error) {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "joiner", Labels: map[string]string{}},
		Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "test"}}},
	}, nil
}

func testPrimaryPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "primary",
			UID:       types.UID("primary-uid"),
			Labels: map[string]string{
				consts.KubeLabelDynamoEngineGroup:        "group",
				consts.KubeLabelDynamoEngineGroupReplica: "replica-0",
				consts.KubeLabelDynamoEngineGroupSlot:    "slot-0",
				consts.KubeLabelDynamoEngineGroupRole:    consts.KubeLabelDynamoEngineGroupRolePrimary,
			},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "test"}}},
	}
}
