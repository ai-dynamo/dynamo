/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestProductionRuntimeProviderResolvesSGLangEP1Profile(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	primary := testSGLangPrimaryPod()
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(primary).Build()
	provider := newProductionEngineGroupRuntimeProvider(kubeClient)
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test", Name: "group", UID: types.UID("group-uid"),
			Labels: map[string]string{consts.KubeLabelDynamoEngineGroupRuntime: consts.KubeLabelDynamoEngineGroupSGLang},
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoEngineGroupVerifyURL:   "http://frontend.test:8000/v1/completions",
				consts.KubeAnnotationDynamoEngineGroupVerifyModel: "model",
			},
		},
	}

	t.Log("resolve the explicitly selected SGLang runtime from stable workload metadata")
	resolved, err := provider.Resolve(context.Background(), group)
	require.NoError(t, err)
	assert.Equal(t, "sglang", resolved.Profile.Backend)
	assert.Equal(t, int32(1), resolved.Profile.PodsPerReplica)
	assert.Equal(t, int32(2), resolved.Profile.MaxSupportedReplicas)
	assert.NotNil(t, resolved.Capacity)
	assert.NotNil(t, resolved.Membership)
	assert.NotNil(t, resolved.Traffic)
	assert.NotNil(t, resolved.Verifier)
	assert.NotNil(t, resolved.Planner)
}

func TestSGLangGrowthPlannerBuildsContiguousIdentityPlan(t *testing.T) {
	planner := sglangGrowthPlanner{profileFingerprint: "profile-v1"}
	status := enginegroup.GroupStatus{Membership: enginegroup.MembershipStatus{Observed: enginegroup.MembershipObservation{
		CommittedTopology: enginegroup.MembershipTopology{
			Generation: 1,
			Replicas: []enginegroup.ReplicaMembership{{
				ReplicaID: "replica-0", RuntimeIncarnation: "pod-0", NativeMembers: []enginegroup.NativeMemberID{"dp-0"},
			}},
		},
	}}}

	t.Log("map the absolute Kubernetes target to one concrete SGLang grow plan")
	resolution, err := planner.ResolveScalePlan(context.Background(), "group", 2, status)
	require.NoError(t, err)
	require.NotNil(t, resolution.Plan)
	require.NotNil(t, resolution.Plan.Change.Grow)
	require.Len(t, resolution.Plan.Change.Grow.Replicas, 1)
	assert.Equal(t, enginegroup.ReplicaID("replica-1"), resolution.Plan.Change.Grow.Replicas[0].ReplicaID)
	assert.Equal(t, []enginegroup.NativeMemberID{"dp-1"}, resolution.Plan.Change.Grow.Replicas[0].NativeMembers)

	t.Log("reject shrink as a definitive backend capability boundary")
	resolution, err = planner.ResolveScalePlan(context.Background(), "group", 0, status)
	require.NoError(t, err)
	require.NotNil(t, resolution.Rejection)
	assert.Equal(t, "SGLangShrinkUnsupported", resolution.Rejection.Reason)
}

func testSGLangPrimaryPod() *corev1.Pod {
	gpu := resource.MustParse("1")
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test", Name: "primary", UID: types.UID("primary-uid"),
			Labels: map[string]string{
				consts.KubeLabelDynamoEngineGroup:         "group",
				consts.KubeLabelDynamoEngineGroupReplica:  "replica-0",
				consts.KubeLabelDynamoEngineGroupSlot:     "slot-0",
				consts.KubeLabelDynamoEngineGroupRole:     consts.KubeLabelDynamoEngineGroupRolePrimary,
				consts.KubeLabelDynamoScaleRepresentative: consts.KubeLabelDynamoScaleRepresentativeYes,
				consts.KubeLabelDynamoWorkerHash:          "worker-revision",
			},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name:    consts.MainContainerName,
			Command: []string{"python3", "-m", "dynamo.sglang"},
			Args: []string{
				"--model-path", "model", "--served-model-name", "model",
				"--tp", "1", "--dp", "1", "--nnodes", "1",
				"--enable-dp-attention", "--enable-dp-lm-head",
				"--moe-a2a-backend", "nixl", "--elastic-ep-backend", "mooncake",
				"--load-balance-method", "round_robin",
				"--elastic-ep-initial-size", "1", "--max-ep-size", "2",
				"--disable-cuda-graph", "--dist-init-addr", "rendezvous.test:24555",
			},
			Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": gpu}},
		}}},
		Status: corev1.PodStatus{PodIP: "10.0.0.8"},
	}
}
