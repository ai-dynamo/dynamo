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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestProductionRuntimeProviderResolvesSGLangEP1Profile(t *testing.T) {
	t.Log("create a Grove-owned primary with one stable member-clique binding")
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))
	primary := testSGLangPrimaryPod()
	clique := testSGLangMemberClique(primary)
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(primary, clique).Build()
	provider := newEngineGroupRuntimeProvider(kubeClient)
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test", Name: "group", UID: types.UID("group-uid"),
			Labels: map[string]string{consts.KubeLabelDynamoEngineGroupRuntime: consts.KubeLabelDynamoEngineGroupSGLang},
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoEngineGroupVerifyURL:    "http://frontend.test:8000/v1/completions",
				consts.KubeAnnotationDynamoEngineGroupVerifyModel:  "model",
				consts.KubeAnnotationDynamoEngineGroupPodClique:    clique.Name,
				consts.KubeAnnotationDynamoEngineGroupPodCliqueUID: string(clique.UID),
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
	_, groveCapacity := resolved.Capacity.(*grovecapacity.Adapter)
	assert.True(t, groveCapacity)
	assert.NotNil(t, resolved.Membership)
	assert.NotNil(t, resolved.Traffic)
	assert.NotNil(t, resolved.Verifier)
	assert.NotNil(t, resolved.Planner)
}

func TestProductionRuntimeProviderRequiresGroveBootstrap(t *testing.T) {
	// Each case changes one input on an otherwise valid native Grove binding.
	cases := []struct {
		name     string
		command  []string
		boundUID types.UID
		want     string
	}{
		{name: "ordinary worker cannot launch joiners from its template", command: []string{"python3", "-m", "dynamo.sglang"}, boundUID: "clique-uid", want: "template-invariant bootstrap"},
		{name: "clique recreation cannot reuse the old group", command: []string{"python3", "-m", dynamo.SGLangElasticEPBootstrapModule}, boundUID: "previous-clique-uid", want: "replaced"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("create a fresh group and its Grove workload with the selected invalid binding")
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, grovev1alpha1.AddToScheme(scheme))
			primary := testSGLangPrimaryPod()
			primary.Spec.Containers[0].Command = tc.command
			clique := testSGLangMemberClique(primary)
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(primary, clique).Build()
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{
				Namespace: "test", Name: "group", UID: "group-uid",
				Labels: map[string]string{consts.KubeLabelDynamoEngineGroupRuntime: consts.KubeLabelDynamoEngineGroupSGLang},
				Annotations: map[string]string{
					consts.KubeAnnotationDynamoEngineGroupPodClique:    clique.Name,
					consts.KubeAnnotationDynamoEngineGroupPodCliqueUID: string(tc.boundUID),
				},
			}}

			t.Log("refuse to construct a runtime whose capacity or bootstrap cannot be proven")
			_, err := newEngineGroupRuntimeProvider(kubeClient).Resolve(t.Context(), group)
			require.ErrorContains(t, err, tc.want)
		})
	}
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
				consts.KubeLabelDynamoScaleRepresentative: consts.KubeLabelDynamoScaleRepresentativeYes,
				grovecommon.LabelPodClique:                "world-0-members",
				grovecommon.LabelPodCliquePodIndex:        "0",
				grovecommon.LabelPodTemplateHash:          "worker-revision",
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodClique",
				Name: "world-0-members", UID: "clique-uid", Controller: ptr.To(true),
			}},
		},
		Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever, Containers: []corev1.Container{{
			Name:    consts.MainContainerName,
			Command: []string{"python3", "-m", dynamo.SGLangElasticEPBootstrapModule},
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

func testSGLangMemberClique(primary *corev1.Pod) *grovev1alpha1.PodClique {
	return &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: primary.Namespace, Name: "world-0-members", UID: "clique-uid",
			Labels: map[string]string{consts.KubeLabelDynamoEngineGroup: "group"},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup",
				Name: "world", UID: "world-uid", Controller: ptr.To(true),
			}},
		},
		Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1, MinAvailable: ptr.To(int32(1)), PodSpec: *primary.Spec.DeepCopy()},
	}
}
