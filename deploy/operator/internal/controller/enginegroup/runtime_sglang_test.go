/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"strconv"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	sglangruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/sglang"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
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

func TestGroveRuntimeJournalsArePhysicalLifetimeScoped(t *testing.T) {
	t.Log("resolve one logical group against its initial physical world")
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))
	primary := testSGLangPrimaryPod()
	clique := testSGLangMemberClique(primary)
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(primary, clique).Build()
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{
		Name: "group", Namespace: "test", UID: "logical-world",
		Labels:      map[string]string{consts.KubeLabelDynamoEngineGroupRuntime: consts.KubeLabelDynamoEngineGroupSGLang},
		Annotations: map[string]string{consts.KubeAnnotationDynamoEngineGroupPodClique: clique.Name, consts.KubeAnnotationDynamoEngineGroupPodCliqueUID: string(clique.UID)},
	}}
	provider := newEngineGroupRuntimeProvider(kube)
	previous, err := provider.Resolve(t.Context(), group)
	require.NoError(t, err)

	t.Log("recreate native resources without changing the logical group UID or template")
	require.NoError(t, kube.Delete(t.Context(), primary))
	require.NoError(t, kube.Delete(t.Context(), clique))
	clique.UID, clique.ResourceVersion = "new-clique", ""
	primary.UID, primary.ResourceVersion = "new-pod", ""
	primary.OwnerReferences[0].UID = clique.UID
	require.NoError(t, kube.Create(t.Context(), clique))
	require.NoError(t, kube.Create(t.Context(), primary))
	group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = string(clique.UID)
	current, err := provider.Resolve(t.Context(), group)
	require.NoError(t, err)
	assert.Equal(t, previous.Profile, current.Profile)

	t.Log("all three adapters use new stores, so old pending requests and release targets cannot leak")
	assert.NotEqual(t, previous.Capacity.(*grovecapacity.Adapter).Journal.Name, current.Capacity.(*grovecapacity.Adapter).Journal.Name)
	assert.NotEqual(t, previous.Membership.(*sglangruntime.LegacyGrowthAdapter).Journal.Name, current.Membership.(*sglangruntime.LegacyGrowthAdapter).Journal.Name)
	assert.NotEqual(t, previous.Traffic.(*sglangruntime.TrafficProjection).Journal.Name, current.Traffic.(*sglangruntime.TrafficProjection).Journal.Name)
	assert.Equal(t, group.UID, current.Membership.(*sglangruntime.LegacyGrowthAdapter).Journal.Owner.UID)
	assert.NotEqual(t, previous.Planner.(sglangGrowthPlanner).worldUID, current.Planner.(sglangGrowthPlanner).worldUID)
}

func TestProductionRuntimeProviderResolvesConfiguredSGLangProfile(t *testing.T) {
	cases := []struct {
		name    string
		initial int32
		maximum int32
	}{
		{name: "EP2 to EP3 fixture", initial: 2, maximum: 3},
		{name: "EP4 to EP8", initial: 4, maximum: 8},
		{name: "EP8 to EP16", initial: 8, maximum: 16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("create a Grove-owned primary whose immutable template declares the configured geometry")
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, grovev1alpha1.AddToScheme(scheme))
			primary := testSGLangPrimaryPod()
			args := primary.Spec.Containers[0].Args
			for index, option := range args {
				switch option {
				case "--tp", "--dp", "--nnodes", "--elastic-ep-initial-size":
					args[index+1] = strconv.Itoa(int(tc.initial))
				case "--max-ep-size":
					args[index+1] = strconv.Itoa(int(tc.maximum))
				}
			}
			primary = podcache.Project(primary)
			clique := testSGLangMemberClique(primary)
			clique.Spec.Replicas = tc.initial
			clique.Spec.MinAvailable = ptr.To(tc.initial)
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
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: tc.maximum},
			}

			t.Log("resolve profile bounds from launch configuration rather than the live scale target")
			resolved, err := provider.Resolve(t.Context(), group)
			require.NoError(t, err)
			assert.Equal(t, "sglang", resolved.Profile.Backend)
			assert.Equal(t, int32(1), resolved.Profile.PodsPerReplica)
			assert.Equal(t, tc.initial, resolved.Profile.MinSupportedReplicas)
			assert.Equal(t, tc.maximum, resolved.Profile.MaxSupportedReplicas)
			_, groveCapacity := resolved.Capacity.(*grovecapacity.Adapter)
			assert.True(t, groveCapacity)
			assert.NotNil(t, resolved.Membership)
			assert.NotNil(t, resolved.Traffic)
			assert.NotNil(t, resolved.Verifier)
			assert.NotNil(t, resolved.Planner)
		})
	}
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
	t.Log("start from a committed two-allocation world")
	planner := sglangGrowthPlanner{profileFingerprint: "profile-v1", worldUID: "clique-uid"}
	status := domain.GroupStatus{Membership: domain.MembershipStatus{Observed: domain.MembershipObservation{
		CommittedTopology: domain.MembershipTopology{
			Generation: 1,
			Replicas: []domain.ReplicaMembership{
				{ReplicaID: "replica-0", Members: []domain.NativeMemberIncarnation{{ID: "dp-0", RuntimeIncarnation: "pod-0"}}},
				{ReplicaID: "replica-1", Members: []domain.NativeMemberIncarnation{{ID: "dp-1", RuntimeIncarnation: "pod-1"}}},
			},
		},
	}}}

	t.Log("reach a larger absolute target through one independently committed append per step")
	previousPlanID := ""
	for rank := 2; rank < 5; rank++ {
		resolution, err := planner.ResolveScalePlan(t.Context(), "group", 5, status)
		require.NoError(t, err)
		require.NotNil(t, resolution.Plan)
		require.Len(t, resolution.Plan.Change.Grow.Replicas, 1)
		joining := resolution.Plan.Change.Grow.Replicas[0]
		assert.Equal(t, domain.ReplicaID("replica-"+strconv.Itoa(rank)), joining.ReplicaID)
		assert.Equal(t, domain.CapacitySlotID("slot-"+strconv.Itoa(rank)), joining.SlotID)
		assert.Equal(t, []domain.NativeMemberID{domain.NativeMemberID("dp-" + strconv.Itoa(rank))}, joining.NativeMembers)
		assert.NotEqual(t, previousPlanID, resolution.Plan.ID)
		previousPlanID = resolution.Plan.ID

		// Retry the same committed base without changing the immutable step, even if the final target grows.
		t.Log("re-resolving the next append preserves its identity and exact payload")
		replayed, err := planner.ResolveScalePlan(t.Context(), "group", 6, status)
		require.NoError(t, err)
		assert.Equal(t, resolution.Plan, replayed.Plan)

		t.Log("observe the appended member as a new committed topology before planning the next step")
		base := &status.Membership.Observed.CommittedTopology
		base.Generation++
		base.Replicas = append(base.Replicas, domain.ReplicaMembership{
			ReplicaID: joining.ReplicaID,
			Members: []domain.NativeMemberIncarnation{{
				ID: joining.NativeMembers[0], RuntimeIncarnation: domain.RuntimeIncarnationID("pod-" + strconv.Itoa(rank)),
			}},
		})
	}

	t.Log("stop planning once committed membership reaches the absolute target")
	resolution, err := planner.ResolveScalePlan(t.Context(), "group", 5, status)
	require.NoError(t, err)
	assert.Nil(t, resolution.Plan)
	assert.Nil(t, resolution.Rejection)

	t.Log("reject shrink as a definitive backend capability boundary")
	resolution, err = planner.ResolveScalePlan(t.Context(), "group", 4, status)
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
				"--tp", "2", "--dp", "2", "--nnodes", "2",
				"--enable-dp-attention", "--enable-dp-lm-head",
				"--moe-a2a-backend", "nixl", "--elastic-ep-backend", "mooncake",
				"--load-balance-method", "round_robin",
				"--elastic-ep-initial-size", "2", "--max-ep-size", "3",
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
		Spec: grovev1alpha1.PodCliqueSpec{Replicas: 2, MinAvailable: ptr.To(int32(2)), PodSpec: *primary.Spec.DeepCopy()},
	}
}
