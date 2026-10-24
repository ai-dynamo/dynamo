//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestVLLMGroveNativeBootstrapUsesRealScaleAndPodUIDs(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := t.Context()

	t.Log("declare a native TP1/DP2 world with capacity reserved for DP3")
	source := dynamo.VLLMProfileGeometrySource{
		Command: []string{"vllm", "serve", "model"},
		Args: []string{
			"--elastic-ep-auto-bootstrap", "--elastic-ep-initial-dp-size=2", "--data-parallel-size-local=1", "--data-parallel-external-lb",
			"--data-parallel-rank=$(DYNAMO_RANK)", "--data-parallel-address=$(DYNAMO_LEADER_ADDRESS)",
			"--tensor-parallel-size=1", "--distributed-executor-backend=mp", "--elastic-ep-max-dp-size=3",
			"--enable-expert-parallel", "--enable-eplb", "--enable-elastic-ep",
		},
		InitialReplicas: 2, MainContainerGPUs: 1, DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest: "test-image-model",
	}
	profile, err := dynamo.ResolveVLLMElasticEPProfile(source)
	require.NoError(t, err)
	dgd := &api.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "deployment", Namespace: env.Namespace(), UID: "dgd-uid"}}
	component := &api.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2}}
	group := dynamo.RenderVLLMEngineGroup(dgd, component, 0, profile)
	group.Finalizers = []string{engineGroupFinalizer}
	require.NoError(t, env.Client().Create(ctx, group))
	template := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
		RestartPolicy:  corev1.RestartPolicyNever,
		ReadinessGates: []corev1.PodReadinessGate{{ConditionType: corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)}},
		Containers:     []corev1.Container{{Name: "main", Image: "test-vllm:latest", Command: source.Command, Args: source.Args}},
	}}
	clique := &grove.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: "vllm-members", Namespace: env.Namespace(),
			Labels: map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grove.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup", Name: "world", UID: "world-uid", Controller: ptr.To(true),
			}},
		},
		Spec: grove.PodCliqueSpec{Replicas: 2, MinAvailable: ptr.To(int32(2)), PodSpec: template.Spec},
	}
	require.NoError(t, env.Client().Create(ctx, clique))
	capacity := &VLLMGroveCapacity{
		Grove: &grovecapacity.Adapter{
			Client: env.Client(), Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID, GroupName: group.Name,
			Journal: kubejournal.NewStore(env.Client(), group.Namespace, group.Name, group.UID, "grove-capacity"),
		},
		Group: group, Source: source,
	}
	pods := make([]*corev1.Pod, 2)
	for rank := range int32(2) {
		pod := newVLLMGroveLaunchPod(clique, rank)
		require.NoError(t, env.Client().Create(ctx, pod))
		pods[rank] = pod
	}

	t.Log("record exact initial slot occupants without publishing commands or waiting for readiness")
	require.NoError(t, capacity.ReconcileAllocations(ctx))
	observed, err := capacity.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	require.Len(t, observed.Allocations, 2)
	for _, allocation := range observed.Allocations {
		assert.False(t, allocation.BootstrapPending, "native startup does not consume a controller receipt")
		assert.False(t, allocation.Available)
	}
	bindings, err := capacity.capacityBindings()
	require.NoError(t, err)
	assert.Equal(t, domain.PodUID(pods[0].UID), bindings.Slots["slot-0"])
	assert.Equal(t, domain.PodUID(pods[1].UID), bindings.Slots["slot-1"])

	t.Log("observe initial membership and serving independently of allocation identity")
	for _, pod := range pods {
		pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
		require.NoError(t, env.Client().Status().Update(ctx, pod))
	}
	observed, err = capacity.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	backend := newEngineGroupControllerTestBackend(0)
	backend.ambiguousApply = false
	for _, allocation := range observed.Allocations {
		member := domain.ReplicaMembership{ReplicaID: allocation.Incarnation.ReplicaID, Members: allocation.Incarnation.Members}
		backend.topology.Replicas = append(backend.topology.Replicas, member)
		backend.traffic.Admitted = append(backend.traffic.Admitted, member)
	}
	state, err := domain.NewGroupStatus(backend.topology, observed, backend.traffic)
	require.NoError(t, err)
	membership := engineGroupControllerTestMembershipAdapter{backend: backend}
	traffic := engineGroupControllerTestTrafficAdapter{backend: backend}
	plan := domain.ResolvedPlan{
		ID: "grow-three", ProfileFingerprint: profile.Geometry.Fingerprint,
		ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:    domain.TrafficRequirementKeepServing, VerificationRequirement: domain.VerificationRequirementRequired,
		Change: domain.ResolvedChange{Kind: domain.PlanKindGrow, Grow: &domain.GrowChange{Replicas: []domain.ReplicaTarget{{
			ReplicaID: "replica-2", SlotID: "slot-2", Bootstrap: domain.BootstrapModeJoin, NativeMembers: []domain.NativeMemberID{"dp-2"},
		}}}},
	}

	t.Log("scale the real clique endpoint without changing any Pod's native template")
	for range 30 {
		result, err := domain.NewCoordinator(capacity, membership, traffic, backend).Reconcile(ctx, domain.GroupID(group.UID), &plan, state)
		require.NoError(t, err)
		state = result.Status
		require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(clique), clique))
		if clique.Spec.Replicas == 3 {
			break
		}
	}
	require.Equal(t, int32(3), clique.Spec.Replicas)
	assert.Equal(t, template.Spec, clique.Spec.PodSpec)
	joiner := newVLLMGroveLaunchPod(clique, 2)
	require.NoError(t, env.Client().Create(ctx, joiner))
	require.NoError(t, capacity.ReconcileAllocations(ctx))

	t.Log("the pending joiner cannot deadlock membership preparation behind inference readiness")
	for range 30 {
		result, err := domain.NewCoordinator(capacity, membership, traffic, backend).Reconcile(ctx, domain.GroupID(group.UID), &plan, state)
		require.NoError(t, err)
		state = result.Status
		if backend.membershipApplyCount() > 0 {
			break
		}
	}
	require.Equal(t, 1, backend.membershipApplyCount())
	record, found := state.Registry.Find("replica-2")
	require.True(t, found)
	require.NotNil(t, record.Current)
	assert.Equal(t, domain.PodUID(joiner.UID), record.Current.CapacityRefs[0].UID)
	observed, err = capacity.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	assert.False(t, observed.Allocations[2].Available)
	assert.False(t, observed.Allocations[2].BootstrapPending)

	t.Log("a replacement cannot inherit a retired or vanished Pod's slot binding")
	joiner.Finalizers = nil
	require.NoError(t, env.Client().Update(ctx, joiner))
	require.NoError(t, env.Client().Delete(ctx, joiner, client.GracePeriodSeconds(0)))
	replacement := newVLLMGroveLaunchPod(clique, 2)
	require.NoError(t, env.Client().Create(ctx, replacement))
	require.NotEqual(t, joiner.UID, replacement.UID)
	_, err = capacity.Observe(ctx, domain.GroupID(group.UID))
	require.ErrorIs(t, err, domain.ErrRecoveryRequired)
	require.ErrorIs(t, capacity.ReconcileAllocations(ctx), domain.ErrRecoveryRequired)

	t.Log("loss of the binding catalog cannot be mistaken for first startup")
	before := group.DeepCopy()
	delete(group.Annotations, consts.KubeAnnotationDynamoVLLMCapacityBindings)
	require.NoError(t, env.Client().Patch(ctx, group, client.MergeFrom(before)))
	require.ErrorIs(t, capacity.ReconcileAllocations(ctx), domain.ErrRecoveryRequired)
}

func newVLLMGroveLaunchPod(clique *grove.PodClique, rank int32) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("allocation-%d", rank), Namespace: clique.Namespace,
			Finalizers: []string{engineGroupProcessFinalizer},
			Labels: map[string]string{
				grovecommon.LabelPodClique: clique.Name, grovecommon.LabelPodCliquePodIndex: fmt.Sprint(rank),
				consts.KubeLabelDynamoEngineGroup:         clique.Labels[consts.KubeLabelDynamoEngineGroup],
				consts.KubeLabelDynamoScaleRepresentative: consts.KubeLabelDynamoScaleRepresentativeYes,
			},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(clique, grove.SchemeGroupVersion.WithKind("PodClique"))},
		},
		Spec: *clique.Spec.PodSpec.DeepCopy(),
	}
}
