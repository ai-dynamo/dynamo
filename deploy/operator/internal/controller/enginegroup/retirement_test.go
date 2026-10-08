//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"errors"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestEngineGroupRetirementPlanContract(t *testing.T) {
	cases := []struct {
		name         string
		kind         domain.PlanKind
		replicas     []domain.ReplicaID
		fingerprint  string
		verify       domain.VerificationRequirement
		rejection    bool
		inconclusive bool
		wantError    string
	}{
		{name: "whole world", kind: domain.PlanKindRetire, replicas: []domain.ReplicaID{"replica-0", "replica-1"}},
		{name: "partial retirement", kind: domain.PlanKindRetire, replicas: []domain.ReplicaID{"replica-0"}, wantError: "every committed replica"},
		{name: "wrong replica", kind: domain.PlanKindRetire, replicas: []domain.ReplicaID{"replica-0", "other"}, wantError: "omits committed replica"},
		{name: "duplicate replica", kind: domain.PlanKindRetire, replicas: []domain.ReplicaID{"replica-0", "replica-0"}, wantError: "duplicate replicas"},
		{name: "new growth", kind: domain.PlanKindGrow, wantError: "every committed replica"},
		{name: "profile changed", kind: domain.PlanKindRetire, fingerprint: "another-profile", wantError: "profile-matched"},
		{name: "verification of an empty world", kind: domain.PlanKindRetire, verify: domain.VerificationRequirementRequired, wantError: "without serving verification"},
		{name: "unsupported backend", rejection: true},
		{name: "inconclusive backend", inconclusive: true, wantError: "planner unavailable"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("resolve deletion separately from the world's positive scaling bounds")
			backend := newEngineGroupControllerTestBackend(2)
			state, err := domain.NewGroupStatus(backend.topology, backend.capacity, backend.traffic)
			require.NoError(t, err)
			profile := "profile-v1"
			fingerprint := tc.fingerprint
			if fingerprint == "" {
				fingerprint = profile
			}
			verification := tc.verify
			if verification == "" {
				verification = domain.VerificationRequirementNone
			}
			planner := &engineGroupControllerTestPlanResolver{resolution: ScalePlanResolution{Plan: &domain.ResolvedPlan{
				ID: "retire-world", ProfileFingerprint: fingerprint,
				ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator,
				TrafficRequirement:    domain.TrafficRequirementQuiesceGroup, VerificationRequirement: verification,
				Change: domain.ResolvedChange{Kind: tc.kind, Retire: &domain.RetireChange{Replicas: tc.replicas}},
			}}}
			if tc.rejection {
				planner.resolution = ScalePlanResolution{Rejection: &domain.Failure{
					Classification: domain.FailureClassificationTerminal, Reason: "RetirementUnsupported",
				}}
			}
			if tc.inconclusive {
				planner.err = errors.New("planner unavailable")
			}

			t.Log("only a complete supported Retire plan may reach the coordinator")
			plan, rejection, err := resolveEngineGroupRetirement(t.Context(), "world", Runtime{
				Profile: api.EngineGroupProfileStatus{Fingerprint: profile, MinSupportedReplicas: 2}, Planner: planner,
			}, state)
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				assert.Nil(t, plan)
				assert.Nil(t, rejection)
			} else if tc.rejection {
				require.NoError(t, err)
				assert.Nil(t, plan)
				assert.Equal(t, planner.resolution.Rejection, rejection)
			} else {
				require.NoError(t, err)
				assert.Equal(t, planner.resolution.Plan, plan)
				assert.Nil(t, rejection)
			}
		})
	}
}

func TestEngineGroupDeletionDuringInitialization(t *testing.T) {
	t.Log("install the child-owned finalizer while a complete initial world is already running")
	backend := newEngineGroupControllerTestBackend(1)
	backend.ambiguousApply = false
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "early-retirement", Namespace: "test", UID: "early-world-uid"},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(group).WithStatusSubresource(group).Build()
	controller := &Reconciler{Client: kube, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
	request := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
	_, err := controller.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.NoError(t, kube.Get(t.Context(), request.NamespacedName, group))
	assert.Contains(t, group.Finalizers, engineGroupFinalizer)
	assert.Nil(t, group.Status.Topology)

	t.Log("delete before checkpoint initialization, capturing startup by observation before retirement")
	require.NoError(t, kube.Delete(t.Context(), group))
	_, err = controller.Reconcile(t.Context(), request)
	require.NoError(t, err)
	_, err = controller.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.NoError(t, kube.Get(t.Context(), request.NamespacedName, group))
	checkpoint := engineGroupControllerTestCheckpoint(t, t.Context(), kube, group)
	assert.Nil(t, checkpoint.State.Transition, "startup is not a synthetic membership operation")
	assert.Equal(t, int32(1), checkpoint.State.Membership.Observed.CommittedTopology.ReplicaCount())

	t.Log("retire the initial world through the normal checkpointed workflow, without changing its scale target")
	deleted := false
	for step := 0; step < 32; step++ {
		_, err = controller.Reconcile(t.Context(), request)
		require.NoError(t, err)
		err = kube.Get(t.Context(), request.NamespacedName, group)
		if apierrors.IsNotFound(err) {
			deleted = true
			break
		}
		require.NoError(t, err)
		assert.Equal(t, int32(1), group.Spec.Replicas)
	}
	require.True(t, deleted)
	assert.Equal(t, 1, backend.membershipApplyCount())
}

func TestEngineGroupDeletionOfObservationOnlyEmptyWorld(t *testing.T) {
	t.Log("capture an empty startup checkpoint before any workflow target or effect exists")
	backend := newEngineGroupControllerTestBackend(0)
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-startup", Namespace: "test", UID: "empty-startup-uid"},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(group).WithStatusSubresource(group).Build()
	controller := &Reconciler{Client: kube, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
	request := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
	for step := 0; step < 3; step++ {
		_, err := controller.Reconcile(t.Context(), request)
		require.NoError(t, err)
	}
	require.NoError(t, kube.Get(t.Context(), request.NamespacedName, group))
	state := engineGroupControllerTestCheckpoint(t, t.Context(), kube, group).State
	assert.Nil(t, state.Transition)
	assert.Nil(t, state.Capacity.Desired)
	assert.Nil(t, state.Traffic.Desired)

	t.Log("deleting an untouched empty world does not require a synthetic retirement transaction")
	require.NoError(t, kube.Delete(t.Context(), group))
	_, err := controller.Reconcile(t.Context(), request)
	require.NoError(t, err)
	assert.True(t, apierrors.IsNotFound(kube.Get(t.Context(), request.NamespacedName, group)))
	assert.Zero(t, backend.membershipApplyCount())
}

func TestEngineGroupUnsupportedRetirementPreservesWorld(t *testing.T) {
	t.Log("initialize a healthy world whose backend cannot implement terminal retirement")
	backend := newEngineGroupControllerTestBackend(1)
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "unsupported-retirement", Namespace: "test", UID: "held-world-uid"},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(group).WithStatusSubresource(group).Build()
	controller := &Reconciler{Client: kube, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
	request := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
	for step := 0; step < 3; step++ {
		_, err := controller.Reconcile(t.Context(), request)
		require.NoError(t, err)
	}
	require.NoError(t, kube.Get(t.Context(), request.NamespacedName, group))
	require.NoError(t, kube.Delete(t.Context(), group))
	runtime, err := controller.RuntimeProvider.Resolve(t.Context(), group)
	require.NoError(t, err)
	runtime.Planner = sglangGrowthPlanner{profileFingerprint: runtime.Profile.Fingerprint}

	t.Log("repeated deletion reconciliation reports the real growth-only limitation without releasing healthy capacity")
	for step := 0; step < 3; step++ {
		require.NoError(t, kube.Get(t.Context(), request.NamespacedName, group))
		result, err := controller.reconcileDeletion(t.Context(), group, runtime)
		require.NoError(t, err)
		assert.Positive(t, result.RequeueAfter)
	}
	require.NoError(t, kube.Get(t.Context(), request.NamespacedName, group))
	assert.Contains(t, group.Finalizers, engineGroupFinalizer)
	condition := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionProgressing)
	require.NotNil(t, condition)
	assert.Equal(t, "SGLangRetirementUnsupported", condition.Reason)
	assert.Len(t, backend.capacity.Allocations, 1)
	assert.Len(t, backend.traffic.Admitted, 1)
	assert.Equal(t, 0, backend.membershipApplyCount())
}

func TestEngineGroupDeletionWaitsForAcceptedGrowth(t *testing.T) {
	t.Log("request growth and retain its ambiguous but accepted membership transition")
	env := sharedEnv.ForTest(t)
	backend := newEngineGroupControllerTestBackend(1)
	provider := engineGroupControllerTestRuntimeProvider{backend: backend}
	controller := &Reconciler{Client: env.Client(), RuntimeProvider: provider}
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "retire-inflight-growth", Namespace: env.Namespace()},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	require.NoError(t, env.Client().Create(t.Context(), group))
	request := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
	var applyErr error
	for step := 0; step < 32; step++ {
		_, applyErr = controller.Reconcile(t.Context(), request)
		if applyErr != nil {
			break
		}
	}
	require.ErrorContains(t, applyErr, "ambiguous membership apply")
	require.NoError(t, env.Client().Get(t.Context(), request.NamespacedName, group))
	operationID := group.Status.Operation.ID
	require.NoError(t, env.Client().Delete(t.Context(), group))

	t.Log("deletion and a controller restart cannot submit a competing collective")
	controller = &Reconciler{Client: env.Client(), RuntimeProvider: provider}
	for step := 0; step < 3; step++ {
		_, err := controller.Reconcile(t.Context(), request)
		require.NoError(t, err)
	}
	require.NoError(t, env.Client().Get(t.Context(), request.NamespacedName, group))
	assert.Equal(t, operationID, group.Status.Operation.ID)
	assert.Equal(t, 1, backend.membershipApplyCount())
	assert.Len(t, backend.capacity.Allocations, 2)

	t.Log("observe growth completion, then retire the resulting exact world with a new operation")
	backend.commitPendingMembership()
	deleted := false
	for step := 0; step < 64; step++ {
		_, err := controller.Reconcile(t.Context(), request)
		require.NoError(t, err)
		err = env.Client().Get(t.Context(), request.NamespacedName, group)
		if apierrors.IsNotFound(err) {
			deleted = true
			break
		}
		require.NoError(t, err)
	}
	require.True(t, deleted)
	assert.Equal(t, 2, backend.membershipApplyCount())
	assert.Len(t, backend.capacity.ReleaseFences, 2)
	assert.Empty(t, backend.capacity.Allocations)
	assert.Empty(t, backend.topology.Replicas)
}
