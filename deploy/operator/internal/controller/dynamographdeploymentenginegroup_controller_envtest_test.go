//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestEngineGroupControllerResumesAmbiguousGrowthAfterRestart(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := context.Background()
	backend := newEngineGroupControllerTestBackend(2)
	provider := engineGroupControllerTestRuntimeProvider{backend: backend}
	reconciler := &DynamoGraphDeploymentEngineGroupReconciler{
		Client:          env.Client(),
		RuntimeProvider: provider,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "restart-growth", Namespace: env.Namespace()},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 3},
	}

	t.Log("create one Engine Group whose desired target exceeds its observed two-member world")
	require.NoError(t, env.Client().Create(ctx, group))
	t.Cleanup(func() {
		backend.clear()
		_ = env.Client().Delete(ctx, group)
	})
	key := types.NamespacedName{Name: group.Name, Namespace: group.Namespace}

	t.Log("reconcile until membership apply returns an intentionally ambiguous transport error")
	var ambiguousErr error
	for step := 0; step < 32; step++ {
		_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		if err != nil {
			ambiguousErr = err
			break
		}
	}
	require.ErrorContains(t, ambiguousErr, "ambiguous membership apply")
	assert.Equal(t, 1, backend.membershipApplyCount())

	t.Log("verify the exact desired membership transition was durable before the ambiguous call")
	stored := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{}
	require.NoError(t, env.Client().Get(ctx, key, stored))
	require.NotNil(t, stored.Status.Reconciliation)
	require.NotNil(t, stored.Status.Reconciliation.Membership.Desired)
	desiredTransitionID := stored.Status.Reconciliation.Membership.Desired.TransitionID
	desiredDigest := stored.Status.Reconciliation.Membership.Desired.TargetDigest
	assert.NotEmpty(t, desiredTransitionID)
	assert.NotEmpty(t, desiredDigest)

	t.Log("let the adapter publish the correlated commit and replace the reconciler instance")
	backend.commitPendingMembership()
	plannerErr := errors.New("planner temporarily unavailable")
	backend.setPlannerError(plannerErr)
	reconciler = &DynamoGraphDeploymentEngineGroupReconciler{
		Client:          env.Client(),
		RuntimeProvider: provider,
	}

	t.Log("resume entirely from Kubernetes status until capacity, membership, verification, and traffic converge")
	var lastReconcileErr error
	var lastObserved *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup
	completed := false
	for step := 0; step < 64; step++ {
		_, lastReconcileErr = reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		current := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{}
		require.NoError(t, env.Client().Get(ctx, key, current))
		lastObserved = current
		if current.Status.Reconciliation != nil && current.Status.Reconciliation.Transition != nil &&
			current.Status.Reconciliation.Transition.Outcome ==
				nvidiacomv1beta1.EngineGroupTransitionOutcomeCompleted {
			completed = true
			break
		}
	}
	require.True(t, completed, "last reconcile error: %v; last reconciliation: %#v", lastReconcileErr,
		lastObserved.Status.Reconciliation)
	require.ErrorIs(t, lastReconcileErr, plannerErr)

	t.Log("verify restart recovery reused one transition and exposed the converged scale/status views")
	require.NoError(t, env.Client().Get(ctx, key, stored))
	assert.Equal(t, int32(3), stored.Status.Replicas)
	assert.Equal(t, int32(3), stored.Status.AvailableReplicas)
	assert.Equal(t, int32(3), stored.Status.ActiveReplicas)
	assert.Equal(t, desiredTransitionID, stored.Status.Reconciliation.Membership.Desired.TransitionID)
	assert.Equal(t, desiredDigest, stored.Status.Reconciliation.Membership.Desired.TargetDigest)
	assert.Equal(t, 1, backend.membershipApplyCount())
	targetValid := meta.FindStatusCondition(stored.Status.Conditions, engineGroupConditionTargetValid)
	require.NotNil(t, targetValid)
	assert.Equal(t, metav1.ConditionUnknown, targetValid.Status)

	t.Log("read the real scale subresource after controller convergence")
	scale := &autoscalingv1.Scale{}
	require.NoError(t, env.Client().SubResource("scale").Get(ctx, stored, scale))
	assert.Equal(t, int32(3), scale.Spec.Replicas)
	assert.Equal(t, int32(3), scale.Status.Replicas)
}

func TestEngineGroupControllerBlocksDeletionUntilWorldIsEmpty(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := context.Background()
	backend := newEngineGroupControllerTestBackend(1)
	backend.ambiguousApply = false
	provider := engineGroupControllerTestRuntimeProvider{backend: backend}
	reconciler := &DynamoGraphDeploymentEngineGroupReconciler{
		Client:          env.Client(),
		RuntimeProvider: provider,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "deletion-guard", Namespace: env.Namespace()},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	key := types.NamespacedName{Name: group.Name, Namespace: group.Namespace}

	t.Log("create and initialize a healthy one-member Engine Group")
	require.NoError(t, env.Client().Create(ctx, group))
	for step := 0; step < 4; step++ {
		_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		require.NoError(t, err)
	}
	require.NoError(t, env.Client().Get(ctx, key, group))
	assert.Contains(t, group.Finalizers, engineGroupFinalizer)

	t.Log("request deletion while authoritative capacity and membership remain non-empty")
	require.NoError(t, env.Client().Delete(ctx, group))
	_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
	require.NoError(t, err)
	require.NoError(t, env.Client().Get(ctx, key, group))
	assert.Contains(t, group.Finalizers, engineGroupFinalizer)

	t.Log("publish terminal empty accepted targets and make every external authority converge to them")
	require.NoError(t, env.Client().Get(ctx, key, group))
	prepareEngineGroupControllerTerminalDeletion(t, env.Client(), ctx, group, backend)
	_, err = reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
	require.NoError(t, err)

	t.Log("verify the controller releases its finalizer only after both authorities report empty")
	err = env.Client().Get(ctx, key, group)
	assert.True(t, apierrors.IsNotFound(err), "expected deleted Engine Group, got %v", err)
}

func TestEngineGroupControllerDeletesEmptyWorldBeforeJournalInitialization(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := context.Background()
	backend := newEngineGroupControllerTestBackend(0)
	backend.ambiguousApply = false
	reconciler := &DynamoGraphDeploymentEngineGroupReconciler{
		Client: env.Client(),
		RuntimeProvider: engineGroupControllerTestRuntimeProvider{
			backend: backend,
		},
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "delete-before-initialization", Namespace: env.Namespace()},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	key := types.NamespacedName{Name: group.Name, Namespace: group.Namespace}

	t.Log("reconcile exactly once so the runtime-owned finalizer is installed before journal initialization")
	require.NoError(t, env.Client().Create(ctx, group))
	_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
	require.NoError(t, err)
	require.NoError(t, env.Client().Get(ctx, key, group))
	assert.Contains(t, group.Finalizers, engineGroupFinalizer)
	assert.Nil(t, group.Status.Reconciliation)

	t.Log("delete while every fresh external authority reports the untouched world is empty")
	require.NoError(t, env.Client().Delete(ctx, group))
	_, err = reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
	require.NoError(t, err)

	t.Log("verify the pre-effect deletion race cannot strand the finalizer")
	err = env.Client().Get(ctx, key, group)
	assert.True(t, apierrors.IsNotFound(err), "expected deleted Engine Group, got %v", err)
}

func TestEngineGroupControllerDoesNotTrapUnownedResources(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := context.Background()
	reconciler := &DynamoGraphDeploymentEngineGroupReconciler{
		Client:          env.Client(),
		RuntimeProvider: unavailableEngineGroupRuntimeProvider{},
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "runtime-unavailable", Namespace: env.Namespace()},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	key := types.NamespacedName{Name: group.Name, Namespace: group.Namespace}

	t.Log("create an Engine Group before a production runtime provider can own its effects")
	require.NoError(t, env.Client().Create(ctx, group))
	_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
	require.NoError(t, err)
	require.NoError(t, env.Client().Get(ctx, key, group))

	t.Log("verify the controller reports the missing runtime without installing an unserviceable finalizer")
	assert.NotContains(t, group.Finalizers, engineGroupFinalizer)
	runtimeReady := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionRuntimeReady)
	require.NotNil(t, runtimeReady)
	assert.Equal(t, metav1.ConditionFalse, runtimeReady.Status)

	t.Log("delete the unowned resource without requiring an unavailable external adapter")
	require.NoError(t, env.Client().Delete(ctx, group))
	require.Eventually(t, func() bool {
		err := env.Client().Get(ctx, key, group)
		return apierrors.IsNotFound(err)
	}, 5*time.Second, 50*time.Millisecond)
}

type engineGroupControllerTestRuntimeProvider struct {
	backend *engineGroupControllerTestBackend
}

func (p engineGroupControllerTestRuntimeProvider) Resolve(
	context.Context,
	*nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (EngineGroupRuntime, error) {
	return EngineGroupRuntime{
		Profile: nvidiacomv1beta1.EngineGroupProfileStatus{
			Backend:                "sglang",
			Fingerprint:            "profile-v1",
			GPUsPerReplica:         1,
			PodsPerReplica:         1,
			MinSafeServingReplicas: 1,
			MinReplicas:            1,
			MaxReplicas:            8,
		},
		Capacity:   engineGroupControllerTestCapacityAdapter(p),
		Membership: engineGroupControllerTestMembershipAdapter(p),
		Traffic:    engineGroupControllerTestTrafficAdapter(p),
		Verifier:   p.backend,
		Planner:    p.backend,
	}, nil
}

type engineGroupControllerTestBackend struct {
	mu                         sync.Mutex
	capacity                   enginegroup.CapacityObservation
	traffic                    enginegroup.TrafficObservation
	topology                   enginegroup.MembershipTopology
	transition                 *enginegroup.MembershipTransitionObservation
	pendingTarget              *enginegroup.MembershipTarget
	ambiguousApply             bool
	plannerErr                 error
	membershipApplyInvocations int
}

func newEngineGroupControllerTestBackend(replicas int) *engineGroupControllerTestBackend {
	backend := &engineGroupControllerTestBackend{ambiguousApply: true}
	backend.topology = enginegroup.MembershipTopology{Generation: 1}
	for index := 0; index < replicas; index++ {
		replicaID := enginegroup.ReplicaID(fmt.Sprintf("replica-%d", index))
		slotID := enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", index))
		runtimeID := enginegroup.RuntimeIncarnationID(fmt.Sprintf("runtime-%d", index))
		membership := enginegroup.ReplicaMembership{
			ReplicaID:          replicaID,
			RuntimeIncarnation: runtimeID,
			NativeMembers:      []enginegroup.NativeMemberID{enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", index))},
		}
		backend.topology.Replicas = append(backend.topology.Replicas, membership)
		backend.traffic.Admitted = append(backend.traffic.Admitted, membership)
		backend.capacity.Allocations = append(backend.capacity.Allocations, enginegroup.CapacityAllocation{
			Incarnation: enginegroup.ReplicaIncarnation{
				ReplicaID:          replicaID,
				SlotID:             slotID,
				RuntimeIncarnation: runtimeID,
				CapacityRefs: []enginegroup.CapacityRef{{
					Name: fmt.Sprintf("worker-%d", index),
					UID:  enginegroup.PodUID(fmt.Sprintf("uid-%d", index)),
				}},
			},
			Available: true,
		})
	}
	return backend
}

func (b *engineGroupControllerTestBackend) ResolveScalePlan(
	_ context.Context,
	_ enginegroup.GroupID,
	targetReplicas int32,
	status enginegroup.GroupStatus,
) (ScalePlanResolution, error) {
	b.mu.Lock()
	plannerErr := b.plannerErr
	b.mu.Unlock()
	if plannerErr != nil {
		return ScalePlanResolution{}, plannerErr
	}
	current, found := status.Topologies.Current()
	if !found {
		return ScalePlanResolution{}, errors.New("current topology is absent")
	}
	if targetReplicas == current.ReplicaCount() {
		return ScalePlanResolution{}, nil
	}
	plan := &enginegroup.ResolvedPlan{
		ID:                      fmt.Sprintf("scale-%d-%d", current.Generation, targetReplicas),
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   enginegroup.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      enginegroup.TrafficRequirementKeepServing,
		VerificationRequirement: enginegroup.VerificationRequirementRequired,
	}
	if targetReplicas > current.ReplicaCount() {
		plan.Change = enginegroup.ResolvedChange{Kind: enginegroup.PlanKindGrow, Grow: &enginegroup.GrowChange{}}
		for index := current.ReplicaCount(); index < targetReplicas; index++ {
			plan.Change.Grow.Replicas = append(plan.Change.Grow.Replicas, enginegroup.ReplicaTarget{
				ReplicaID:     enginegroup.ReplicaID(fmt.Sprintf("replica-%d", index)),
				SlotID:        enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", index)),
				Bootstrap:     enginegroup.BootstrapModeJoin,
				NativeMembers: []enginegroup.NativeMemberID{enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", index))},
			})
		}
		return ScalePlanResolution{Plan: plan}, nil
	}

	plan.Change = enginegroup.ResolvedChange{Kind: enginegroup.PlanKindRetire, Retire: &enginegroup.RetireChange{}}
	for index := current.ReplicaCount() - 1; index >= targetReplicas; index-- {
		plan.Change.Retire.Replicas = append(plan.Change.Retire.Replicas,
			enginegroup.ReplicaID(fmt.Sprintf("replica-%d", index)))
	}
	return ScalePlanResolution{Plan: plan}, nil
}

type engineGroupControllerTestCapacityAdapter struct {
	backend *engineGroupControllerTestBackend
}

func (a engineGroupControllerTestCapacityAdapter) Observe(
	context.Context,
	enginegroup.GroupID,
) (enginegroup.CapacityObservation, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	return cloneEngineGroupControllerTestCapacity(a.backend.capacity), nil
}

func (a engineGroupControllerTestCapacityAdapter) Apply(
	_ context.Context,
	_ enginegroup.GroupID,
	target enginegroup.CapacityTarget,
) (enginegroup.ApplyResult, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	existing := make(map[enginegroup.ReplicaID]enginegroup.CapacityAllocation, len(a.backend.capacity.Allocations))
	for _, allocation := range a.backend.capacity.Allocations {
		existing[allocation.Incarnation.ReplicaID] = allocation
	}
	allocations := make([]enginegroup.CapacityAllocation, 0, len(target.Replicas))
	for _, replica := range target.Replicas {
		if allocation, found := existing[replica.ReplicaID]; found {
			allocations = append(allocations, allocation)
			continue
		}
		if replica.Bootstrap == nil {
			continue
		}
		index := len(existing)
		allocations = append(allocations, enginegroup.CapacityAllocation{
			Incarnation: enginegroup.ReplicaIncarnation{
				ReplicaID:          replica.ReplicaID,
				SlotID:             replica.SlotID,
				RuntimeIncarnation: enginegroup.RuntimeIncarnationID(fmt.Sprintf("runtime-%d", index)),
				CapacityRefs: []enginegroup.CapacityRef{{
					Name: fmt.Sprintf("worker-%d", index),
					UID:  enginegroup.PodUID(fmt.Sprintf("uid-%d", index)),
				}},
			},
			Available: true,
		})
	}
	for _, fence := range target.ReleaseFences {
		for index := len(allocations) - 1; index >= 0; index-- {
			if allocations[index].Incarnation.ReplicaID == fence.ReplicaID {
				allocations = append(allocations[:index], allocations[index+1:]...)
			}
		}
	}
	a.backend.capacity = enginegroup.CapacityObservation{
		AppliedRevision: target.ControlRevision,
		Allocations:     allocations,
		ReleaseFences:   append([]enginegroup.ReleaseFence(nil), target.ReleaseFences...),
	}
	return enginegroup.ApplyResult{}, nil
}

type engineGroupControllerTestTrafficAdapter struct {
	backend *engineGroupControllerTestBackend
}

func (a engineGroupControllerTestTrafficAdapter) Observe(
	context.Context,
	enginegroup.GroupID,
) (enginegroup.TrafficObservation, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	return cloneEngineGroupControllerTestTraffic(a.backend.traffic), nil
}

func (a engineGroupControllerTestTrafficAdapter) Apply(
	_ context.Context,
	_ enginegroup.GroupID,
	target enginegroup.TrafficTarget,
) (enginegroup.ApplyResult, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	drained := make([]enginegroup.ReplicaMembership, 0, len(target.Drain))
	for _, target := range target.Drain {
		drained = append(drained, cloneEngineGroupControllerTestMembership(target.Membership))
	}
	a.backend.traffic = enginegroup.TrafficObservation{
		AppliedRevision: target.ControlRevision,
		Admitted:        cloneEngineGroupControllerTestMemberships(target.Admitted),
		Drained:         drained,
	}
	return enginegroup.ApplyResult{}, nil
}

type engineGroupControllerTestMembershipAdapter struct {
	backend *engineGroupControllerTestBackend
}

func (a engineGroupControllerTestMembershipAdapter) ValidatePlan(
	_ context.Context,
	_ enginegroup.GroupID,
	request enginegroup.PlanValidationRequest,
) (enginegroup.PreflightResult, error) {
	return enginegroup.PreflightResult{Evidence: &enginegroup.ValidationEvidence{
		PlanDigest:           request.PlanDigest,
		ProfileFingerprint:   request.Plan.ProfileFingerprint,
		CapabilityGeneration: "test-v1",
	}}, nil
}

func (a engineGroupControllerTestMembershipAdapter) ValidateTarget(
	_ context.Context,
	_ enginegroup.GroupID,
	target enginegroup.MembershipTarget,
) (enginegroup.PreflightResult, error) {
	return enginegroup.PreflightResult{Evidence: &enginegroup.ValidationEvidence{
		PlanDigest:           target.Validation.PlanDigest,
		TargetDigest:         target.TargetDigest,
		ProfileFingerprint:   target.Plan.ProfileFingerprint,
		CapabilityGeneration: "test-v1",
	}}, nil
}

func (a engineGroupControllerTestMembershipAdapter) Observe(
	_ context.Context,
	_ enginegroup.GroupID,
	transitionID string,
) (enginegroup.MembershipObservation, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	observation := enginegroup.MembershipObservation{
		CommittedTopology:     cloneEngineGroupControllerTestTopology(a.backend.topology),
		RequestedTransitionID: transitionID,
	}
	if transitionID != "" && a.backend.transition != nil && a.backend.transition.TransitionID == transitionID {
		transition := *a.backend.transition
		if transition.ResultTopology != nil {
			result := cloneEngineGroupControllerTestTopology(*transition.ResultTopology)
			transition.ResultTopology = &result
		}
		observation.Transition = &transition
	}
	return observation, nil
}

func (a engineGroupControllerTestMembershipAdapter) Apply(
	_ context.Context,
	_ enginegroup.GroupID,
	target enginegroup.MembershipTarget,
) error {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	if a.backend.transition != nil && a.backend.transition.TransitionID == target.TransitionID &&
		a.backend.transition.Phase == enginegroup.MembershipTransitionPhaseCommitted {
		return nil
	}
	a.backend.membershipApplyInvocations++
	targetCopy := target
	a.backend.pendingTarget = &targetCopy
	a.backend.transition = &enginegroup.MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           enginegroup.MembershipTransitionPhasePending,
	}
	if a.backend.ambiguousApply {
		a.backend.ambiguousApply = false
		return errors.New("ambiguous membership apply")
	}
	a.backend.commitMembershipLocked(target)
	return nil
}

func (b *engineGroupControllerTestBackend) Verify(
	_ context.Context,
	_ enginegroup.GroupID,
	topology enginegroup.MembershipTopology,
) (enginegroup.VerificationResult, error) {
	return enginegroup.VerificationResult{Proof: &enginegroup.ServingProof{
		TopologyGeneration: topology.Generation,
		RuntimeDigest:      enginegroup.TopologyRuntimeDigest(topology),
		ObservedAt:         time.Date(2026, time.September, 15, 12, 0, 0, 0, time.UTC),
	}}, nil
}

func (b *engineGroupControllerTestBackend) membershipApplyCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.membershipApplyInvocations
}

func (b *engineGroupControllerTestBackend) commitPendingMembership() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.pendingTarget == nil {
		return
	}
	b.commitMembershipLocked(*b.pendingTarget)
}

func (b *engineGroupControllerTestBackend) setPlannerError(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.plannerErr = err
}

func (b *engineGroupControllerTestBackend) clear() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.capacity.Allocations = nil
	b.capacity.ReleaseFences = nil
	b.traffic.Admitted = nil
	b.traffic.Draining = nil
	b.traffic.Drained = nil
	b.topology.Replicas = nil
	b.topology.Generation++
	if b.topology.Generation < 1 {
		b.topology.Generation = 1
	}
}

func (b *engineGroupControllerTestBackend) convergeTerminalDeletion(
	capacityRevision int64,
	trafficRevision int64,
) {
	b.mu.Lock()
	defer b.mu.Unlock()
	drained := cloneEngineGroupControllerTestMemberships(b.topology.Replicas)
	b.capacity = enginegroup.CapacityObservation{AppliedRevision: capacityRevision}
	b.traffic = enginegroup.TrafficObservation{
		AppliedRevision: trafficRevision,
		Drained:         drained,
	}
	b.topology = enginegroup.MembershipTopology{Generation: b.topology.Generation + 1}
}

func prepareEngineGroupControllerTerminalDeletion(
	t *testing.T,
	kubeClient client.Client,
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	backend *engineGroupControllerTestBackend,
) {
	t.Helper()
	require.NotNil(t, group.Status.Reconciliation)
	status := engineGroupStatusFromAPI(group.Status.Reconciliation)
	base := status.Membership.Observed.CommittedTopology
	status.ControlRevision++
	capacityRevision := status.ControlRevision
	status.Capacity.Desired = &enginegroup.CapacityTarget{
		ControlRevision:       capacityRevision,
		TransitionID:          "terminal-retirement",
		ProfileFingerprint:    "profile-v1",
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
	}
	status.Capacity.Accepted = engineGroupControllerCloneCapacityTarget(status.Capacity.Desired)
	status.Capacity.Observed = enginegroup.CapacityObservation{AppliedRevision: capacityRevision}
	status.ControlRevision++
	trafficRevision := status.ControlRevision
	status.Traffic.Desired = &enginegroup.TrafficTarget{
		ControlRevision:    trafficRevision,
		TransitionID:       "terminal-retirement",
		TopologyGeneration: base.Generation,
	}
	for _, member := range base.Replicas {
		status.Traffic.Desired.Drain = append(status.Traffic.Desired.Drain, enginegroup.TrafficDrainTarget{
			Membership: member,
			Mode:       enginegroup.TrafficDrainModeGraceful,
		})
	}
	status.Traffic.Accepted = engineGroupControllerCloneTrafficTarget(status.Traffic.Desired)
	status.Traffic.Observed = enginegroup.TrafficObservation{
		AppliedRevision: trafficRevision,
		Drained:         cloneEngineGroupControllerTestMemberships(base.Replicas),
	}
	group.Status.Reconciliation = engineGroupStatusToAPI(status)
	require.NoError(t, kubeClient.Status().Update(ctx, group))
	backend.convergeTerminalDeletion(capacityRevision, trafficRevision)
}

func engineGroupControllerCloneCapacityTarget(target *enginegroup.CapacityTarget) *enginegroup.CapacityTarget {
	if target == nil {
		return nil
	}
	clone := *target
	clone.Replicas = append([]enginegroup.CapacityReplicaTarget(nil), target.Replicas...)
	clone.ReleaseFences = append([]enginegroup.ReleaseFence(nil), target.ReleaseFences...)
	return &clone
}

func engineGroupControllerCloneTrafficTarget(target *enginegroup.TrafficTarget) *enginegroup.TrafficTarget {
	if target == nil {
		return nil
	}
	clone := *target
	clone.Admitted = cloneEngineGroupControllerTestMemberships(target.Admitted)
	clone.Drain = append([]enginegroup.TrafficDrainTarget(nil), target.Drain...)
	for index := range clone.Drain {
		clone.Drain[index].Membership = cloneEngineGroupControllerTestMembership(clone.Drain[index].Membership)
	}
	return &clone
}

func (b *engineGroupControllerTestBackend) commitMembershipLocked(target enginegroup.MembershipTarget) {
	result := target.BaseTopology
	result.Generation++
	switch target.Plan.Change.Kind {
	case enginegroup.PlanKindGrow:
		for _, joining := range target.Joining {
			for _, planned := range target.Plan.Change.Grow.Replicas {
				if joining.ReplicaID == planned.ReplicaID {
					result.Replicas = append(result.Replicas, enginegroup.ReplicaMembership{
						ReplicaID:          joining.ReplicaID,
						RuntimeIncarnation: joining.RuntimeIncarnation,
						NativeMembers:      append([]enginegroup.NativeMemberID(nil), planned.NativeMembers...),
					})
				}
			}
		}
	case enginegroup.PlanKindRetire:
		retiring := make(map[enginegroup.ReplicaID]struct{}, len(target.Plan.Change.Retire.Replicas))
		for _, replicaID := range target.Plan.Change.Retire.Replicas {
			retiring[replicaID] = struct{}{}
		}
		result.Replicas = result.Replicas[:0]
		for _, member := range target.BaseTopology.Replicas {
			if _, found := retiring[member.ReplicaID]; !found {
				result.Replicas = append(result.Replicas, member)
			}
		}
	}
	b.topology = result
	b.transition = &enginegroup.MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           enginegroup.MembershipTransitionPhaseCommitted,
		ResultTopology:  &result,
	}
	b.pendingTarget = nil
}

func cloneEngineGroupControllerTestCapacity(
	value enginegroup.CapacityObservation,
) enginegroup.CapacityObservation {
	value.Allocations = append([]enginegroup.CapacityAllocation(nil), value.Allocations...)
	for index := range value.Allocations {
		value.Allocations[index].Incarnation.CapacityRefs = append(
			[]enginegroup.CapacityRef(nil),
			value.Allocations[index].Incarnation.CapacityRefs...,
		)
	}
	value.ReleaseFences = append([]enginegroup.ReleaseFence(nil), value.ReleaseFences...)
	return value
}

func cloneEngineGroupControllerTestTraffic(value enginegroup.TrafficObservation) enginegroup.TrafficObservation {
	value.Admitted = cloneEngineGroupControllerTestMemberships(value.Admitted)
	value.Draining = cloneEngineGroupControllerTestMemberships(value.Draining)
	value.Drained = cloneEngineGroupControllerTestMemberships(value.Drained)
	return value
}

func cloneEngineGroupControllerTestTopology(value enginegroup.MembershipTopology) enginegroup.MembershipTopology {
	value.Replicas = cloneEngineGroupControllerTestMemberships(value.Replicas)
	return value
}

func cloneEngineGroupControllerTestMemberships(
	values []enginegroup.ReplicaMembership,
) []enginegroup.ReplicaMembership {
	replicas := make([]enginegroup.ReplicaMembership, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, cloneEngineGroupControllerTestMembership(value))
	}
	return replicas
}

func cloneEngineGroupControllerTestMembership(
	value enginegroup.ReplicaMembership,
) enginegroup.ReplicaMembership {
	value.NativeMembers = append([]enginegroup.NativeMemberID(nil), value.NativeMembers...)
	return value
}

// Compile-time interface assertions keep the fake aligned with every typed production boundary.
var (
	_ EngineGroupScalePlanResolver  = (*engineGroupControllerTestBackend)(nil)
	_ enginegroup.ServingVerifier   = (*engineGroupControllerTestBackend)(nil)
	_ enginegroup.CapacityAdapter   = engineGroupControllerTestCapacityAdapter{}
	_ enginegroup.TrafficAdapter    = engineGroupControllerTestTrafficAdapter{}
	_ enginegroup.MembershipAdapter = engineGroupControllerTestMembershipAdapter{}
	_ client.Object                 = (*nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup)(nil)
)
