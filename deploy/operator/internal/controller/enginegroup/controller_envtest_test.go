//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
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
	reconciler := &Reconciler{
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
	checkpoint := engineGroupControllerTestCheckpoint(t, ctx, env.Client(), stored)
	require.NotNil(t, checkpoint.State.Membership.Desired)
	desiredTransitionID := checkpoint.State.Membership.Desired.TransitionID
	desiredDigest := checkpoint.State.Membership.Desired.TargetDigest
	require.NotNil(t, stored.Status.Operation)
	assert.Equal(t, checkpoint.State.Transition.Spec.ID, stored.Status.Operation.ID)
	assert.Equal(t, int32(3), stored.Status.Operation.TargetReplicas)
	assert.NotEmpty(t, desiredTransitionID)
	assert.NotEmpty(t, desiredDigest)

	t.Log("let the adapter publish the correlated commit and replace the reconciler instance")
	backend.commitPendingMembership()
	plannerErr := errors.New("planner temporarily unavailable")
	backend.setPlannerError(plannerErr)
	reconciler = &Reconciler{
		Client:          env.Client(),
		RuntimeProvider: provider,
	}

	t.Log("resume from the private checkpoint until capacity, membership, verification, and traffic converge")
	var lastReconcileErr error
	var lastObserved *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup
	completed := false
	for step := 0; step < 64; step++ {
		_, lastReconcileErr = reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		current := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{}
		require.NoError(t, env.Client().Get(ctx, key, current))
		lastObserved = current
		checkpoint := engineGroupControllerTestCheckpoint(t, ctx, env.Client(), current)
		if checkpoint.State.Transition != nil &&
			checkpoint.State.Transition.Outcome == domain.TransitionOutcomeCompleted {
			completed = true
			break
		}
	}
	require.True(t, completed, "last reconcile error: %v; last reconciliation: %#v", lastReconcileErr,
		lastObserved.Status.Operation)
	require.ErrorIs(t, lastReconcileErr, plannerErr)

	t.Log("verify restart recovery reused one transition and exposed the converged scale/status views")
	require.NoError(t, env.Client().Get(ctx, key, stored))
	assert.Equal(t, int32(3), stored.Status.Replicas)
	assert.Equal(t, int32(3), stored.Status.AvailableReplicas)
	assert.Equal(t, int32(3), stored.Status.ActiveNativeMemberCount)
	checkpoint = engineGroupControllerTestCheckpoint(t, ctx, env.Client(), stored)
	assert.Equal(t, desiredTransitionID, checkpoint.State.Membership.Desired.TransitionID)
	assert.Equal(t, desiredDigest, checkpoint.State.Membership.Desired.TargetDigest)
	assert.Equal(t, nvidiacomv1beta1.EngineGroupOperationPhaseCommitted, stored.Status.Operation.Phase)
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

func TestEngineGroupControllerRetiresWorldBeforeDeletion(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := context.Background()
	backend := newEngineGroupControllerTestBackend(1)
	provider := engineGroupControllerTestRuntimeProvider{backend: backend}
	reconciler := &Reconciler{
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

	t.Log("drive terminal retirement until the adapter accepts it but the HTTP result is ambiguous")
	for step := 0; step < 32; step++ {
		_, err = reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		if err != nil {
			break
		}
	}
	require.ErrorContains(t, err, "ambiguous membership apply")
	require.NoError(t, env.Client().Get(ctx, key, group))
	assert.Equal(t, int32(1), group.Spec.Replicas)
	require.NotNil(t, group.Status.Operation)
	assert.Equal(t, "Retire", group.Status.Operation.Intent)
	assert.Zero(t, group.Status.Operation.TargetReplicas)
	assert.Empty(t, group.Status.Operation.TargetNativeMembers)
	checkpoint := engineGroupControllerTestCheckpoint(t, ctx, env.Client(), group)
	require.NotNil(t, checkpoint.State.Membership.Desired)
	transitionID := checkpoint.State.Membership.Desired.TransitionID
	assert.Empty(t, backend.traffic.Admitted)
	require.Len(t, backend.traffic.Drained, 1)
	assert.Len(t, backend.capacity.Allocations, 1, "capacity must remain held before the terminal commit")

	t.Log("restart the controller, observe the exact commit, then converge UID-bound release and terminal targets")
	backend.commitPendingMembership()
	reconciler = &Reconciler{Client: env.Client(), RuntimeProvider: provider}
	deleted := false
	for step := 0; step < 32; step++ {
		_, err = reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: key})
		require.NoError(t, err)
		err = env.Client().Get(ctx, key, group)
		if apierrors.IsNotFound(err) {
			deleted = true
			break
		}
		require.NoError(t, err)
		assert.Equal(t, int32(1), group.Spec.Replicas)
	}
	require.True(t, deleted, "retirement must eventually remove the Engine Group")
	assert.Equal(t, 1, backend.membershipApplyCount())
	assert.Equal(t, transitionID, backend.transition.TransitionID)
	assert.Empty(t, backend.topology.Replicas)
	assert.Empty(t, backend.capacity.Allocations)
	assert.Empty(t, backend.traffic.Admitted)
	require.Len(t, backend.capacity.ReleaseFences, 1)
	assert.Equal(t, domain.PodUID("uid-0"), backend.capacity.ReleaseFences[0].CapacityRefs[0].UID)
	assert.Equal(t, backend.topology.Generation, backend.capacity.ReleaseFences[0].AuthorizingTopologyGeneration)
}

func TestEngineGroupControllerDeletesEmptyWorldBeforeJournalInitialization(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := context.Background()
	backend := newEngineGroupControllerTestBackend(0)
	backend.ambiguousApply = false
	reconciler := &Reconciler{
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
	assert.Nil(t, group.Status.Operation)

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
	reconciler := &Reconciler{
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
) (Runtime, error) {
	return Runtime{
		Profile: nvidiacomv1beta1.EngineGroupProfileStatus{
			Backend:                     "sglang",
			Fingerprint:                 "profile-v1",
			GPUsPerReplica:              1,
			PodsPerReplica:              1,
			NativeMembersPerReplica:     1,
			MinSafeServingNativeMembers: 1,
			MinSupportedReplicas:        1,
			MaxSupportedReplicas:        8,
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
	capacity                   domain.CapacityObservation
	traffic                    domain.TrafficObservation
	topology                   domain.MembershipTopology
	transition                 *domain.MembershipTransitionObservation
	pendingTarget              *domain.MembershipTarget
	ambiguousApply             bool
	plannerErr                 error
	observationErrors          map[domain.ObservationAuthority]error
	verificationFailure        *domain.Failure
	membershipApplyInvocations int
}

func newEngineGroupControllerTestBackend(replicas int) *engineGroupControllerTestBackend {
	backend := &engineGroupControllerTestBackend{ambiguousApply: true}
	backend.topology = domain.MembershipTopology{Generation: 1}
	for index := 0; index < replicas; index++ {
		replicaID := domain.ReplicaID(fmt.Sprintf("replica-%d", index))
		slotID := domain.CapacitySlotID(fmt.Sprintf("slot-%d", index))
		runtimeID := domain.RuntimeIncarnationID(fmt.Sprintf("runtime-%d", index))
		membership := domain.ReplicaMembership{
			ReplicaID: replicaID,
			Members:   []domain.NativeMemberIncarnation{{ID: domain.NativeMemberID(fmt.Sprintf("dp-%d", index)), RuntimeIncarnation: runtimeID}},
		}
		backend.topology.Replicas = append(backend.topology.Replicas, membership)
		backend.traffic.Admitted = append(backend.traffic.Admitted, membership)
		backend.capacity.Allocations = append(backend.capacity.Allocations, domain.CapacityAllocation{
			Incarnation: domain.ReplicaIncarnation{
				ReplicaID: replicaID,
				SlotID:    slotID,
				Members:   append([]domain.NativeMemberIncarnation(nil), membership.Members...),
				CapacityRefs: []domain.CapacityRef{{
					Name: fmt.Sprintf("worker-%d", index),
					UID:  domain.PodUID(fmt.Sprintf("uid-%d", index)),
				}},
			},
			Available: true,
		})
	}
	return backend
}

func (b *engineGroupControllerTestBackend) ResolveScalePlan(
	_ context.Context,
	_ domain.GroupID,
	targetReplicas int32,
	status domain.GroupStatus,
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
	plan := &domain.ResolvedPlan{
		ID:                      fmt.Sprintf("scale-%d-%d", current.Generation, targetReplicas),
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   domain.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      domain.TrafficRequirementKeepServing,
		VerificationRequirement: domain.VerificationRequirementRequired,
	}
	if targetReplicas > current.ReplicaCount() {
		plan.Change = domain.ResolvedChange{Kind: domain.PlanKindGrow, Grow: &domain.GrowChange{}}
		for index := current.ReplicaCount(); index < targetReplicas; index++ {
			plan.Change.Grow.Replicas = append(plan.Change.Grow.Replicas, domain.ReplicaTarget{
				ReplicaID:     domain.ReplicaID(fmt.Sprintf("replica-%d", index)),
				SlotID:        domain.CapacitySlotID(fmt.Sprintf("slot-%d", index)),
				Bootstrap:     domain.BootstrapModeJoin,
				NativeMembers: []domain.NativeMemberID{domain.NativeMemberID(fmt.Sprintf("dp-%d", index))},
			})
		}
		return ScalePlanResolution{Plan: plan}, nil
	}

	plan.Change = domain.ResolvedChange{Kind: domain.PlanKindRetire, Retire: &domain.RetireChange{}}
	if targetReplicas == 0 {
		plan.TrafficRequirement = domain.TrafficRequirementQuiesceGroup
		plan.VerificationRequirement = domain.VerificationRequirementNone
	}
	for index := current.ReplicaCount() - 1; index >= targetReplicas; index-- {
		plan.Change.Retire.Replicas = append(plan.Change.Retire.Replicas,
			domain.ReplicaID(fmt.Sprintf("replica-%d", index)))
	}
	return ScalePlanResolution{Plan: plan}, nil
}

type engineGroupControllerTestCapacityAdapter struct {
	backend *engineGroupControllerTestBackend
}

func (a engineGroupControllerTestCapacityAdapter) Observe(
	context.Context,
	domain.GroupID,
) (domain.CapacityObservation, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	if err := a.backend.observationErrors[domain.ObservationAuthorityCapacity]; err != nil {
		return domain.CapacityObservation{}, err
	}
	return cloneEngineGroupControllerTestCapacity(a.backend.capacity), nil
}

func (a engineGroupControllerTestCapacityAdapter) Apply(
	_ context.Context,
	_ domain.GroupID,
	target domain.CapacityTarget,
) (domain.ApplyResult, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	existing := make(map[domain.ReplicaID]domain.CapacityAllocation, len(a.backend.capacity.Allocations))
	for _, allocation := range a.backend.capacity.Allocations {
		existing[allocation.Incarnation.ReplicaID] = allocation
	}
	allocations := make([]domain.CapacityAllocation, 0, len(target.Replicas))
	for _, replica := range target.Replicas {
		if allocation, found := existing[replica.ReplicaID]; found {
			allocations = append(allocations, allocation)
			continue
		}
		if replica.Bootstrap == nil {
			continue
		}
		index := len(existing)
		allocations = append(allocations, domain.CapacityAllocation{
			Incarnation: domain.ReplicaIncarnation{
				ReplicaID: replica.ReplicaID,
				SlotID:    replica.SlotID,
				Members:   []domain.NativeMemberIncarnation{{ID: domain.NativeMemberID(fmt.Sprintf("dp-%d", index)), RuntimeIncarnation: domain.RuntimeIncarnationID(fmt.Sprintf("runtime-%d", index))}},
				CapacityRefs: []domain.CapacityRef{{
					Name: fmt.Sprintf("worker-%d", index),
					UID:  domain.PodUID(fmt.Sprintf("uid-%d", index)),
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
	a.backend.capacity = domain.CapacityObservation{
		AppliedRevision: target.ControlRevision,
		Allocations:     allocations,
		ReleaseFences:   append([]domain.ReleaseFence(nil), target.ReleaseFences...),
	}
	return domain.ApplyResult{}, nil
}

type engineGroupControllerTestTrafficAdapter struct {
	backend *engineGroupControllerTestBackend
}

func (a engineGroupControllerTestTrafficAdapter) Observe(
	context.Context,
	domain.GroupID,
) (domain.TrafficObservation, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	if err := a.backend.observationErrors[domain.ObservationAuthorityTraffic]; err != nil {
		return domain.TrafficObservation{}, err
	}
	return cloneEngineGroupControllerTestTraffic(a.backend.traffic), nil
}

func (a engineGroupControllerTestTrafficAdapter) Apply(
	_ context.Context,
	_ domain.GroupID,
	target domain.TrafficTarget,
) (domain.ApplyResult, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	drained := make([]domain.ReplicaMembership, 0, len(target.Drain))
	for _, target := range target.Drain {
		drained = append(drained, cloneEngineGroupControllerTestMembership(target.Membership))
	}
	a.backend.traffic = domain.TrafficObservation{
		AppliedRevision: target.ControlRevision,
		Admitted:        cloneEngineGroupControllerTestMemberships(target.Admitted),
		Drained:         drained,
	}
	return domain.ApplyResult{}, nil
}

type engineGroupControllerTestMembershipAdapter struct {
	backend *engineGroupControllerTestBackend
}

func (a engineGroupControllerTestMembershipAdapter) ValidatePlan(
	_ context.Context,
	_ domain.GroupID,
	request domain.PlanValidationRequest,
) (domain.PreflightResult, error) {
	return domain.PreflightResult{Evidence: &domain.ValidationEvidence{
		PlanDigest:           request.PlanDigest,
		ProfileFingerprint:   request.Plan.ProfileFingerprint,
		CapabilityGeneration: "test-v1",
	}}, nil
}

func (a engineGroupControllerTestMembershipAdapter) ValidateTarget(
	_ context.Context,
	_ domain.GroupID,
	target domain.MembershipTarget,
) (domain.PreflightResult, error) {
	return domain.PreflightResult{Evidence: &domain.ValidationEvidence{
		PlanDigest:           target.Validation.PlanDigest,
		TargetDigest:         target.TargetDigest,
		ProfileFingerprint:   target.Plan.ProfileFingerprint,
		CapabilityGeneration: "test-v1",
	}}, nil
}

func (a engineGroupControllerTestMembershipAdapter) Observe(
	_ context.Context,
	_ domain.GroupID,
	transitionID string,
) (domain.MembershipObservation, error) {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	if err := a.backend.observationErrors[domain.ObservationAuthorityMembership]; err != nil {
		return domain.MembershipObservation{}, err
	}
	observation := domain.MembershipObservation{
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
	_ domain.GroupID,
	target domain.MembershipTarget,
) error {
	a.backend.mu.Lock()
	defer a.backend.mu.Unlock()
	if a.backend.transition != nil && a.backend.transition.TransitionID == target.TransitionID &&
		a.backend.transition.Phase == domain.MembershipTransitionPhaseCommitted {
		return nil
	}
	a.backend.membershipApplyInvocations++
	targetCopy := target
	a.backend.pendingTarget = &targetCopy
	a.backend.transition = &domain.MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           domain.MembershipTransitionPhasePending,
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
	_ domain.GroupID,
	topology domain.MembershipTopology,
) (domain.VerificationResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.verificationFailure != nil {
		return domain.VerificationResult{Failure: b.verificationFailure}, nil
	}
	return domain.VerificationResult{Proof: &domain.ServingProof{
		TopologyGeneration: topology.Generation,
		RuntimeDigest:      domain.TopologyRuntimeDigest(topology),
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

func (b *engineGroupControllerTestBackend) commitMembershipLocked(target domain.MembershipTarget) {
	result := target.BaseTopology
	result.Generation++
	switch target.Plan.Change.Kind {
	case domain.PlanKindGrow:
		for _, joining := range target.Joining {
			for _, planned := range target.Plan.Change.Grow.Replicas {
				if joining.ReplicaID == planned.ReplicaID {
					result.Replicas = append(result.Replicas, domain.ReplicaMembership{
						ReplicaID: joining.ReplicaID,
						Members:   append([]domain.NativeMemberIncarnation(nil), joining.Members...),
					})
				}
			}
		}
	case domain.PlanKindRetire:
		retiring := make(map[domain.ReplicaID]struct{}, len(target.Plan.Change.Retire.Replicas))
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
	b.transition = &domain.MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           domain.MembershipTransitionPhaseCommitted,
		ResultTopology:  &result,
	}
	b.pendingTarget = nil
}

func cloneEngineGroupControllerTestCapacity(
	value domain.CapacityObservation,
) domain.CapacityObservation {
	value.Allocations = append([]domain.CapacityAllocation(nil), value.Allocations...)
	for index := range value.Allocations {
		value.Allocations[index].Incarnation.CapacityRefs = append(
			[]domain.CapacityRef(nil),
			value.Allocations[index].Incarnation.CapacityRefs...,
		)
	}
	value.ReleaseFences = append([]domain.ReleaseFence(nil), value.ReleaseFences...)
	return value
}

func cloneEngineGroupControllerTestTraffic(value domain.TrafficObservation) domain.TrafficObservation {
	value.Admitted = cloneEngineGroupControllerTestMemberships(value.Admitted)
	value.Draining = cloneEngineGroupControllerTestMemberships(value.Draining)
	value.Drained = cloneEngineGroupControllerTestMemberships(value.Drained)
	return value
}

func cloneEngineGroupControllerTestTopology(value domain.MembershipTopology) domain.MembershipTopology {
	value.Replicas = cloneEngineGroupControllerTestMemberships(value.Replicas)
	return value
}

func cloneEngineGroupControllerTestMemberships(
	values []domain.ReplicaMembership,
) []domain.ReplicaMembership {
	replicas := make([]domain.ReplicaMembership, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, cloneEngineGroupControllerTestMembership(value))
	}
	return replicas
}

func cloneEngineGroupControllerTestMembership(
	value domain.ReplicaMembership,
) domain.ReplicaMembership {
	value.Members = append([]domain.NativeMemberIncarnation(nil), value.Members...)
	return value
}

// Compile-time interface assertions keep the fake aligned with every typed production boundary.
var (
	_ ScalePlanResolver        = (*engineGroupControllerTestBackend)(nil)
	_ domain.ServingVerifier   = (*engineGroupControllerTestBackend)(nil)
	_ domain.CapacityAdapter   = engineGroupControllerTestCapacityAdapter{}
	_ domain.TrafficAdapter    = engineGroupControllerTestTrafficAdapter{}
	_ domain.MembershipAdapter = engineGroupControllerTestMembershipAdapter{}
	_ client.Object            = (*nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup)(nil)
)
