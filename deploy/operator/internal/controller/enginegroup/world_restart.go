/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"
	"fmt"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	ctrl "sigs.k8s.io/controller-runtime"
)

func (r *Reconciler) initializeRestartedEngineGroup(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, runtime Runtime, store kubejournal.Store, snapshot kubejournal.Snapshot) (ctrl.Result, error) {
	// A terminal formation failure cannot disappear during an operator restart or an implicit retry.
	fenceStore := runtime.Workload.Fence
	var fence worldFence
	fenceSnapshot, err := fenceStore.Load(ctx, &fence)
	if err != nil {
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	if fence.VerificationFailure != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("restarted world requires recreation after terminal verification failure: %s", fence.VerificationFailure.Reason))
	}

	// New formation is observation, not an invented membership resize of the old topology.
	observations, err := observeEngineGroupRuntime(ctx, runtime, engineGroupID(group), "")
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("observe restarted world: %w", err))
	}
	_, err = domain.NewGroupStatus(observations.membership.CommittedTopology, observations.capacity, observations.traffic)
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("initialize restarted world: %w", err))
	}
	if observations.membership.CommittedTopology.ReplicaCount() < runtime.Profile.MinSupportedReplicas ||
		observations.membership.CommittedTopology.NativeMemberCount() < runtime.Profile.MinSafeServingNativeMembers {
		return r.reconcileUnavailableRuntime(ctx, group, errors.New("restarted world has not formed its initial serving membership"))
	}
	verified, err := runtime.Verifier.Verify(ctx, engineGroupID(group), observations.membership.CommittedTopology)
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("verify restarted world: %w", err))
	}
	if verified.Failure != nil && verified.Failure.Classification == domain.FailureClassificationTerminal {
		fence.VerificationFailure = verified.Failure
		if _, err := fenceStore.Save(ctx, fenceSnapshot, fence); err != nil {
			return r.reconcileCheckpointFailure(ctx, group, err)
		}
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("restarted world failed serving verification: %s", verified.Failure.Reason))
	}
	if verified.Proof == nil || verified.Failure != nil ||
		verified.Proof.ObservedAt.IsZero() ||
		verified.Proof.TopologyGeneration != observations.membership.CommittedTopology.Generation ||
		verified.Proof.RuntimeDigest != domain.TopologyRuntimeDigest(observations.membership.CommittedTopology) {
		return r.reconcileUnavailableRuntime(ctx, group, errors.New("restarted world has no matching serving proof"))
	}

	// Revalidate exact membership after the probe; an unrelated incarnation cannot inherit its proof.
	fresh, err := observeEngineGroupRuntime(ctx, runtime, engineGroupID(group), "")
	if err != nil || !domain.SameTopology(observations.membership.CommittedTopology, fresh.membership.CommittedTopology) {
		return r.reconcileUnavailableRuntime(ctx, group, errors.Join(errors.New("restarted membership changed during serving verification"), err))
	}
	state, err := domain.NewGroupStatus(fresh.membership.CommittedTopology, fresh.capacity, fresh.traffic)
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("revalidate restarted world: %w", err))
	}
	if !allEngineGroupMembersAvailable(state.Membership.Observed.CommittedTopology, state.Registry, state.Capacity.Observed) ||
		!allEngineGroupMembersAdmitted(state.Membership.Observed.CommittedTopology, state.Traffic.Observed) {
		return r.reconcileUnavailableRuntime(ctx, group, errors.New("restarted world is not fully available and admitted"))
	}

	// Replace only lifetime-scoped evidence; desired replicas, policy, identity and immutable profile survive.
	before := group.Status.DeepCopy()
	next, err := newEngineGroupCheckpoint(runtime.Profile, state, runtime.Workload.ID).advance(group, runtime.Profile, state, nil, true, true)
	if err != nil {
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	if _, err := store.Save(ctx, snapshot, next); err != nil {
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	group.Status = api.DynamoGraphDeploymentEngineGroupStatus{Profile: runtime.Profile.DeepCopy()}
	next.projectHistory(group)
	r.projectEngineGroupStatus(group, runtime.Profile, state, nil, nil)
	if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}
