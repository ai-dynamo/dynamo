/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"k8s.io/apimachinery/pkg/types"
)

const engineGroupCheckpointVersion = 1

type checkpointBindingChangedError struct {
	PreviousUID types.UID
}

func (e *checkpointBindingChangedError) Error() string {
	return "checkpoint belongs to a different PodClique incarnation"
}

// engineGroupCheckpoint is private persistence, not a second public Kubernetes API.
// Version identifies the persisted recovery schema introduced with this feature.
// A physical binding change requires explicit whole-world recovery, never silent adoption.
type engineGroupCheckpoint struct {
	Version            int
	ProfileFingerprint string
	BindingUID         types.UID
	State              domain.GroupStatus

	// Preserve projection inputs across a failed public status write.
	DesiredNativeMembers         []string
	DesiredAssignmentGeneration  int64
	LastStableReplicas           int32
	LastStableTopologyGeneration int64
	Operation                    *api.EngineGroupOperationStatus
}

func engineGroupCheckpointStore(r *Reconciler, group *api.DynamoGraphDeploymentEngineGroup) kubejournal.Store {
	return kubejournal.NewStore(r.Client, group.Namespace, group.Name, group.UID, "checkpoint")
}

func loadEngineGroupCheckpoint(
	ctx context.Context,
	store kubejournal.Store,
	group *api.DynamoGraphDeploymentEngineGroup,
) (engineGroupCheckpoint, kubejournal.Snapshot, error) {
	var checkpoint engineGroupCheckpoint
	snapshot, err := store.Load(ctx, &checkpoint)
	if err != nil {
		return engineGroupCheckpoint{}, snapshot, err
	}
	if !snapshot.Exists() {
		if group.Status.Topology != nil || group.Status.Operation != nil {
			return checkpoint, snapshot, errors.New("checkpoint is missing for an initialized Engine Group")
		}
		return checkpoint, snapshot, nil
	}

	// Validate the payload before considering a separately authorized whole-world restart.
	err = validateEngineGroupCheckpoint(checkpoint, group, types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]))
	return checkpoint, snapshot, err
}

func validateEngineGroupCheckpoint(checkpoint engineGroupCheckpoint, group *api.DynamoGraphDeploymentEngineGroup, bindingUID types.UID) error {
	// Refuse unknown schemas and evidence belonging to another profile or physical world.
	if checkpoint.Version != engineGroupCheckpointVersion {
		return fmt.Errorf("unsupported checkpoint version %d", checkpoint.Version)
	}
	if group.Status.Profile == nil || checkpoint.ProfileFingerprint != group.Status.Profile.Fingerprint {
		return errors.New("checkpoint profile does not match the Engine Group")
	}
	if topology, found := checkpoint.State.Topologies.Current(); !found || topology.Generation <= 0 {
		return errors.New("checkpoint has no authoritative base topology")
	}
	if transition := checkpoint.State.Transition; transition != nil {
		if checkpoint.Operation == nil || checkpoint.Operation.ID != transition.Spec.ID {
			return errors.New("checkpoint operation correlation is invalid")
		}
	} else if checkpoint.Operation != nil {
		return errors.New("checkpoint contains an operation without its transition")
	}
	if err := domain.ValidateGroupStatus(checkpoint.State); err != nil {
		return fmt.Errorf("invalid checkpoint state: %w", err)
	}
	if checkpoint.BindingUID != bindingUID {
		return &checkpointBindingChangedError{PreviousUID: checkpoint.BindingUID}
	}
	return nil
}

func newEngineGroupCheckpoint(group *api.DynamoGraphDeploymentEngineGroup, profile api.EngineGroupProfileStatus, state domain.GroupStatus) engineGroupCheckpoint {
	return engineGroupCheckpoint{
		Version: engineGroupCheckpointVersion, ProfileFingerprint: profile.Fingerprint,
		BindingUID: types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]),
		State:      state,
	}
}

// advance updates durable history without consulting mutable public projections.
// A failed observation preserves history, not a claim of fresh serving health.
func (c engineGroupCheckpoint) advance(group *api.DynamoGraphDeploymentEngineGroup, profile api.EngineGroupProfileStatus, state domain.GroupStatus, plan *domain.ResolvedPlan, assignmentValid, observationsFresh bool) (engineGroupCheckpoint, error) {
	next := c
	next.State = state
	if assignmentValid {
		next.DesiredNativeMembers, next.DesiredAssignmentGeneration = resolveEngineGroupDesiredAssignment(
			group, profile, state, plan, c.DesiredNativeMembers, c.DesiredAssignmentGeneration)
	}
	operation, err := resolveEngineGroupOperation(group, state, c.Operation)
	if err != nil {
		return engineGroupCheckpoint{}, err
	}
	next.Operation = operation
	if observationsFresh && next.isStable(group, profile) {
		next.LastStableReplicas = int32(len(state.Capacity.Observed.Allocations))
		next.LastStableTopologyGeneration = state.Membership.Observed.CommittedTopology.Generation
	}
	return next, nil
}

func (c engineGroupCheckpoint) isStable(group *api.DynamoGraphDeploymentEngineGroup, profile api.EngineGroupProfileStatus) bool {
	state := c.State
	topology := state.Membership.Observed.CommittedTopology
	current, found := state.Topologies.Current()
	settled := state.Transition == nil || state.Transition.Outcome == domain.TransitionOutcomeCompleted ||
		state.Transition.Outcome == domain.TransitionOutcomeRolledBack
	return found && settled && domain.SameTopology(current, topology) && topology.Generation > 0 &&
		c.DesiredAssignmentGeneration == group.Generation &&
		int32(len(state.Capacity.Observed.Allocations)) == group.Spec.Replicas &&
		int64(len(c.DesiredNativeMembers)) == int64(group.Spec.Replicas)*int64(profile.NativeMembersPerReplica) &&
		slices.Equal(engineGroupTopologyNativeMembers(topology), c.DesiredNativeMembers) &&
		topology.NativeMemberCount() >= profile.MinSafeServingNativeMembers &&
		allEngineGroupMembersAvailable(topology, state.Registry, state.Capacity.Observed) &&
		allEngineGroupMembersAdmitted(topology, state.Traffic.Observed) &&
		!engineGroupHasDegradedAllocation(projectEngineGroupReplicaStates(state))
}

// projectHistory copies already-durable correlation, never historical health as fresh evidence.
func (c engineGroupCheckpoint) projectHistory(group *api.DynamoGraphDeploymentEngineGroup) {
	group.Status.DesiredNativeMembers = slices.Clone(c.DesiredNativeMembers)
	group.Status.DesiredAssignmentGeneration = c.DesiredAssignmentGeneration
	group.Status.LastStableReplicas = c.LastStableReplicas
	group.Status.LastStableTopologyGeneration = c.LastStableTopologyGeneration
	group.Status.Operation = c.Operation.DeepCopy()
}

func persistEngineGroupCheckpoint(
	ctx context.Context,
	store kubejournal.Store,
	snapshot kubejournal.Snapshot,
	previous, next engineGroupCheckpoint,
) error {
	// Avoid watch churn on a settled world while retaining compare-and-swap fencing for changed levels.
	if snapshot.Exists() && reflect.DeepEqual(previous, next) {
		return nil
	}
	if _, err := store.Save(ctx, snapshot, next); err != nil {
		return fmt.Errorf("persist Engine Group checkpoint: %w", err)
	}
	return nil
}
