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
// Version it whenever the serialized coordinator state or its recovery invariants change.
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

func newEngineGroupCheckpoint(group *api.DynamoGraphDeploymentEngineGroup, state domain.GroupStatus) engineGroupCheckpoint {
	return engineGroupCheckpoint{
		Version: engineGroupCheckpointVersion, ProfileFingerprint: group.Status.Profile.Fingerprint,
		BindingUID:                   types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]),
		State:                        state,
		DesiredNativeMembers:         slices.Clone(group.Status.DesiredNativeMembers),
		DesiredAssignmentGeneration:  group.Status.DesiredAssignmentGeneration,
		LastStableReplicas:           group.Status.LastStableReplicas,
		LastStableTopologyGeneration: group.Status.LastStableTopologyGeneration,
		Operation:                    group.Status.Operation.DeepCopy(),
	}
}

// restoreProjectionInputs restores durable correlation, never cached health as a fresh observation.
func (c engineGroupCheckpoint) restoreProjectionInputs(group *api.DynamoGraphDeploymentEngineGroup) {
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
