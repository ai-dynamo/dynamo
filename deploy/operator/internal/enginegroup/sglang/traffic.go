/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
)

// TrafficAdapter records the identity-level traffic projection around SGLang's
// group endpoint. SGLang currently admits new ranks internally at commit, so
// this adapter does not claim to provide a pre-admission pause for joiners.
type TrafficAdapter struct {
	Client   *Client
	Capacity CapacityObserver
	Journal  kubejournal.Store
}

type trafficJournal struct {
	AppliedRevision int64                     `json:"appliedRevision"`
	TargetDigest    string                    `json:"targetDigest"`
	Target          enginegroup.TrafficTarget `json:"target"`
}

// Observe returns the durable projection, or bootstraps it from the engine's
// current effective ranks before the first transition.
func (a *TrafficAdapter) Observe(
	ctx context.Context,
	groupID enginegroup.GroupID,
) (enginegroup.TrafficObservation, error) {
	state := trafficJournal{}
	found, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	if found {
		return enginegroup.TrafficObservation{
			AppliedRevision: state.AppliedRevision,
			Admitted:        cloneMembership(state.Target.Admitted),
			Draining:        drainMembership(state.Target.Drain),
			Drained:         drainedMembership(state.Target.Drain),
		}, nil
	}
	if a.Client == nil || a.Capacity == nil {
		return enginegroup.TrafficObservation{}, fmt.Errorf("SGLang traffic client and capacity observer are required")
	}
	engineState, err := a.Client.Observe(ctx)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	topology, err := topologyFromCapacity(ctx, a.Capacity, groupID, 1, engineState.EffectiveEPSize)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	return enginegroup.TrafficObservation{Admitted: topology.Replicas}, nil
}

// Apply accepts an absolute logical projection. Drain is unsupported because
// the merged SGLang slice is growth-only.
func (a *TrafficAdapter) Apply(
	ctx context.Context,
	_ enginegroup.GroupID,
	target enginegroup.TrafficTarget,
) (enginegroup.ApplyResult, error) {
	if len(target.Drain) != 0 {
		return rejectedApply("DrainUnsupported", "the current SGLang adapter supports growth only"), nil
	}
	digest, err := trafficTargetDigest(target)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	state := trafficJournal{}
	found, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if found {
		switch {
		case target.ControlRevision < state.AppliedRevision:
			return rejectedApply("StaleTrafficRevision", "traffic target revision is older than the accepted revision"), nil
		case target.ControlRevision == state.AppliedRevision && digest != state.TargetDigest:
			return rejectedApply("ConflictingTrafficRevision", "traffic target revision has a different payload"), nil
		}
	}
	state = trafficJournal{AppliedRevision: target.ControlRevision, TargetDigest: digest, Target: target}
	if err := a.Journal.Save(ctx, state); err != nil {
		return enginegroup.ApplyResult{}, err
	}
	return enginegroup.ApplyResult{}, nil
}

func rejectedApply(reason, message string) enginegroup.ApplyResult {
	return enginegroup.ApplyResult{Rejection: terminalFailure(reason, message)}
}

func topologyFromCapacity(
	ctx context.Context,
	capacity CapacityObserver,
	groupID enginegroup.GroupID,
	generation int64,
	size int32,
) (enginegroup.MembershipTopology, error) {
	observation, err := capacity.Observe(ctx, groupID)
	if err != nil {
		return enginegroup.MembershipTopology{}, fmt.Errorf("observe SGLang capacity: %w", err)
	}
	byReplica := make(map[enginegroup.ReplicaID]enginegroup.CapacityAllocation, len(observation.Allocations))
	for _, allocation := range observation.Allocations {
		byReplica[allocation.Incarnation.ReplicaID] = allocation
	}
	topology := enginegroup.MembershipTopology{Generation: generation}
	for rank := int32(0); rank < size; rank++ {
		replicaID := enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank))
		allocation, found := byReplica[replicaID]
		if !found {
			return enginegroup.MembershipTopology{}, fmt.Errorf("SGLang rank %d has no correlated capacity", rank)
		}
		topology.Replicas = append(topology.Replicas, enginegroup.ReplicaMembership{
			ReplicaID:          replicaID,
			RuntimeIncarnation: allocation.Incarnation.RuntimeIncarnation,
			NativeMembers:      []enginegroup.NativeMemberID{enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank))},
		})
	}
	return topology, nil
}

func trafficTargetDigest(target enginegroup.TrafficTarget) (string, error) {
	encoded, err := json.Marshal(target)
	if err != nil {
		return "", fmt.Errorf("marshal traffic target: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

func cloneMembership(source []enginegroup.ReplicaMembership) []enginegroup.ReplicaMembership {
	cloned := make([]enginegroup.ReplicaMembership, len(source))
	for i := range source {
		cloned[i] = source[i]
		cloned[i].NativeMembers = append([]enginegroup.NativeMemberID(nil), source[i].NativeMembers...)
	}
	return cloned
}

func drainMembership(source []enginegroup.TrafficDrainTarget) []enginegroup.ReplicaMembership {
	result := make([]enginegroup.ReplicaMembership, 0, len(source))
	for _, target := range source {
		result = append(result, target.Membership)
	}
	return result
}

func drainedMembership(source []enginegroup.TrafficDrainTarget) []enginegroup.ReplicaMembership {
	// A persisted drain target is terminal evidence only for this adapter's
	// unsupported path; kept separate for future backend drain integration.
	return drainMembership(source)
}
