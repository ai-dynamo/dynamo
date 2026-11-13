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

// TrafficProjection records the identity-level traffic projection around SGLang's
// group endpoint. SGLang currently admits new ranks internally at commit, so
// this adapter does not claim to provide a pre-admission pause for joiners.
// It projects current engine membership, not independently observed routing or
// drain evidence. Production integration needs actual serving/registration
// observations and durable withdrawal/drain, not a positive controller grant.
type TrafficProjection struct {
	Client   *Client
	Capacity CapacityObserver
	Journal  kubejournal.Store
}

type trafficJournal struct {
	AppliedRevision int64                     `json:"appliedRevision"`
	TargetDigest    string                    `json:"targetDigest"`
	Target          enginegroup.TrafficTarget `json:"target"`
}

// Observe projects the engine's current effective ranks independently of target acceptance.
func (a *TrafficProjection) Observe(
	ctx context.Context,
	groupID enginegroup.GroupID,
) (enginegroup.TrafficObservation, error) {
	state := trafficJournal{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	if snapshot.Exists() {
		if len(state.Target.Drain) != 0 {
			return enginegroup.TrafficObservation{}, fmt.Errorf("the traffic projection cannot prove drain completion")
		}
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
	return enginegroup.TrafficObservation{
		AppliedRevision: state.AppliedRevision,
		Admitted:        topology.Replicas,
	}, nil
}

// Apply records acceptance of an empty withdrawal target without granting admission.
// Drain is unsupported because the merged SGLang slice is growth-only.
func (a *TrafficProjection) Apply(
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
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if snapshot.Exists() {
		switch {
		case target.ControlRevision < state.AppliedRevision:
			return rejectedApply("StaleTrafficRevision", "traffic target revision is older than the accepted revision"), nil
		case target.ControlRevision == state.AppliedRevision && digest != state.TargetDigest:
			return rejectedApply("ConflictingTrafficRevision", "traffic target revision has a different payload"), nil
		}
	}
	state = trafficJournal{AppliedRevision: target.ControlRevision, TargetDigest: digest, Target: target}
	if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
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
	nativeCount := int32(0)
	for replica := int32(0); nativeCount < size; replica++ {
		replicaID := enginegroup.ReplicaID(fmt.Sprintf("replica-%d", replica))
		allocation, found := byReplica[replicaID]
		if !found {
			return enginegroup.MembershipTopology{}, fmt.Errorf("SGLang allocation %d has no correlated capacity", replica)
		}
		if len(allocation.Incarnation.Members) == 0 || nativeCount+int32(len(allocation.Incarnation.Members)) > size {
			return enginegroup.MembershipTopology{}, fmt.Errorf("SGLang effective membership is not a complete allocation set")
		}
		width := int32(len(allocation.Incarnation.Members))
		if !hasNativeMemberRange(allocation.Incarnation.Members, nativeCount, width) {
			return enginegroup.MembershipTopology{}, fmt.Errorf("SGLang allocation has unexpected native members")
		}
		nativeCount += width
		topology.Replicas = append(topology.Replicas, enginegroup.ReplicaMembership{
			ReplicaID: replicaID,
			Members:   append([]enginegroup.NativeMemberIncarnation(nil), allocation.Incarnation.Members...),
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
