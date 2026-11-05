/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"fmt"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// ScalePlanner chooses supported native resize plans; runtime assembly only supplies its resolved profile.
type ScalePlanner struct {
	ProfileFingerprint string
	Minimum            int32
	Maximum            int32
	Layout             ReplicaLayout
}

func (p ScalePlanner) ResolveScalePlan(_ context.Context, _ domain.GroupID, target int32, status domain.GroupStatus) (domain.ScalePlanResolution, error) {
	// Finish the persisted operation before choosing another absolute target.
	if status.Transition != nil && status.Transition.Outcome != domain.TransitionOutcomeCompleted && status.Transition.Outcome != domain.TransitionOutcomeRolledBack {
		return domain.ScalePlanResolution{}, nil
	}
	if target < p.Minimum || target > p.Maximum {
		return domain.ScalePlanResolution{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal, Reason: "TargetOutOfBounds",
			Message: fmt.Sprintf("allocation target %d is outside supported bounds [%d, %d]", target, p.Minimum, p.Maximum),
		}}, nil
	}

	// Scale counts allocations; partial native membership requires recovery, never a smaller allocation target.
	topology, found := status.Topologies.Current()
	if !found {
		return domain.ScalePlanResolution{}, fmt.Errorf("resize planning requires committed membership")
	}
	if err := p.Layout.ValidateComplete(topology); err != nil {
		return domain.ScalePlanResolution{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal, Reason: "RecoveryUnsupported",
			Message: "partial allocation membership requires survivor recovery or restoration, not planned resizing",
		}}, nil
	}
	if target == topology.ReplicaCount() {
		return domain.ScalePlanResolution{}, nil
	}
	if target < topology.ReplicaCount() {
		// One victim per operation keeps single-victim release selection unambiguous before its count write.
		last := domain.ReplicaID(fmt.Sprintf("replica-%d", topology.ReplicaCount()-1))
		return domain.ScalePlanResolution{Plan: &domain.ResolvedPlan{
			ID: fmt.Sprintf("retire-g%d-%s", topology.Generation, last), ProfileFingerprint: p.ProfileFingerprint,
			ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator, TrafficRequirement: domain.TrafficRequirementEngineManagedDrain,
			VerificationRequirement: domain.VerificationRequirementRequired,
			Change:                  domain.ResolvedChange{Kind: domain.PlanKindRetire, Retire: &domain.RetireChange{Replicas: []domain.ReplicaID{last}}},
		}}, nil
	}

	// Reusing a released rank needs explicit restoration and a new launch binding, not a fresh Grow plan.
	joining := make([]domain.ReplicaTarget, 0, target-topology.ReplicaCount())
	for index := topology.ReplicaCount(); index < target; index++ {
		replica, err := p.Layout.Target(index)
		if err != nil {
			return domain.ScalePlanResolution{}, err
		}
		if _, found := status.Registry.Find(replica.ReplicaID); found {
			return domain.ScalePlanResolution{Rejection: &domain.Failure{
				Classification: domain.FailureClassificationTerminal, Reason: "RestorationUnsupported",
				Message: "reusing a released vLLM slot requires restoration with a new Pod-bound launch identity",
			}}, nil
		}
		joining = append(joining, replica)
	}
	return domain.ScalePlanResolution{Plan: &domain.ResolvedPlan{
		ID: fmt.Sprintf("grow-g%d-to-%d", topology.Generation, target), ProfileFingerprint: p.ProfileFingerprint,
		ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator, TrafficRequirement: domain.TrafficRequirementKeepServing,
		VerificationRequirement: domain.VerificationRequirementRequired,
		Change:                  domain.ResolvedChange{Kind: domain.PlanKindGrow, Grow: &domain.GrowChange{Replicas: joining}},
	}}, nil
}
