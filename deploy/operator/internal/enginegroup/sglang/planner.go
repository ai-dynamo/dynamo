/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"fmt"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// ScalePlanner chooses supported native resize plans; runtime assembly only supplies its resolved profile.
type ScalePlanner struct {
	ProfileFingerprint string
	WorldID            string
	MembersPerReplica  int32
	RetirementEnabled  bool
}

func (p ScalePlanner) ResolveScalePlan(
	_ context.Context,
	_ domain.GroupID,
	targetReplicas int32,
	status domain.GroupStatus,
) (domain.ScalePlanResolution, error) {
	// An unresolved operation owns the current collective; a newer target may only queue behind it.
	if status.Transition != nil && status.Transition.Outcome != domain.TransitionOutcomeCompleted &&
		status.Transition.Outcome != domain.TransitionOutcomeRolledBack {
		return domain.ScalePlanResolution{}, nil
	}

	base := status.Membership.Observed.CommittedTopology
	layout := ReplicaLayout{NativeMembersPerReplica: p.MembersPerReplica}
	if err := layout.ValidateComplete(base); err != nil {
		return domain.ScalePlanResolution{Rejection: terminalFailure("PartialMembershipUnsupported", err.Error())}, nil
	}
	current := base.ReplicaCount()
	if targetReplicas == current {
		return domain.ScalePlanResolution{}, nil
	}

	// Retiring the primary would destroy control authority and requires whole-world retirement.
	if targetReplicas == 0 {
		return domain.ScalePlanResolution{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal,
			Reason:         "SGLangRetirementUnsupported",
			Message:        "the SGLang adapter cannot safely retire the primary engine world",
		}}, nil
	}
	if targetReplicas < current {
		if !p.RetirementEnabled {
			return domain.ScalePlanResolution{Rejection: &domain.Failure{
				Classification: domain.FailureClassificationTerminal,
				Reason:         "SGLangShrinkUnsupported",
				Message:        "planned shrink requires explicit native external retiree lifecycle",
			}}, nil
		}
		return domain.ScalePlanResolution{Plan: &domain.ResolvedPlan{
			ID:                      fmt.Sprintf("sglang-retire-%s-%d-%d", p.WorldID, base.Generation, current-1),
			ProfileFingerprint:      p.ProfileFingerprint,
			ProcessLifecycleOwner:   domain.ProcessLifecycleOwnerOrchestrator,
			TrafficRequirement:      domain.TrafficRequirementEngineManagedDrain,
			VerificationRequirement: domain.VerificationRequirementRequired,
			Change: domain.ResolvedChange{Kind: domain.PlanKindRetire,
				Retire: &domain.RetireChange{Replicas: []domain.ReplicaID{domain.ReplicaID(fmt.Sprintf("replica-%d", current-1))}},
			},
		}}, nil
	}
	// Each native append admits one complete allocation; larger absolute targets converge through separate commits.
	joining, err := layout.Target(current)
	if err != nil {
		return domain.ScalePlanResolution{}, err
	}
	return domain.ScalePlanResolution{Plan: &domain.ResolvedPlan{
		ID:                      fmt.Sprintf("sglang-grow-%s-%d-%d", p.WorldID, base.Generation, current+1),
		ProfileFingerprint:      p.ProfileFingerprint,
		ProcessLifecycleOwner:   domain.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      domain.TrafficRequirementKeepServing,
		VerificationRequirement: domain.VerificationRequirementRequired,
		Change: domain.ResolvedChange{
			Kind: domain.PlanKindGrow,
			Grow: &domain.GrowChange{Replicas: []domain.ReplicaTarget{joining}},
		},
	}}, nil
}
