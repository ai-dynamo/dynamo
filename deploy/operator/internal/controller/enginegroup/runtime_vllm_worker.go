/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"net/http"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// vllmWorkerRuntimeResolver composes physical allocations and the Python worker's native controls.
// Client and HTTP must not be nil. Resolution is read-only, including before first formation.
type vllmWorkerRuntimeResolver struct {
	Client client.Client
	HTTP   *http.Client
}

// Resolve binds physical allocation and native membership to runtime-owned serving observations.
func (p *vllmWorkerRuntimeResolver) Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, world WorkloadBinding) (Runtime, error) {
	// Native-server and sidecar deployments need supported remote serving observations.
	if group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != string(dynamo.BackendFrameworkVLLM) {
		return Runtime{}, fmt.Errorf("%w: the vLLM runtime supports the Python worker only", ErrRuntimeUnavailable)
	}
	capacity, err := resolveVLLMCapacity(ctx, p.Client, group, world)
	if err != nil {
		return Runtime{}, err
	}
	if !dynamo.IsVLLMWorkerLaunch(capacity.Source) {
		return Runtime{}, fmt.Errorf("%w: vLLM managed resizing requires the Dynamo Python worker", ErrRuntimeUnavailable)
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(capacity.Source)
	if err != nil {
		return Runtime{}, err
	}
	membership, err := ResolveVLLMMembership(ctx, capacity, p.HTTP)
	if err != nil {
		return Runtime{}, err
	}
	membership.DrainTimeout = 120
	membership.ManagedDrainValidator = vllmruntime.ValidateWorkerManagedDrain

	// The traffic journal fences withdrawal revisions; it never grants registration.
	traffic := &vllmruntime.WorkerTraffic{
		Membership: membership,
		Journal:    runtimeJournal(p.Client, group, world.ID, "vllm-traffic"),
	}
	return Runtime{
		Workload: world,
		Profile: api.EngineGroupProfileStatus{
			Backend: "vllm", Fingerprint: profile.Geometry.Fingerprint,
			GPUsPerReplica: profile.Geometry.GPUsPerReplica, PodsPerReplica: profile.Geometry.PodsPerReplica,
			NativeMembersPerReplica: profile.NativeMembersPerReplica, MinSafeServingNativeMembers: membership.MinDataParallelSize,
			MinSupportedReplicas: profile.InitialReplicas, MaxSupportedReplicas: profile.MaximumReplicas,
		},
		Capacity: capacity, Membership: membership, Traffic: traffic,
		Verifier: &vllmruntime.WorkerVerifier{Membership: membership},
		Planner: vllmScalePlanner{
			fingerprint: profile.Geometry.Fingerprint, minimum: profile.InitialReplicas, maximum: profile.MaximumReplicas,
			layout: membership.Layout,
		},
	}, nil
}

type vllmScalePlanner struct {
	fingerprint string
	minimum     int32
	maximum     int32
	layout      vllmruntime.ReplicaLayout
}

func (p vllmScalePlanner) ResolveScalePlan(_ context.Context, _ domain.GroupID, target int32, status domain.GroupStatus) (ScalePlanResolution, error) {
	// Finish the persisted operation before choosing another absolute target.
	if status.Transition != nil && status.Transition.Outcome != domain.TransitionOutcomeCompleted && status.Transition.Outcome != domain.TransitionOutcomeRolledBack {
		return ScalePlanResolution{}, nil
	}
	if target < p.minimum || target > p.maximum {
		return ScalePlanResolution{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal, Reason: "TargetOutOfBounds",
			Message: fmt.Sprintf("allocation target %d is outside supported bounds [%d, %d]", target, p.minimum, p.maximum),
		}}, nil
	}

	// Scale counts allocations; partial native membership requires recovery, never a smaller allocation target.
	topology, found := status.Topologies.Current()
	if !found {
		return ScalePlanResolution{}, fmt.Errorf("resize planning requires committed membership")
	}
	expected, err := p.layout.NativeSize(topology.ReplicaCount())
	if err != nil {
		return ScalePlanResolution{}, err
	}
	complete := topology.NativeMemberCount() == expected
	for _, allocation := range topology.Replicas {
		complete = complete && len(allocation.Members) == int(p.layout.NativeMembersPerReplica)
	}
	if !complete {
		return ScalePlanResolution{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal, Reason: "RecoveryUnsupported",
			Message: "partial allocation membership requires survivor recovery or restoration, not planned resizing",
		}}, nil
	}
	if target == topology.ReplicaCount() {
		return ScalePlanResolution{}, nil
	}
	if target < topology.ReplicaCount() {
		// One victim per operation keeps single-victim release selection unambiguous before its count write.
		last := domain.ReplicaID(fmt.Sprintf("replica-%d", topology.ReplicaCount()-1))
		return ScalePlanResolution{Plan: &domain.ResolvedPlan{
			ID: fmt.Sprintf("retire-g%d-%s", topology.Generation, last), ProfileFingerprint: p.fingerprint,
			ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator, TrafficRequirement: domain.TrafficRequirementEngineManagedDrain,
			VerificationRequirement: domain.VerificationRequirementRequired,
			Change:                  domain.ResolvedChange{Kind: domain.PlanKindRetire, Retire: &domain.RetireChange{Replicas: []domain.ReplicaID{last}}},
		}}, nil
	}

	// Reusing a released rank needs explicit restoration and a new launch binding, not a fresh Grow plan.
	joining := make([]domain.ReplicaTarget, 0, target-topology.ReplicaCount())
	for index := topology.ReplicaCount(); index < target; index++ {
		id := domain.ReplicaID(fmt.Sprintf("replica-%d", index))
		if _, found := status.Registry.Find(id); found {
			return ScalePlanResolution{Rejection: &domain.Failure{
				Classification: domain.FailureClassificationTerminal, Reason: "RestorationUnsupported",
				Message: "reusing a released vLLM slot requires restoration with a new Pod-bound launch identity",
			}}, nil
		}
		members, err := p.layout.Members(index)
		if err != nil {
			return ScalePlanResolution{}, err
		}
		joining = append(joining, domain.ReplicaTarget{
			ReplicaID: id, SlotID: domain.CapacitySlotID(fmt.Sprintf("slot-%d", index)),
			Bootstrap: domain.BootstrapModeJoin, NativeMembers: members,
		})
	}
	return ScalePlanResolution{Plan: &domain.ResolvedPlan{
		ID: fmt.Sprintf("grow-g%d-to-%d", topology.Generation, target), ProfileFingerprint: p.fingerprint,
		ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator, TrafficRequirement: domain.TrafficRequirementKeepServing,
		VerificationRequirement: domain.VerificationRequirementRequired,
		Change:                  domain.ResolvedChange{Kind: domain.PlanKindGrow, Grow: &domain.GrowChange{Replicas: joining}},
	}}, nil
}
