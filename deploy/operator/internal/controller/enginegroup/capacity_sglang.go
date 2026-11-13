/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	sglangruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/sglang"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/workload"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// SGLangCapacity binds the native profile's process identity to physical capacity.
// The native allocation is one process tree; container restarts require explicit group recovery.
type SGLangCapacity struct {
	Physical          workload.Capacity
	Reader            client.Reader
	Namespace         string
	MembersPerReplica int32
}

// ReconcileInitialCapacity prepares physical allocation prerequisites through the controller's existing capacity hook.
// Native bootstrap remains template-driven and does not wait for a serving permission.
func (a *SGLangCapacity) ReconcileInitialCapacity(ctx context.Context, _ domain.GroupID) error {
	return a.Physical.Prepare(ctx)
}

// Observe binds each allocation's fixed member range to its non-restarting process tree.
func (a *SGLangCapacity) Observe(ctx context.Context, groupID domain.GroupID) (domain.CapacityObservation, error) {
	// The workload provider owns Pod identity and placement; the engine integration owns process lifetime and rank assignment.
	physical, err := a.Physical.Observe(ctx, groupID)
	if err != nil {
		return domain.CapacityObservation{}, err
	}
	observation := domain.CapacityObservation{AppliedRevision: physical.AppliedRevision, ReleaseFences: physical.ReleaseFences, ReleaseMode: physical.ReleaseMode}
	for _, allocation := range physical.Allocations {
		pod := &corev1.Pod{}
		if err := a.Reader.Get(ctx, client.ObjectKey{Namespace: a.Namespace, Name: allocation.Pod.Name}, pod); err != nil {
			return domain.CapacityObservation{}, err
		}
		if domain.PodUID(pod.UID) != allocation.Pod.UID {
			return domain.CapacityObservation{}, fmt.Errorf("%w: SGLang Pod changed during process observation", domain.ErrRecoveryRequired)
		}
		for _, container := range pod.Status.ContainerStatuses {
			if container.Name == consts.MainContainerName && container.RestartCount > 0 {
				return domain.CapacityObservation{}, fmt.Errorf("%w: SGLang Pod %s restarted outside membership recovery", domain.ErrRecoveryRequired, pod.Name)
			}
		}
		members, err := sglangSlotMembers(allocation.SlotID, pod)
		if err != nil {
			return domain.CapacityObservation{}, err
		}
		observation.Allocations = append(observation.Allocations, domain.CapacityAllocation{
			Incarnation: domain.ReplicaIncarnation{
				ReplicaID: allocation.ReplicaID, SlotID: allocation.SlotID, CapacityRefs: []domain.CapacityRef{allocation.Pod},
				Members: members,
			},
			Available: allocation.Ready, Health: allocation.Health,
		})
	}
	return observation, nil
}

// Apply validates native bootstrap and frozen processes before any physical capacity effect.
func (a *SGLangCapacity) Apply(ctx context.Context, groupID domain.GroupID, target domain.CapacityTarget) (domain.ApplyResult, error) {
	// Recovery is distinct from appending the fixed native-member range of a new allocation.
	for _, replica := range target.Replicas {
		if replica.Bootstrap == nil {
			continue
		}
		// Every allocation's member range is fixed by its GPU width and physical slot.
		members, err := (sglangruntime.ReplicaLayout{NativeMembersPerReplica: a.MembersPerReplica}).MembersForSlot(replica.SlotID)
		bootstrap := replica.Bootstrap
		if err != nil || replica.SlotID == "slot-0" || bootstrap.Mode != domain.BootstrapModeJoin ||
			bootstrap.BaseTopologyGeneration <= 0 || !slices.Equal(bootstrap.NativeMembers, members) {
			return domain.ApplyResult{Rejection: &domain.Failure{
				Classification: domain.FailureClassificationTerminal, Reason: "UnsupportedBootstrap",
				Message: "SGLang growth requires the complete fixed member range of one joining allocation",
			}}, nil
		}
	}

	// A matching Pod UID cannot hide a different or restarted native process.
	observed, err := a.Observe(ctx, groupID)
	if err != nil {
		return domain.ApplyResult{}, err
	}
	for _, replica := range target.Replicas {
		if replica.Incarnation == nil {
			continue
		}
		if !slices.ContainsFunc(observed.Allocations, func(allocation domain.CapacityAllocation) bool {
			return domain.SameIncarnation(allocation.Incarnation, *replica.Incarnation)
		}) {
			return domain.ApplyResult{}, fmt.Errorf("%w: SGLang capacity target lost an exact process binding", domain.ErrRecoveryRequired)
		}
	}
	return a.Physical.Apply(ctx, groupID, target)
}

func sglangSlotMembers(slot domain.CapacitySlotID, pod *corev1.Pod) ([]domain.NativeMemberIncarnation, error) {
	// Resolve physical width from the same explicit GPU allocation used by native automatic bootstrap.
	main, err := engineGroupMainContainer(pod)
	if err != nil {
		return nil, err
	}
	gpus := main.Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]
	width, exact := gpus.AsInt64()
	if !exact || width < 1 || width > 2147483647 {
		return nil, fmt.Errorf("SGLang allocation requires a positive whole-GPU width")
	}
	ids, err := (sglangruntime.ReplicaLayout{NativeMembersPerReplica: int32(width)}).MembersForSlot(slot)
	if err != nil {
		return nil, err
	}
	members := make([]domain.NativeMemberIncarnation, len(ids))
	for i, id := range ids {
		// Each rank is a distinct non-restarting process within this Pod lifetime.
		members[i] = domain.NativeMemberIncarnation{ID: id,
			RuntimeIncarnation: domain.RuntimeIncarnationID(fmt.Sprintf("%s/%s", pod.UID, id)),
		}
	}
	return members, nil
}
