/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// SGLangGroveCapacity binds the growth-only profile's process identity to Grove capacity.
// The native allocation is one process tree; container restarts require explicit group recovery.
type SGLangGroveCapacity struct {
	Grove             *grovecapacity.Adapter
	MembersPerReplica int32
}

// Observe binds each allocation's fixed member range to its non-restarting process tree.
func (a *SGLangGroveCapacity) Observe(ctx context.Context, groupID domain.GroupID) (domain.CapacityObservation, error) {
	// Grove owns Pod identity and placement; the engine integration owns process lifetime and rank assignment.
	physical, err := a.Grove.Observe(ctx, groupID)
	if err != nil {
		return domain.CapacityObservation{}, err
	}
	observation := domain.CapacityObservation{AppliedRevision: physical.AppliedRevision, ReleaseFences: physical.ReleaseFences}
	for _, allocation := range physical.Allocations {
		pod := &corev1.Pod{}
		if err := a.Grove.Client.Get(ctx, client.ObjectKey{Namespace: a.Grove.Clique.Namespace, Name: allocation.Pod.Name}, pod); err != nil {
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
func (a *SGLangGroveCapacity) Apply(ctx context.Context, groupID domain.GroupID, target domain.CapacityTarget) (domain.ApplyResult, error) {
	// Recovery is distinct from appending the fixed native-member range of a new allocation.
	for _, replica := range target.Replicas {
		if replica.Bootstrap == nil {
			continue
		}
		// Every allocation's member range is fixed by its GPU width and Grove slot.
		members, err := sglangMembersForSlot(replica.SlotID, a.MembersPerReplica)
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
	return a.Grove.Apply(ctx, groupID, target)
}

func sglangMembersForSlot(slot domain.CapacitySlotID, width int32) ([]domain.NativeMemberID, error) {
	// Stable Grove slots map to non-overlapping native rank ranges.
	literal := strings.TrimPrefix(string(slot), "slot-")
	index, err := strconv.ParseInt(literal, 10, 32)
	if err != nil || index < 0 || string(slot) != "slot-"+strconv.FormatInt(index, 10) {
		return nil, fmt.Errorf("invalid SGLang allocation slot %q", slot)
	}
	if width < 1 {
		return nil, fmt.Errorf("SGLang allocation width must be positive")
	}
	members := make([]domain.NativeMemberID, width)
	for local := range members {
		members[local] = domain.NativeMemberID(fmt.Sprintf("dp-%d", index*int64(width)+int64(local)))
	}
	slices.Sort(members)
	return members, nil
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
	ids, err := sglangMembersForSlot(slot, int32(width))
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
