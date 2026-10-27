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

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// VLLMGroveCapacity binds native process lifetimes to physical Grove allocations.
// Grove and Group must not be nil. Source is immutable; effects require lifecycle ownership.
type VLLMGroveCapacity struct {
	Grove  *grovecapacity.Adapter
	Group  *api.DynamoGraphDeploymentEngineGroup
	Source dynamo.VLLMProfileGeometrySource
}

// ReconcileInitialCapacity protects and records physical identities before membership reconciliation.
func (a *VLLMGroveCapacity) ReconcileInitialCapacity(ctx context.Context, groupID domain.GroupID) error {
	// Lifecycle effects remain scoped to the exact logical group.
	if groupID == "" || groupID != domain.GroupID(a.Group.UID) {
		return fmt.Errorf("initial capacity belongs to another Engine Group")
	}
	return a.ReconcileAllocations(ctx)
}

// Observe remains read-only and reports allocation readiness independently of native API readiness.
func (a *VLLMGroveCapacity) Observe(ctx context.Context, groupID domain.GroupID) (domain.CapacityObservation, error) {
	if groupID != domain.GroupID(a.Group.UID) || groupID == "" {
		return domain.CapacityObservation{}, fmt.Errorf("vLLM capacity observation belongs to a different Engine Group")
	}
	bindings, err := a.capacityBindings()
	if err != nil {
		return domain.CapacityObservation{}, err
	}
	physical, err := a.Grove.Observe(ctx, groupID)
	if err != nil {
		return domain.CapacityObservation{}, err
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(a.Source)
	if err != nil {
		return domain.CapacityObservation{}, err
	}
	observation := domain.CapacityObservation{AppliedRevision: physical.AppliedRevision, ReleaseFences: physical.ReleaseFences}

	// The engine integration owns process bindings; Grove reports only the physical Pod and slot.
	for _, allocation := range physical.Allocations {
		ref := allocation.Pod
		pod := &corev1.Pod{}
		if err := a.Grove.Client.Get(ctx, client.ObjectKey{Namespace: a.Group.Namespace, Name: ref.Name}, pod); err != nil {
			return domain.CapacityObservation{}, err
		}
		if domain.PodUID(pod.UID) != ref.UID {
			return domain.CapacityObservation{}, fmt.Errorf("%w: vLLM Pod changed during availability observation", domain.ErrRecoveryRequired)
		}
		members, err := resolveVLLMLaunchMembers(pod, allocation.SlotID, profile.NativeMembersPerReplica)
		if err != nil {
			return domain.CapacityObservation{}, err
		}
		observed := domain.CapacityAllocation{
			Incarnation: domain.ReplicaIncarnation{ReplicaID: allocation.ReplicaID, SlotID: allocation.SlotID,
				CapacityRefs: []domain.CapacityRef{ref}, Members: members},
			Available: allocation.Ready, Health: allocation.Health,
		}
		// Start native preparation only after the main process can bootstrap, not while its image is pulling.
		observed.BootstrapPending = pod.Status.PodIP == "" || !slices.ContainsFunc(pod.Status.ContainerStatuses, func(status corev1.ContainerStatus) bool {
			return status.Name == consts.MainContainerName && status.State.Running != nil
		})
		if uid, found := bindings.Slots[allocation.SlotID]; found && uid != ref.UID {
			return domain.CapacityObservation{}, fmt.Errorf("%w: vLLM slot %s changed its Pod UID", domain.ErrRecoveryRequired, allocation.SlotID)
		}
		observation.Allocations = append(observation.Allocations, observed)
	}
	return observation, nil
}

// Apply reasserts absolute Grove capacity while preserving exact process and release bindings.
func (a *VLLMGroveCapacity) Apply(ctx context.Context, groupID domain.GroupID, target domain.CapacityTarget) (domain.ApplyResult, error) {
	// Validate the immutable source before permitting a physical allocation effect.
	if groupID != domain.GroupID(a.Group.UID) || groupID == "" || !slices.Contains(a.Group.Finalizers, engineGroupFinalizer) {
		return domain.ApplyResult{}, fmt.Errorf("vLLM capacity effects require the exact lifecycle-owned Engine Group")
	}
	if _, err := a.capacityBindings(); err != nil {
		return domain.ApplyResult{}, err
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(a.Source)
	if err != nil {
		return domain.ApplyResult{}, err
	}
	if target.ProfileFingerprint != profile.Geometry.Fingerprint ||
		a.Group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] != profile.Geometry.Fingerprint ||
		len(target.Replicas) > int(profile.MaximumReplicas) {
		return domain.ApplyResult{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal, Reason: "InvalidVLLMCapacityTarget",
			Message: "capacity target does not match the immutable vLLM profile or reserved maximum",
		}}, nil
	}
	// Validate member ranges and pinned processes before any physical allocation or release.
	if failure := validateVLLMCapacityBootstrap(target, profile); failure != nil {
		return domain.ApplyResult{Rejection: failure}, nil
	}
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
			return domain.ApplyResult{}, fmt.Errorf("%w: vLLM capacity target lost an exact process binding", domain.ErrRecoveryRequired)
		}
	}
	result, err := a.Grove.Apply(ctx, groupID, target)
	if err != nil || result.Rejection != nil {
		return result, err
	}
	return result, a.ReconcileAllocations(ctx)
}

// ReconcileAllocations protects and records the physical slots without delivering executable commands.
func (a *VLLMGroveCapacity) ReconcileAllocations(ctx context.Context) error {
	// Native startup is autonomous, while controller effects still require lifecycle ownership.
	if !slices.Contains(a.Group.Finalizers, engineGroupFinalizer) {
		return fmt.Errorf("%w: vLLM capacity is waiting for Engine Group lifecycle ownership", ErrRuntimeUnavailable)
	}
	if _, err := a.capacityBindings(); err != nil {
		return err
	}
	if err := a.Grove.InitializeRetention(ctx); err != nil {
		return err
	}
	physical, err := a.Grove.Observe(ctx, domain.GroupID(a.Group.UID))
	if err != nil {
		return err
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(a.Source)
	if err != nil {
		return err
	}

	// A new occupant is observed as a new incarnation, never mistaken for a stopped or retired process.
	for _, allocation := range physical.Allocations {
		pod := &corev1.Pod{}
		if err := a.Grove.Client.Get(ctx, client.ObjectKey{Namespace: a.Group.Namespace, Name: allocation.Pod.Name}, pod); err != nil {
			return err
		}
		if domain.PodUID(pod.UID) != allocation.Pod.UID || pod.Spec.RestartPolicy != corev1.RestartPolicyNever {
			return fmt.Errorf("%w: vLLM allocation lifetime or restart policy changed", domain.ErrRecoveryRequired)
		}
		if _, err := resolveVLLMLaunchMembers(pod, allocation.SlotID, profile.NativeMembersPerReplica); err != nil {
			return err
		}
	}
	return a.bindAllocations(ctx, physical.Allocations)
}

func validateVLLMCapacityBootstrap(target domain.CapacityTarget, profile dynamo.ResolvedVLLMElasticEPProfile) *domain.Failure {
	// Grove treats native IDs as opaque; this launcher accepts only a complete profile-defined append assignment.
	layout := vllmruntime.ReplicaLayout{NativeMembersPerReplica: profile.NativeMembersPerReplica}
	for _, replica := range target.Replicas {
		bootstrap := replica.Bootstrap
		if bootstrap == nil {
			continue
		}
		index, err := strconv.ParseInt(strings.TrimPrefix(string(replica.SlotID), "slot-"), 10, 32)
		if err == nil && index >= int64(profile.InitialReplicas) && bootstrap.Mode == domain.BootstrapModeJoin && bootstrap.BaseTopologyGeneration > 0 {
			ids, layoutErr := layout.Members(int32(index))
			actual := slices.Clone(bootstrap.NativeMembers)
			slices.Sort(ids)
			slices.Sort(actual)
			if layoutErr == nil && slices.Equal(ids, actual) {
				continue
			}
		}
		return &domain.Failure{Classification: domain.FailureClassificationTerminal, Reason: "UnsupportedVLLMBootstrap",
			Message: "vLLM allocation bootstrap must append its complete profile-defined native-member assignment"}
	}
	return nil
}

func resolveVLLMLaunchMembers(pod *corev1.Pod, slot domain.CapacitySlotID, width int32) ([]domain.NativeMemberIncarnation, error) {
	// Native children are not independently restarted in this profile; each member is pinned to its Pod lifetime and rank.
	for _, container := range pod.Status.ContainerStatuses {
		if container.Name == consts.MainContainerName && container.RestartCount > 0 {
			return nil, fmt.Errorf("%w: vLLM Pod %s restarted outside membership recovery", domain.ErrRecoveryRequired, pod.Name)
		}
	}
	index, err := strconv.ParseInt(strings.TrimPrefix(string(slot), "slot-"), 10, 32)
	if err != nil {
		return nil, err
	}
	ids, err := (vllmruntime.ReplicaLayout{NativeMembersPerReplica: width}).Members(int32(index))
	if err != nil {
		return nil, err
	}
	members := make([]domain.NativeMemberIncarnation, len(ids))
	for i, id := range ids {
		incarnation := domain.RuntimeIncarnationID(pod.UID)
		if width > 1 {
			incarnation += "/" + domain.RuntimeIncarnationID(id)
		}
		members[i] = domain.NativeMemberIncarnation{ID: id, RuntimeIncarnation: incarnation}
	}
	return members, nil
}

var _ domain.CapacityAdapter = (*VLLMGroveCapacity)(nil)
