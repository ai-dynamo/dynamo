/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package grovecapacity translates Engine Group capacity targets into Grove member-clique scaling.
package grovecapacity

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Adapter scales one PCSG-owned member clique with one Pod per logical allocation.
// Grove owns physical identity and placement, never native-member assignment or process identity.
// An exact physical identity never authorizes bootstrapping a replacement.
// The journal stores revision fencing and exact release receipts, not the pod-count target
// already persisted in Grove. Child-clique annotations
// cannot replace it: PCSG reconciliation resets them from the PCS template.
// The controller checkpoint owns the full target. This adapter journal is required
// until Grove can retain the same acceptance and release evidence.
type Adapter struct {
	Client    client.Client
	Clique    types.NamespacedName
	CliqueUID types.UID
	GroupName string
	Journal   kubejournal.Store
	// ReadinessScaleIn enables the single-victim steering workaround, not exact authorized deletion.
	ReadinessScaleIn bool
}

// Allocation is one physical Pod in a stable Grove slot, independent of the members hosted there.
type Allocation struct {
	ReplicaID enginegroup.ReplicaID
	SlotID    enginegroup.CapacitySlotID
	Pod       enginegroup.CapacityRef
	Ready     bool
	Health    enginegroup.AllocationHealth
}

// Observation reports only physical capacity. Engine integrations add native-process bindings.
type Observation struct {
	AppliedRevision int64
	Allocations     []Allocation
	ReleaseFences   []enginegroup.ReleaseFence
	ReleaseMode     enginegroup.CapacityReleaseMode
}

type journalState struct {
	AppliedRevision int64                      `json:"appliedRevision"`
	TargetDigest    string                     `json:"targetDigest"`
	Release         *releaseIntent             `json:"release,omitempty"`
	ReleaseFences   []enginegroup.ReleaseFence `json:"releaseFences,omitempty"`
}

// Observe derives allocation identities from Grove's stable per-clique slots, not Pod names.
func (a *Adapter) Observe(ctx context.Context, _ enginegroup.GroupID) (Observation, error) {
	// Refuse to reinterpret capacity after Grove recreates the bound clique.
	clique, err := a.clique(ctx)
	if err != nil {
		return Observation{}, err
	}
	pods, err := a.pods(ctx)
	if err != nil {
		return Observation{}, err
	}
	state := journalState{}
	_, err = a.Journal.Load(ctx, &state)
	if err != nil {
		return Observation{}, err
	}
	// Release proof is derived from durable acceptance and exact current UIDs without an observation-side write.
	fences, err := observeReleaseFences(state, pods, clique.Spec.Replicas)
	if err != nil {
		return Observation{}, err
	}

	// Pod readiness is a physical input; the engine integration determines serving capacity and member health.
	allocations := make([]Allocation, 0, len(pods))
	for i := range pods {
		allocation, err := a.allocation(&pods[i])
		if err != nil {
			return Observation{}, err
		}
		available := podAvailable(&pods[i])
		health := enginegroup.AllocationHealthUnknown
		if pods[i].Status.Phase == corev1.PodFailed {
			health = enginegroup.AllocationHealthFailed
		} else if available {
			health = enginegroup.AllocationHealthHealthy
		}
		allocation.Ready, allocation.Health = available, health
		allocations = append(allocations, allocation)
	}
	sort.Slice(allocations, func(i, j int) bool {
		return allocations[i].ReplicaID < allocations[j].ReplicaID
	})
	mode := enginegroup.CapacityReleaseUnsupported
	if a.ReadinessScaleIn {
		mode = enginegroup.CapacityReleaseReadinessSteered
	}
	return Observation{AppliedRevision: state.AppliedRevision, Allocations: allocations, ReleaseFences: fences, ReleaseMode: mode}, nil
}

// Apply accepts an absolute allocation target durably before writing the PodClique's Scale subresource.
// Replays repair count drift without changing the template or creating Pods directly.
// Native-member payloads are opaque; the caller validates engine/bootstrap semantics before this physical effect.
func (a *Adapter) Apply(
	ctx context.Context,
	_ enginegroup.GroupID,
	target enginegroup.CapacityTarget,
) (enginegroup.ApplyResult, error) {
	// No count write may stand in for exact authorized release.
	if len(target.ReleaseFences) != 0 && !a.ReadinessScaleIn {
		return rejected("ReleaseUnsupported", "the Grove growth adapter does not implement authorized release"), nil
	}
	if target.ControlRevision <= 0 || len(target.Replicas) == 0 {
		return rejected("InvalidCapacityTarget", "a positive revision and non-empty capacity target are required"), nil
	}
	if target.ProcessLifecycleOwner != enginegroup.ProcessLifecycleOwnerOrchestrator {
		return rejected("UnsupportedLifecycleOwner", "Grove capacity requires orchestrator-owned process lifecycle"), nil
	}
	encoded, err := json.Marshal(target)
	if err != nil {
		return enginegroup.ApplyResult{}, fmt.Errorf("marshal Grove capacity target: %w", err)
	}
	digestBytes := sha256.Sum256(encoded)
	digest := hex.EncodeToString(digestBytes[:])
	state := journalState{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if snapshot.Exists() && target.ControlRevision < state.AppliedRevision {
		return rejected("StaleCapacityRevision", "capacity revision is older than the accepted revision"), nil
	}
	if snapshot.Exists() && target.ControlRevision == state.AppliedRevision && digest != state.TargetDigest {
		return rejected("ConflictingCapacityRevision", "an accepted revision cannot change its payload"), nil
	}
	if len(target.ReleaseFences) != 0 {
		return a.applyReadinessRelease(ctx, target, digest, state, snapshot)
	}

	// Validate both the native binding and every retained or joining slot before acceptance.
	clique, err := a.clique(ctx)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	pods, err := a.pods(ctx)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	state.ReleaseFences, err = observeReleaseFences(state, pods, clique.Spec.Replicas)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if state.Release != nil && !hasReleaseFence(state.ReleaseFences, state.Release.Fence) {
		return rejected("ReleaseInProgress", "an accepted release must finish before a different capacity target"), nil
	}
	if failure := a.validateTarget(target, pods); failure != nil {
		return enginegroup.ApplyResult{Rejection: failure}, nil
	}
	wanted := int32(len(target.Replicas))
	scale := &autoscalingv1.Scale{}
	if err := a.Client.SubResource("scale").Get(ctx, clique, scale); err != nil {
		return enginegroup.ApplyResult{}, fmt.Errorf("get member-clique scale: %w", err)
	}
	if scale.UID != a.CliqueUID {
		return enginegroup.ApplyResult{}, fmt.Errorf("member-clique scale UID no longer matches the bound clique")
	}
	if scale.Spec.Replicas > wanted {
		return rejected("ReleaseUnsupported", "lowering a member-clique count requires authorized release"), nil
	}

	// Persist revision/digest acceptance before requesting Grove's absolute pod count.
	if !snapshot.Exists() || state.AppliedRevision != target.ControlRevision {
		state.AppliedRevision, state.TargetDigest, state.Release = target.ControlRevision, digest, nil
		state.ReleaseFences = releaseFencesAfterTarget(state.ReleaseFences, target)
		if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
			return enginegroup.ApplyResult{}, err
		}
	}
	if scale.Spec.Replicas == wanted {
		return enginegroup.ApplyResult{}, nil
	}

	// ResourceVersion prevents a scale write from racing clique recreation or another writer.
	scale.Spec.Replicas = wanted
	if err := a.Client.SubResource("scale").Update(ctx, clique, client.WithSubResourceBody(scale)); err != nil {
		return enginegroup.ApplyResult{}, fmt.Errorf("scale Grove member clique: %w", err)
	}
	return enginegroup.ApplyResult{}, nil
}

func (a *Adapter) clique(ctx context.Context) (*grovev1alpha1.PodClique, error) {
	// Name alone must never bind a recreated workload to an existing engine world.
	if a.CliqueUID == "" {
		return nil, fmt.Errorf("Grove member-clique UID is required")
	}
	clique := &grovev1alpha1.PodClique{}
	if err := a.Client.Get(ctx, a.Clique, clique); err != nil {
		return nil, fmt.Errorf("get Grove member clique: %w", err)
	}
	if clique.UID != a.CliqueUID || clique.DeletionTimestamp != nil {
		return nil, fmt.Errorf("Grove member clique was replaced or is terminating")
	}
	owner := metav1.GetControllerOf(clique)
	if owner == nil || owner.Kind != "PodCliqueScalingGroup" || owner.UID == "" ||
		owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() {
		return nil, fmt.Errorf("capacity must be a PCSG-owned member PodClique")
	}
	if clique.Labels[consts.KubeLabelDynamoEngineGroup] != a.GroupName || clique.Spec.ScaleConfig != nil {
		return nil, fmt.Errorf("member clique must belong to this Engine Group and have no competing autoscaler")
	}
	return clique, nil
}

func (a *Adapter) pods(ctx context.Context) ([]corev1.Pod, error) {
	// Select the complete native clique, including capacity missing Dynamo labels.
	pods := &corev1.PodList{}
	if err := a.Client.List(ctx, pods, client.InNamespace(a.Clique.Namespace),
		client.MatchingLabels{grovecommon.LabelPodClique: a.Clique.Name}); err != nil {
		return nil, fmt.Errorf("list Grove member-clique Pods: %w", err)
	}
	seen := make(map[enginegroup.CapacitySlotID]bool, len(pods.Items))
	for i := range pods.Items {
		allocation, err := a.allocation(&pods.Items[i])
		if err != nil {
			return nil, err
		}
		if seen[allocation.SlotID] {
			return nil, fmt.Errorf("multiple Pods occupy Grove slot %q", allocation.SlotID)
		}
		seen[allocation.SlotID] = true
	}
	return pods.Items, nil
}

func (a *Adapter) allocation(pod *corev1.Pod) (Allocation, error) {
	// Only Grove's authoritative slot label and exact Pod UID establish identity.
	owner := metav1.GetControllerOf(pod)
	if owner == nil || owner.Kind != "PodClique" || owner.Name != a.Clique.Name ||
		owner.UID != a.CliqueUID || owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() || pod.UID == "" ||
		pod.Labels[consts.KubeLabelDynamoEngineGroup] != a.GroupName ||
		pod.Labels[consts.KubeLabelDynamoScaleRepresentative] != consts.KubeLabelDynamoScaleRepresentativeYes {
		return Allocation{}, fmt.Errorf("Pod %s has an invalid Grove capacity binding", pod.Name)
	}
	literal := pod.Labels[grovecommon.LabelPodCliquePodIndex]
	index, err := strconv.ParseInt(literal, 10, 32)
	if err != nil || index < 0 || strconv.FormatInt(index, 10) != literal {
		return Allocation{}, fmt.Errorf("Pod %s has an invalid Grove slot index %q", pod.Name, literal)
	}
	return Allocation{
		ReplicaID: enginegroup.ReplicaID("replica-" + literal),
		SlotID:    enginegroup.CapacitySlotID("slot-" + literal),
		Pod:       enginegroup.CapacityRef{Name: pod.Name, UID: enginegroup.PodUID(pod.UID)},
	}, nil
}

func (a *Adapter) validateTarget(target enginegroup.CapacityTarget, pods []corev1.Pod) *enginegroup.Failure {
	// Grove growth allocates a contiguous suffix; sparse and replacement targets need a different contract.
	targets := make(map[enginegroup.ReplicaID]enginegroup.CapacityReplicaTarget, len(target.Replicas))
	for i := range target.Replicas {
		replica := target.Replicas[i]
		if _, duplicate := targets[replica.ReplicaID]; duplicate {
			return failure("InvalidCapacityTarget", "replica identities must be unique")
		}
		targets[replica.ReplicaID] = replica
	}
	for index := range target.Replicas {
		literal := strconv.Itoa(index)
		replica, present := targets[enginegroup.ReplicaID("replica-"+literal)]
		if !present || replica.SlotID != enginegroup.CapacitySlotID("slot-"+literal) {
			return failure("UnsupportedSlotMapping", "growth requires contiguous Grove slots matching replica ordinals")
		}
		if (replica.Incarnation == nil) == (replica.Bootstrap == nil) {
			return failure("InvalidCapacityTarget", "each replica requires exactly one incarnation or bootstrap intent")
		}
	}

	// Preserve exact retained UIDs and reject any unrequested allocation before touching /scale.
	present := make(map[enginegroup.ReplicaID]bool, len(pods))
	for i := range pods {
		allocation, err := a.allocation(&pods[i])
		if err != nil {
			return failure("InvalidCapacityBinding", err.Error())
		}
		replica, retained := targets[allocation.ReplicaID]
		if !retained {
			return failure("ReleaseUnsupported", "existing capacity is absent from the growth target")
		}
		if replica.Incarnation != nil && (replica.Incarnation.ReplicaID != allocation.ReplicaID ||
			replica.Incarnation.SlotID != allocation.SlotID || len(replica.Incarnation.CapacityRefs) != 1 || replica.Incarnation.CapacityRefs[0] != allocation.Pod) {
			return failure("MissingExactIncarnation", "a retained Grove Pod UID was replaced")
		}
		present[allocation.ReplicaID] = true
	}
	for _, replica := range target.Replicas {
		if replica.Incarnation != nil && !present[replica.ReplicaID] {
			return failure("MissingExactIncarnation", "an exact retained incarnation is missing; recovery is required")
		}
	}
	return nil
}

func podAvailable(pod *corev1.Pod) bool {
	// Readiness is allocation evidence only, never engine membership or admission evidence.
	if pod.DeletionTimestamp != nil {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func failure(reason, message string) *enginegroup.Failure {
	return &enginegroup.Failure{Classification: enginegroup.FailureClassificationTerminal, Reason: reason, Message: message}
}

func rejected(reason, message string) enginegroup.ApplyResult {
	return enginegroup.ApplyResult{Rejection: failure(reason, message)}
}
