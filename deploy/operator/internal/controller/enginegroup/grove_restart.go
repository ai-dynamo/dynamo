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
	"strconv"
	"strings"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const engineGroupProcessFinalizer = "nvidia.com/engine-group-process-fence"

// groveWorldFence retains kubelet-confirmed stop evidence after Grove deletes Pods.
// An API deletion, missing endpoint, NotReady node, or replacement UID is not stop evidence.
// The growth profile uses restartPolicy Never; live-member recovery and infrastructure
// fencing are deliberately not inferred by this whole-world restart path.
type groveWorldFence struct {
	CliqueUID           types.UID
	PreviousUID         types.UID
	Pods                []grovePodFence
	VerificationFailure *domain.Failure
}

type grovePodFence struct {
	Ref     domain.CapacityRef
	SlotID  domain.CapacitySlotID
	Stopped bool
}

func groveWorldFenceStore(kubeClient client.Client, group *api.DynamoGraphDeploymentEngineGroup, uid types.UID) kubejournal.Store {
	return kubejournal.NewStore(kubeClient, group.Namespace, group.Name, group.UID, "grove-fence-"+string(uid))
}

// reconcileGroveWorld pauses normal effects until the current binding is protected
// or a replacement has passed a durable, exact-UID stop fence. true means requeue.
func (r *Reconciler) reconcileGroveWorld(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (bool, error) {
	// Only the concrete Grove realization has this physical lifecycle boundary.
	name := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique]
	uid := types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	if name == "" || uid == "" || group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != consts.KubeLabelDynamoEngineGroupSGLang {
		return false, nil
	}
	if !controllerutil.ContainsFinalizer(group, engineGroupFinalizer) {
		return false, nil
	}
	clique := &grove.PodClique{}
	err := r.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: name}, clique)
	if err != nil && !apierrors.IsNotFound(err) {
		return true, fmt.Errorf("observe bound Grove clique: %w", err)
	}
	sourcePresent := err == nil && clique.UID == uid

	// Persist stop observations before letting Pod garbage collection erase them.
	fence, changed, err := r.reconcileGroveProcessFences(ctx, group, name, uid)
	if err != nil {
		return true, err
	}
	if sourcePresent && clique.DeletionTimestamp.IsZero() {
		return changed, nil
	}
	if sourcePresent || !group.DeletionTimestamp.IsZero() {
		return true, nil
	}
	if err := r.validateStoppedGroveWorld(ctx, group, fence); err != nil {
		return true, err
	}

	// Rebinding is restricted to the same logical world under the exact DGD-owned PCS.
	candidate, err := r.resolveReplacementGroveClique(ctx, group)
	if err != nil || candidate == nil || !candidate.DeletionTimestamp.IsZero() {
		return true, err
	}
	if candidate.UID == uid {
		return true, nil
	}
	nextGroup := group.DeepCopy()
	nextGroup.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] = candidate.Name
	nextGroup.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = string(candidate.UID)
	runtime, err := r.RuntimeProvider.Resolve(ctx, nextGroup)
	if err != nil {
		return true, err
	}
	if err := validateEngineGroupRuntime(runtime); err != nil {
		return true, err
	}
	if group.Status.Profile != nil && runtime.Profile.Fingerprint != group.Status.Profile.Fingerprint {
		return true, errors.New("replacement Grove world has a different immutable profile")
	}

	// The predecessor record authorizes only this exact new lifetime, before the binding write.
	store := groveWorldFenceStore(r.Client, group, candidate.UID)
	var nextFence groveWorldFence
	snapshot, err := store.Load(ctx, &nextFence)
	if err != nil {
		return true, err
	}
	if snapshot.Exists() && (nextFence.CliqueUID != candidate.UID || nextFence.PreviousUID != uid) {
		return true, errors.New("replacement Grove world has conflicting restart authority")
	}
	if !snapshot.Exists() {
		nextFence = groveWorldFence{CliqueUID: candidate.UID, PreviousUID: uid}
		if _, err := store.Save(ctx, snapshot, nextFence); err != nil {
			return true, err
		}
	}
	if err := r.Patch(ctx, nextGroup, client.MergeFromWithOptions(group, client.MergeFromWithOptimisticLock{})); err != nil {
		return true, fmt.Errorf("accept restarted Grove world binding: %w", err)
	}
	*group = *nextGroup
	return true, nil
}

// authorizeGroveCheckpointRestart proves the handoff even after the binding update
// succeeded but controller initialization or public status publication failed.
func (r *Reconciler) authorizeGroveCheckpointRestart(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, previousUID types.UID) error {
	uid := types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	if uid == "" || previousUID == "" {
		return errors.New("checkpoint binding changed without Grove restart authority")
	}
	var current, previous groveWorldFence
	snapshot, err := groveWorldFenceStore(r.Client, group, uid).Load(ctx, &current)
	if err != nil {
		return err
	}
	if !snapshot.Exists() || current.CliqueUID != uid || current.PreviousUID != previousUID {
		return errors.New("checkpoint binding changed without correlated Grove restart authority")
	}
	snapshot, err = groveWorldFenceStore(r.Client, group, previousUID).Load(ctx, &previous)
	if err != nil {
		return err
	}
	if !snapshot.Exists() || previous.CliqueUID != previousUID {
		return errors.New("previous Grove world has no durable process fence")
	}
	return r.validateStoppedGroveWorld(ctx, group, previous)
}

func (r *Reconciler) initializeRestartedEngineGroup(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, runtime Runtime, store kubejournal.Store, snapshot kubejournal.Snapshot) (ctrl.Result, error) {
	// A terminal formation failure cannot disappear during an operator restart or an implicit retry.
	fenceStore := groveWorldFenceStore(r.Client, group, types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]))
	var fence groveWorldFence
	fenceSnapshot, err := fenceStore.Load(ctx, &fence)
	if err != nil {
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	if fence.VerificationFailure != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("restarted world requires recreation after terminal verification failure: %s", fence.VerificationFailure.Reason))
	}

	// New formation is observation, not an invented membership resize of the old topology.
	observations, err := observeEngineGroupRuntime(ctx, runtime, engineGroupID(group), "")
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("observe restarted world: %w", err))
	}
	_, err = domain.NewGroupStatus(observations.membership.CommittedTopology, observations.capacity, observations.traffic)
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("initialize restarted world: %w", err))
	}
	if observations.membership.CommittedTopology.ReplicaCount() < runtime.Profile.MinSupportedReplicas ||
		observations.membership.CommittedTopology.NativeMemberCount() < runtime.Profile.MinSafeServingNativeMembers {
		return r.reconcileUnavailableRuntime(ctx, group, errors.New("restarted world has not formed its initial serving membership"))
	}
	verified, err := runtime.Verifier.Verify(ctx, engineGroupID(group), observations.membership.CommittedTopology)
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("verify restarted world: %w", err))
	}
	if verified.Failure != nil && verified.Failure.Classification == domain.FailureClassificationTerminal {
		fence.VerificationFailure = verified.Failure
		if _, err := fenceStore.Save(ctx, fenceSnapshot, fence); err != nil {
			return r.reconcileCheckpointFailure(ctx, group, err)
		}
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("restarted world failed serving verification: %s", verified.Failure.Reason))
	}
	if verified.Proof == nil || verified.Failure != nil ||
		verified.Proof.ObservedAt.IsZero() ||
		verified.Proof.TopologyGeneration != observations.membership.CommittedTopology.Generation ||
		verified.Proof.RuntimeDigest != domain.TopologyRuntimeDigest(observations.membership.CommittedTopology) {
		return r.reconcileUnavailableRuntime(ctx, group, errors.New("restarted world has no matching serving proof"))
	}

	// Revalidate exact membership after the probe; an unrelated incarnation cannot inherit its proof.
	fresh, err := observeEngineGroupRuntime(ctx, runtime, engineGroupID(group), "")
	if err != nil || !domain.SameTopology(observations.membership.CommittedTopology, fresh.membership.CommittedTopology) {
		return r.reconcileUnavailableRuntime(ctx, group, errors.Join(errors.New("restarted membership changed during serving verification"), err))
	}
	state, err := domain.NewGroupStatus(fresh.membership.CommittedTopology, fresh.capacity, fresh.traffic)
	if err != nil {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf("revalidate restarted world: %w", err))
	}
	if !allEngineGroupMembersAvailable(state.Membership.Observed.CommittedTopology, state.Registry, state.Capacity.Observed) ||
		!allEngineGroupMembersAdmitted(state.Membership.Observed.CommittedTopology, state.Traffic.Observed) {
		return r.reconcileUnavailableRuntime(ctx, group, errors.New("restarted world is not fully available and admitted"))
	}

	// Replace only lifetime-scoped evidence; desired replicas, policy, identity and immutable profile survive.
	before := group.Status.DeepCopy()
	group.Status = api.DynamoGraphDeploymentEngineGroupStatus{Profile: runtime.Profile.DeepCopy()}
	reconcileEngineGroupDesiredAssignment(group, runtime.Profile, state, nil)
	r.projectEngineGroupStatus(group, runtime.Profile, state, nil, nil)
	if _, err := store.Save(ctx, snapshot, newEngineGroupCheckpoint(group, state)); err != nil {
		group.Status = *before
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}

func (r *Reconciler) reconcileGroveProcessFences(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, cliqueName string, uid types.UID) (groveWorldFence, bool, error) {
	// This journal is scoped to the physical clique, so old tombstones cannot authorize new Pods.
	store := groveWorldFenceStore(r.Client, group, uid)
	var fence groveWorldFence
	snapshot, err := store.Load(ctx, &fence)
	if err != nil {
		return fence, false, err
	}
	if snapshot.Exists() && fence.CliqueUID != uid {
		return fence, false, errors.New("Grove process fence belongs to a different clique")
	}
	previous := fence
	fence.Pods = slices.Clone(fence.Pods)
	fence.CliqueUID = uid

	// Include native capacity even when a Pod lost its Dynamo label; never adopt another clique UID.
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(group.Namespace), client.MatchingLabels{grovecommon.LabelPodClique: cliqueName}); err != nil {
		return fence, false, fmt.Errorf("observe Grove process lifetimes: %w", err)
	}
	owned := make([]*corev1.Pod, 0, len(pods.Items))
	for i := range pods.Items {
		pod := &pods.Items[i]
		owner := metav1.GetControllerOf(pod)
		if owner == nil || owner.UID != uid {
			continue
		}
		if owner.Kind != "PodClique" || owner.APIVersion != grove.SchemeGroupVersion.String() || owner.Name != cliqueName ||
			pod.UID == "" || pod.Spec.RestartPolicy != corev1.RestartPolicyNever ||
			pod.Labels[consts.KubeLabelDynamoEngineGroup] != group.Name {
			return fence, false, fmt.Errorf("Pod %s has an invalid Grove process fence binding", pod.Name)
		}
		literal := pod.Labels[grovecommon.LabelPodCliquePodIndex]
		index, err := strconv.ParseInt(literal, 10, 32)
		if err != nil || index < 0 || strconv.FormatInt(index, 10) != literal {
			return fence, false, fmt.Errorf("Pod %s has an invalid Grove slot index", pod.Name)
		}
		ref := domain.CapacityRef{Name: pod.Name, UID: domain.PodUID(pod.UID)}
		recordIndex := slices.IndexFunc(fence.Pods, func(record grovePodFence) bool { return record.Ref == ref })
		if recordIndex < 0 {
			fence.Pods = append(fence.Pods, grovePodFence{Ref: ref, SlotID: domain.CapacitySlotID("slot-" + literal)})
			recordIndex = len(fence.Pods) - 1
		}
		if podProcessesStopped(pod) {
			fence.Pods[recordIndex].Stopped = true
		}
		owned = append(owned, pod)
	}
	slices.SortFunc(fence.Pods, func(a, b grovePodFence) int { return strings.Compare(string(a.Ref.UID), string(b.Ref.UID)) })
	if !snapshot.Exists() || !reflect.DeepEqual(previous, fence) {
		if _, err := store.Save(ctx, snapshot, fence); err != nil {
			return fence, false, err
		}
	}

	// Protect observed processes before native membership work; release only after durable stop proof.
	changed := false
	for _, pod := range owned {
		before := pod.DeepCopy()
		if pod.DeletionTimestamp.IsZero() {
			if !controllerutil.AddFinalizer(pod, engineGroupProcessFinalizer) {
				continue
			}
		} else {
			if !podProcessesStopped(pod) || !controllerutil.RemoveFinalizer(pod, engineGroupProcessFinalizer) {
				continue
			}
		}
		if err := r.Patch(ctx, pod, client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{})); err != nil {
			return fence, changed, fmt.Errorf("persist process fence for Pod %s: %w", pod.Name, err)
		}
		changed = true
	}
	return fence, changed, nil
}

// podProcessesStopped requires terminal kubelet state, not inferred death from API absence.
func podProcessesStopped(pod *corev1.Pod) bool {
	if pod.Spec.RestartPolicy != corev1.RestartPolicyNever ||
		(pod.Status.Phase != corev1.PodSucceeded && pod.Status.Phase != corev1.PodFailed) {
		return false
	}
	for _, container := range pod.Spec.Containers {
		index := slices.IndexFunc(pod.Status.ContainerStatuses, func(status corev1.ContainerStatus) bool { return status.Name == container.Name })
		if index < 0 || pod.Status.ContainerStatuses[index].State.Terminated == nil ||
			pod.Status.ContainerStatuses[index].State.Terminated.Reason == "ContainerStatusUnknown" {
			return false
		}
	}
	return len(pod.Spec.Containers) > 0
}

func (r *Reconciler) validateStoppedGroveWorld(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, fence groveWorldFence) error {
	// Retain ambiguous or missing process identities, including ones GC removed before protection was installed.
	if len(fence.Pods) == 0 {
		return errors.New("old Grove world has no process-stop evidence")
	}
	for _, record := range fence.Pods {
		if !record.Stopped {
			return fmt.Errorf("old Grove Pod %s (UID %s) is not proven stopped", record.Ref.Name, record.Ref.UID)
		}
	}
	var checkpoint engineGroupCheckpoint
	snapshot, err := engineGroupCheckpointStore(r, group).Load(ctx, &checkpoint)
	if err != nil {
		return err
	}
	if !snapshot.Exists() {
		if group.Status.Topology != nil || group.Status.Operation != nil {
			return errors.New("initialized Grove world has no checkpoint")
		}
		return nil
	}
	if err := validateEngineGroupCheckpoint(checkpoint, group, fence.CliqueUID); err != nil {
		return err
	}

	// Every exact allocated incarnation and every potentially dispatched bootstrap must be accounted for.
	var refs []domain.CapacityRef
	for _, record := range checkpoint.State.Registry.Replicas {
		if record.Current == nil {
			continue
		}
		refs = append(refs, record.Current.CapacityRefs...)
	}
	for _, allocation := range checkpoint.State.Capacity.Observed.Allocations {
		refs = append(refs, allocation.Incarnation.CapacityRefs...)
	}
	for _, target := range []*domain.CapacityTarget{checkpoint.State.Capacity.Desired, checkpoint.State.Capacity.Accepted} {
		if target == nil {
			continue
		}
		for _, replica := range target.Replicas {
			if replica.Incarnation != nil {
				refs = append(refs, replica.Incarnation.CapacityRefs...)
			}
			if replica.Bootstrap != nil && !slices.ContainsFunc(fence.Pods, func(pod grovePodFence) bool { return pod.SlotID == replica.SlotID && pod.Stopped }) {
				return fmt.Errorf("old bootstrap slot %s has no stop proof", replica.SlotID)
			}
		}
	}
	for _, ref := range refs {
		if !slices.ContainsFunc(fence.Pods, func(pod grovePodFence) bool { return pod.Ref == ref && pod.Stopped }) {
			return fmt.Errorf("old allocated Pod %s (UID %s) has no stop proof", ref.Name, ref.UID)
		}
	}
	return nil
}

func (r *Reconciler) resolveReplacementGroveClique(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (*grove.PodClique, error) {
	// Recovery may not adopt a same-named PCS owned by another DGD lifetime.
	owner := metav1.GetControllerOf(group)
	if owner == nil || owner.Kind != "DynamoGraphDeployment" || owner.APIVersion != api.GroupVersion.String() {
		return nil, errors.New("Grove world restart requires a DGD-owned Engine Group")
	}
	dgd := &api.DynamoGraphDeployment{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: owner.Name}, dgd); err != nil {
		return nil, client.IgnoreNotFound(err)
	}
	if dgd.UID != owner.UID || !dgd.DeletionTimestamp.IsZero() {
		return nil, errors.New("Grove world restart owner changed or is terminating")
	}
	pcs := &grove.PodCliqueSet{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components)}, pcs); err != nil {
		return nil, client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(pcs, dgd) || !pcs.DeletionTimestamp.IsZero() {
		return nil, errors.New("replacement Grove PCS has conflicting ownership or is terminating")
	}
	index := group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex]
	if index != "0" {
		return nil, errors.New("Grove world restart currently requires logical world index zero")
	}
	return resolveGroveMemberClique(ctx, r.Client, group, pcs, 0)
}
