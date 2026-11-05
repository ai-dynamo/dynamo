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
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const engineGroupProcessFinalizer = "nvidia.com/engine-group-process-fence"

func groveWorldFenceStore(kubeClient client.Client, group *api.DynamoGraphDeploymentEngineGroup, uid types.UID) kubejournal.Store {
	return kubejournal.NewStore(kubeClient, group.Namespace, group.Name, group.UID, "grove-fence-"+string(uid))
}

// Reconcile protects the current physical world and proposes a fenced successor without accepting it.
func (p *GroveWorkloadProvider) Reconcile(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (*api.DynamoGraphDeploymentEngineGroup, bool, error) {
	// Only the concrete Grove realization has this physical lifecycle boundary.
	name := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique]
	uid := types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	if name == "" || uid == "" {
		return nil, false, nil
	}
	if !controllerutil.ContainsFinalizer(group, engineGroupFinalizer) {
		return nil, false, nil
	}
	clique := &grove.PodClique{}
	err := p.Client.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: name}, clique)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, true, fmt.Errorf("observe bound Grove clique: %w", err)
	}
	sourcePresent := err == nil && clique.UID == uid

	// Persist stop observations before letting Pod garbage collection erase them.
	fence, changed, err := p.reconcileGroveProcessFences(ctx, group, name, uid)
	if err != nil {
		return nil, true, err
	}
	if sourcePresent && clique.DeletionTimestamp.IsZero() {
		return nil, changed, nil
	}
	if sourcePresent || !group.DeletionTimestamp.IsZero() {
		return nil, true, nil
	}
	if err := p.validateStoppedGroveWorld(ctx, group, fence); err != nil {
		return nil, true, err
	}

	// Rebinding is restricted to the same logical world under the exact DGD-owned PCS.
	candidate, err := p.resolveReplacementGroveClique(ctx, group)
	if err != nil || candidate == nil || !candidate.DeletionTimestamp.IsZero() {
		return nil, true, err
	}
	if candidate.UID == uid {
		return nil, true, nil
	}
	nextGroup := group.DeepCopy()
	nextGroup.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] = candidate.Name
	nextGroup.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = string(candidate.UID)
	return nextGroup, true, nil
}

// Accept rechecks physical authority before persisting a successor approved by engine resolution.
func (p *GroveWorkloadProvider) Accept(ctx context.Context, group, nextGroup *api.DynamoGraphDeploymentEngineGroup) error {
	// A new observation must still propose exactly the same lifetime and preserve the complete declaration.
	candidate, _, err := p.Reconcile(ctx, group)
	if err != nil {
		return err
	}
	if candidate == nil || !reflect.DeepEqual(candidate, nextGroup) {
		return errors.New("replacement Grove binding changed before acceptance")
	}
	uid := types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	nextUID := types.UID(nextGroup.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])

	// The predecessor record authorizes only this exact new lifetime, before the binding write.
	store := groveWorldFenceStore(p.Client, group, nextUID)
	var nextFence worldFence
	snapshot, err := store.Load(ctx, &nextFence)
	if err != nil {
		return err
	}
	if snapshot.Exists() && (nextFence.WorldID != nextUID || nextFence.PreviousID != uid) {
		return errors.New("replacement Grove world has conflicting restart authority")
	}
	if !snapshot.Exists() {
		nextFence = worldFence{WorldID: nextUID, PreviousID: uid}
		if _, err := store.Save(ctx, snapshot, nextFence); err != nil {
			return err
		}
	}
	if err := p.Client.Patch(ctx, nextGroup, client.MergeFromWithOptions(group, client.MergeFromWithOptimisticLock{})); err != nil {
		return fmt.Errorf("accept restarted Grove world binding: %w", err)
	}
	*group = *nextGroup
	return nil
}

// authorizeGroveCheckpointRestart proves the handoff even after the binding update
// succeeded but controller initialization or public status publication failed.
func (p *GroveWorkloadProvider) AuthorizeRestart(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, previousUID types.UID) error {
	uid := types.UID(group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	if uid == "" || previousUID == "" {
		return errors.New("checkpoint binding changed without Grove restart authority")
	}
	var current, previous worldFence
	snapshot, err := groveWorldFenceStore(p.Client, group, uid).Load(ctx, &current)
	if err != nil {
		return err
	}
	if !snapshot.Exists() || current.WorldID != uid || current.PreviousID != previousUID {
		return errors.New("checkpoint binding changed without correlated Grove restart authority")
	}
	snapshot, err = groveWorldFenceStore(p.Client, group, previousUID).Load(ctx, &previous)
	if err != nil {
		return err
	}
	if !snapshot.Exists() || previous.WorldID != previousUID {
		return errors.New("previous Grove world has no durable process fence")
	}
	return p.validateStoppedGroveWorld(ctx, group, previous)
}

func (p *GroveWorkloadProvider) reconcileGroveProcessFences(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, cliqueName string, uid types.UID) (worldFence, bool, error) {
	// This journal is scoped to the physical clique, so old tombstones cannot authorize new Pods.
	store := groveWorldFenceStore(p.Client, group, uid)
	var fence worldFence
	snapshot, err := store.Load(ctx, &fence)
	if err != nil {
		return fence, false, err
	}
	if snapshot.Exists() && fence.WorldID != uid {
		return fence, false, errors.New("Grove process fence belongs to a different clique")
	}
	previous := fence
	fence.Pods = slices.Clone(fence.Pods)
	fence.WorldID = uid

	// Include native capacity even when a Pod lost its Dynamo label; never adopt another clique UID.
	pods := &corev1.PodList{}
	if err := p.Client.List(ctx, pods, client.InNamespace(group.Namespace), client.MatchingLabels{grovecommon.LabelPodClique: cliqueName}); err != nil {
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
		recordIndex := slices.IndexFunc(fence.Pods, func(record worldPodFence) bool { return record.Ref == ref })
		if recordIndex < 0 {
			fence.Pods = append(fence.Pods, worldPodFence{Ref: ref, SlotID: domain.CapacitySlotID("slot-" + literal)})
			recordIndex = len(fence.Pods) - 1
		}
		if podProcessesStopped(pod) {
			fence.Pods[recordIndex].Stopped = true
		}
		owned = append(owned, pod)
	}
	slices.SortFunc(fence.Pods, func(a, b worldPodFence) int { return strings.Compare(string(a.Ref.UID), string(b.Ref.UID)) })
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
		if err := p.Client.Patch(ctx, pod, client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{})); err != nil {
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

func (p *GroveWorkloadProvider) validateStoppedGroveWorld(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, fence worldFence) error {
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
	snapshot, err := engineGroupCheckpointStore(p.Client, group).Load(ctx, &checkpoint)
	if err != nil {
		return err
	}
	if !snapshot.Exists() {
		if group.Status.Topology != nil || group.Status.Operation != nil {
			return errors.New("initialized Grove world has no checkpoint")
		}
		return nil
	}
	if err := validateEngineGroupCheckpoint(checkpoint, group, fence.WorldID); err != nil {
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
			if replica.Bootstrap != nil && !slices.ContainsFunc(fence.Pods, func(pod worldPodFence) bool { return pod.SlotID == replica.SlotID && pod.Stopped }) {
				return fmt.Errorf("old bootstrap slot %s has no stop proof", replica.SlotID)
			}
		}
	}
	for _, ref := range refs {
		if !slices.ContainsFunc(fence.Pods, func(pod worldPodFence) bool { return pod.Ref == ref && pod.Stopped }) {
			return fmt.Errorf("old allocated Pod %s (UID %s) has no stop proof", ref.Name, ref.UID)
		}
	}
	return nil
}

func (p *GroveWorkloadProvider) resolveReplacementGroveClique(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (*grove.PodClique, error) {
	// Recovery may not adopt a same-named PCS owned by another DGD lifetime.
	owner := metav1.GetControllerOf(group)
	if owner == nil || owner.Kind != "DynamoGraphDeployment" || owner.APIVersion != api.GroupVersion.String() {
		return nil, errors.New("Grove world restart requires a DGD-owned Engine Group")
	}
	dgd := &api.DynamoGraphDeployment{}
	if err := p.Client.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: owner.Name}, dgd); err != nil {
		return nil, client.IgnoreNotFound(err)
	}
	if dgd.UID != owner.UID || !dgd.DeletionTimestamp.IsZero() {
		return nil, errors.New("Grove world restart owner changed or is terminating")
	}
	pcs := &grove.PodCliqueSet{}
	if err := p.Client.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: dynamo.PCSNameForDGD(dgd, (*api.DynamoComponentDeploymentSharedSpec).ManagedByExternalController)}, pcs); err != nil {
		return nil, client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(pcs, dgd) || !pcs.DeletionTimestamp.IsZero() {
		return nil, errors.New("replacement Grove PCS has conflicting ownership or is terminating")
	}
	index := group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex]
	if index != "0" {
		return nil, errors.New("Grove world restart currently requires logical world index zero")
	}
	return resolveGroveMemberClique(ctx, p.Client, group, pcs, 0)
}
