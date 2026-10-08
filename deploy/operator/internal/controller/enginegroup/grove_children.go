/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"strconv"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// GroveChildrenReconciler owns child creation and binding, never live scale targets or child status.
type GroveChildrenReconciler struct {
	Client client.Client
}

// Reconcile creates each world once and binds it after Grove publishes its member clique.
// dgd and pcs must not be nil. A missing asynchronous dependency is pending, not a failed world.
func (r *GroveChildrenReconciler) Reconcile(ctx context.Context, dgd *api.DynamoGraphDeployment, pcs *grove.PodCliqueSet) (map[string]api.ComponentReplicaStatus, bool, error) {
	statuses := make(map[string]api.ComponentReplicaStatus)
	ready := true
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.EngineGroup == nil {
			continue
		}

		// Resolve declarative geometry before creating a durable child or binding native resources.
		profile, err := dynamo.ResolveComponentEngineGroupProfile(component)
		if err != nil {
			return nil, false, fmt.Errorf("component %q: %w", component.ComponentName, err)
		}

		// The validated one-world profile selects index zero; pass it explicitly into every binding.
		worldIndex := int32(0)
		worldIndexLabel := strconv.FormatInt(int64(worldIndex), 10)
		name := dynamo.EngineGroupNameForComponent(dgd.Name, component.ComponentName, worldIndex)
		group := &api.DynamoGraphDeploymentEngineGroup{}
		key := client.ObjectKey{Namespace: dgd.Namespace, Name: name}
		if err := r.Client.Get(ctx, key, group); err != nil {
			if !apierrors.IsNotFound(err) {
				return nil, false, fmt.Errorf("get Engine Group %s: %w", key, err)
			}

			// Only creation seeds replicas and policy; subsequent DGD reconciles preserve both.
			group = &api.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: name, Namespace: dgd.Namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName:   dgd.Name,
						consts.KubeLabelDynamoComponent:             component.ComponentName,
						consts.KubeLabelDynamoEngineGroupWorldIndex: worldIndexLabel,
						consts.KubeLabelDynamoEngineGroupRuntime:    consts.KubeLabelDynamoEngineGroupSGLang,
					},
					Annotations: map[string]string{consts.KubeAnnotationDynamoEngineGroupProfile: profile.Geometry.Fingerprint},
				},
				Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: component.EngineGroup.InitialSize},
			}
			if policy := component.EngineGroup.Policy; policy != nil {
				group.Spec.Policy = &api.EngineGroupScalingPolicy{MinReplicas: ptr.To(ptr.Deref(policy.MinSize, profile.InitialReplicas)), MaxReplicas: ptr.To(ptr.Deref(policy.MaxSize, profile.MaximumReplicas))}
			}
			// Verify through the graph's stable frontend Service, not an ephemeral frontend Pod IP.
			for j := range dgd.Spec.Components {
				frontend := &dgd.Spec.Components[j]
				if string(frontend.ComponentType) == consts.ComponentTypeFrontend {
					group.Annotations[consts.KubeAnnotationDynamoEngineGroupVerifyURL] = fmt.Sprintf(
						"http://%s.%s.svc:8000/v1/completions", dynamo.GetDCDResourceName(dgd, frontend.ComponentName, ""), dgd.Namespace,
					)
					break
				}
			}
			if err := controllerutil.SetControllerReference(dgd, group, r.Client.Scheme()); err != nil {
				return nil, false, fmt.Errorf("set Engine Group owner: %w", err)
			}
			if err := r.Client.Create(ctx, group); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					return nil, false, fmt.Errorf("create Engine Group %s: %w", key, err)
				}
				statuses[component.ComponentName] = api.ComponentReplicaStatus{ReadyReplicas: ptr.To(int32(0))}
			} else {
				statuses[component.ComponentName] = api.ComponentReplicaStatus{Replicas: 1, ReadyReplicas: ptr.To(int32(0))}
			}
			ready = false
			continue
		}

		// Never adopt another controller's world or silently reinterpret an existing launch profile.
		if !metav1.IsControlledBy(group, dgd) || group.Labels[consts.KubeLabelDynamoComponent] != component.ComponentName ||
			group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] != profile.Geometry.Fingerprint {
			return nil, false, fmt.Errorf("Engine Group %s has conflicting ownership or immutable profile", key)
		}
		if !group.DeletionTimestamp.IsZero() {
			return nil, false, fmt.Errorf("Engine Group %s is terminating; automatic world recreation is unsupported", key)
		}
		bound, err := r.bindClique(ctx, group, pcs, worldIndex)
		if err != nil {
			return nil, false, err
		}

		// Available, fresh, engine-authoritative state makes a recovering world Ready without requiring its full target.
		available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
		known := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTopologyKnown)
		worldReady := bound && available != nil && known != nil &&
			available.Status == metav1.ConditionTrue && known.Status == metav1.ConditionTrue &&
			available.ObservedGeneration == group.Generation && known.ObservedGeneration == group.Generation
		status := api.ComponentReplicaStatus{Replicas: 1, ReadyReplicas: ptr.To(int32(0))}
		if group.Status.Profile != nil {
			allocatedGPUs := int64(group.Status.Replicas) * group.Status.Profile.GPUsPerReplica
			status.GPUsPerReplica = ptr.To(allocatedGPUs)
			status.GPUsPerEngine = ptr.To(allocatedGPUs)
		}
		if worldReady {
			status.ReadyReplicas = ptr.To(int32(1))
		}
		statuses[component.ComponentName] = status
		ready = ready && worldReady
	}
	return statuses, ready, nil
}

// RequestDeletion hands retirement to each owned child's controller and waits for its disappearance.
// dgd must not be nil. UID preconditions prevent a stale list from deleting a replacement world.
func (r *GroveChildrenReconciler) RequestDeletion(ctx context.Context, dgd *api.DynamoGraphDeployment) (bool, error) {
	// Ownership, not labels or the current component list, determines which worlds must retire.
	groups := &api.DynamoGraphDeploymentEngineGroupList{}
	if err := r.Client.List(ctx, groups, client.InNamespace(dgd.Namespace)); err != nil {
		return false, fmt.Errorf("list Engine Groups before DGD deletion: %w", err)
	}
	pending := false
	for i := range groups.Items {
		group := &groups.Items[i]
		if !metav1.IsControlledBy(group, dgd) {
			continue
		}
		pending = true
		if !group.DeletionTimestamp.IsZero() {
			continue
		}

		// Keep the parent and its workload alive while the child drains and fences its effects.
		if err := r.Client.Delete(ctx, group, client.Preconditions{UID: &group.UID}); err != nil && !apierrors.IsNotFound(err) {
			return true, fmt.Errorf("request Engine Group %s retirement: %w", group.Name, err)
		}
	}
	return pending, nil
}

func (r *GroveChildrenReconciler) bindClique(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, pcs *grove.PodCliqueSet, worldIndex int32) (bool, error) {
	// The native world ordinal must agree with the logical identity, independent of clique names and UIDs.
	wantIndex := strconv.FormatInt(int64(worldIndex), 10)
	if index, present := group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex]; present && index != wantIndex {
		return false, fmt.Errorf("Engine Group %s has conflicting logical world index", group.Name)
	}

	// Resolve the same native ownership chain used by the child's restart handoff.
	clique, err := resolveGroveMemberClique(ctx, r.Client, group, pcs, worldIndex)
	if err != nil || clique == nil {
		return false, err
	}
	name := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique]
	uid := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]
	if name != "" || uid != "" {
		if name != clique.Name || uid != string(clique.UID) {
			// The child controller owns the fenced restart handoff, not the DGD's seed reconciliation.
			return false, nil
		}
		if group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex] == wantIndex {
			return true, nil
		}
	}

	// Persist logical identity separately from the physical incarnation; backfill owned children without changing capacity.
	before := group.DeepCopy()
	if group.Labels == nil {
		group.Labels = make(map[string]string)
	}
	if group.Annotations == nil {
		group.Annotations = make(map[string]string)
	}
	group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex] = wantIndex
	group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] = clique.Name
	group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = string(clique.UID)
	if err := r.Client.Patch(ctx, group, client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{})); err != nil {
		return false, fmt.Errorf("bind Engine Group member clique: %w", err)
	}
	return true, nil
}

// resolveGroveMemberClique returns only a member of the exact logical world and PCS.
// A nil clique means Grove has not published that asynchronous dependency yet.
func resolveGroveMemberClique(ctx context.Context, reader client.Reader, group *api.DynamoGraphDeploymentEngineGroup, pcs *grove.PodCliqueSet, worldIndex int32) (*grove.PodClique, error) {
	// Select by the rendered world label, not a guessed Pod name or a mutable rank ordinal.
	wantIndex := strconv.FormatInt(int64(worldIndex), 10)
	cliques := &grove.PodCliqueList{}
	if err := reader.List(ctx, cliques, client.InNamespace(group.Namespace), client.MatchingLabels{consts.KubeLabelDynamoEngineGroup: group.Name}); err != nil {
		return nil, fmt.Errorf("list Engine Group member cliques: %w", err)
	}
	if len(cliques.Items) == 0 {
		return nil, nil
	}
	if len(cliques.Items) != 1 {
		return nil, fmt.Errorf("Engine Group %s requires exactly one member clique", group.Name)
	}
	clique := &cliques.Items[0]
	if clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] != wantIndex {
		return nil, fmt.Errorf("member clique %s does not belong to world index %s", clique.Name, wantIndex)
	}
	owner := metav1.GetControllerOf(clique)
	if owner == nil || owner.Kind != "PodCliqueScalingGroup" || owner.APIVersion != grove.SchemeGroupVersion.String() || clique.UID == "" {
		return nil, fmt.Errorf("member clique %s has invalid world ownership or UID", clique.Name)
	}

	// Verify the complete native ownership chain before accepting a capacity binding.
	world := &grove.PodCliqueScalingGroup{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: owner.Name}, world); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("get member-clique world: %w", err)
	}
	if world.UID != owner.UID || !metav1.IsControlledBy(world, pcs) {
		return nil, fmt.Errorf("member clique %s is not owned by the DGD's Grove world", clique.Name)
	}
	return clique, nil
}
