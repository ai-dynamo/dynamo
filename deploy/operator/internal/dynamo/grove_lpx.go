// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"fmt"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// EvaluateLPXGroveReadiness observes one complete engine, including every role
// in every replica. Ordinary DGD components are deliberately not evaluated here.
func EvaluateLPXGroveReadiness(ctx context.Context, reader client.Reader, source *v1beta1.DynamoGraphDeployment, deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet) (GroveReadiness, error) {
	component := lpx.ServingComponent(source)
	status := v1beta1.ComponentReplicaStatus{ComponentKind: v1beta1.ComponentKindPodCliqueScalingGroup, RuntimeNamespace: source.GetDynamoNamespaceForComponent(component)}
	// Draft instances are counted from their own complete Agent cliques.
	statuses := make(map[string]v1beta1.ComponentReplicaStatus)
	for _, member := range lpx.Components(source) {
		if member.ComponentName != component.ComponentName {
			statuses[member.ComponentName] = v1beta1.ComponentReplicaStatus{
				ComponentKind: v1beta1.ComponentKindPodClique, RuntimeNamespace: source.GetDynamoNamespaceForComponent(member),
			}
		}
	}
	verifiedAvailable := int32(0)
	result := func(ready bool, classification, message string) GroveReadiness {
		status.Ready = ready
		if status.AvailableReplicas != nil {
			status.AvailableReplicas = ptr.To(min(*status.AvailableReplicas, verifiedAvailable))
		}
		statuses[component.ComponentName] = status
		for name, member := range statuses {
			member.Ready = ready
			statuses[name] = member
		}
		return GroveReadiness{Ready: ready, Classification: classification, Message: message, ComponentStatuses: statuses}
	}
	pending := func(message string) GroveReadiness {
		return result(false, v1beta1.DGDReadyReasonSomeResourcesNotReady, message)
	}
	if pcs == nil || !pcs.DeletionTimestamp.IsZero() || !metav1.IsControlledBy(pcs, deployment) ||
		len(pcs.Spec.Template.PodCliqueScalingGroupConfigs) != 1 {
		return pending("Waiting for the exact LPX PodCliqueSet"), nil
	}
	hash := getAcceptedPCSRevisionHash(pcs)
	if hash == nil {
		return pending("Waiting for Grove to accept the LPX PodCliqueSet revision"), nil
	}
	config := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
	groupName := grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, config.Name)
	status.ComponentNames = []string{groupName}
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: pcs.Namespace, Name: groupName}, group); err != nil {
		if apierrors.IsNotFound(err) {
			return pending("Waiting for the LPX scaling group"), nil
		}
		return GroveReadiness{}, err
	}
	status.Replicas, status.UpdatedReplicas = group.Status.Replicas, group.Status.UpdatedReplicas
	status.AvailableReplicas = ptr.To(group.Status.AvailableReplicas)
	if !metav1.IsControlledBy(group, pcs) || !group.DeletionTimestamp.IsZero() || group.Status.ObservedGeneration == nil || *group.Status.ObservedGeneration != group.Generation {
		return pending("Waiting for the exact observed LPX scaling group"), nil
	}
	status.ScheduledReplicas = ptr.To(group.Status.ScheduledReplicas)
	replicas := ptr.Deref(component.Replicas, group.Spec.Replicas)
	if group.Spec.Replicas != replicas || group.Status.CurrentPodCliqueSetGenerationHash == nil || *group.Status.CurrentPodCliqueSetGenerationHash != *hash {
		return result(false, v1beta1.DGDReadyReasonUpdating, "LPX scaling group has not applied the desired revision and capacity"), nil
	}
	// Observe every member before returning so partial draft readiness remains visible.
	unreadyClassification, unreadyMessage := "", ""
	noteUnready := func(classification, message string) {
		if unreadyMessage == "" {
			unreadyClassification, unreadyMessage = classification, message
		}
	}
	for replica := int32(0); replica < replicas; replica++ {
		replicaReady := true
		for _, template := range pcs.Spec.Template.Cliques {
			name := grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: groupName, Replica: int(replica)}, template.Name)
			memberName := template.Labels[lpx.StageLabel]
			readiness, err := observeLPXRole(ctx, reader, group, name, template.Spec.Replicas)
			if err != nil {
				return GroveReadiness{}, err
			}
			// Sum complete model instances, never physical Agent Pod counts.
			if draft, isDraft := statuses[memberName]; isDraft {
				draft.ComponentNames = append(draft.ComponentNames, name)
				if readiness.status.ReadyReplicas != nil {
					draft.Replicas += readiness.status.Replicas
					draft.UpdatedReplicas += readiness.status.UpdatedReplicas
					draft.ScheduledReplicas = ptr.To(ptr.Deref(draft.ScheduledReplicas, 0) + ptr.Deref(readiness.status.ScheduledReplicas, 0))
					draft.ReadyReplicas = ptr.To(ptr.Deref(draft.ReadyReplicas, 0) + *readiness.status.ReadyReplicas)
				}
				statuses[memberName] = draft
			}
			if !readiness.ready {
				noteUnready(readiness.classification, readiness.reason)
				replicaReady = false
			}
		}
		if replicaReady {
			verifiedAvailable++
		}
	}
	if unreadyMessage != "" {
		return result(false, unreadyClassification, unreadyMessage), nil
	}
	ready, message, classification := pcsgStatusReady(group, replicas)
	if ready {
		classification = v1beta1.DGDReadyReasonAllResourcesReady
	}
	return result(ready, classification, message), nil
}

// observeLPXRole fences one clique by ownership and revision, then reports its
// counts in complete model instances. The group must already be observed at the
// accepted PCS revision; all pointer inputs are non-nil.
func observeLPXRole(ctx context.Context, reader client.Reader, group *grovev1alpha1.PodCliqueScalingGroup, name string, width int32) (groveComponentReadiness, error) {
	role := groveComponentReadiness{}
	clique := &grovev1alpha1.PodClique{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: name}, clique); err != nil {
		if apierrors.IsNotFound(err) {
			return role.withResult(false, fmt.Sprintf("Waiting for LPX role %s", name), v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
		}
		return role, err
	}
	if !metav1.IsControlledBy(clique, group) || !clique.DeletionTimestamp.IsZero() {
		return role.withResult(false, fmt.Sprintf("Waiting for exact LPX role %s", name), v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
	}
	if clique.Spec.Replicas != width || clique.Status.CurrentPodCliqueSetGenerationHash == nil ||
		*clique.Status.CurrentPodCliqueSetGenerationHash != *group.Status.CurrentPodCliqueSetGenerationHash {
		return role.withResult(false, fmt.Sprintf("LPX role %s has not applied the desired revision and capacity", name), v1beta1.DGDReadyReasonUpdating), nil
	}

	// A model instance is counted only after the complete build width is observed.
	readiness := podCliqueReadiness(clique, log.FromContext(ctx))
	role = role.withResult(readiness.ready, readiness.reason, readiness.classification)
	if clique.Status.ObservedGeneration == nil || *clique.Status.ObservedGeneration != clique.Generation {
		return role, nil
	}
	if clique.Status.Replicas >= width {
		role.status.Replicas = 1
	}
	if clique.Status.UpdatedReplicas >= width {
		role.status.UpdatedReplicas = 1
	}
	scheduled, ready := int32(0), int32(0)
	if clique.Status.ScheduledReplicas >= width || readiness.ready {
		scheduled = 1
	}
	if readiness.ready {
		ready = 1
	}
	role.status.ScheduledReplicas, role.status.ReadyReplicas = &scheduled, &ready
	return role, nil
}

func pcsgStatusReady(pcsg *grovev1alpha1.PodCliqueScalingGroup, desiredReplicas int32) (bool, string, string) {
	if pcsg.Status.Replicas == desiredReplicas &&
		pcsg.Status.UpdatedReplicas == desiredReplicas &&
		pcsg.Status.AvailableReplicas == desiredReplicas {
		return true, "", ""
	}

	minAvailable := meta.FindStatusCondition(pcsg.Status.Conditions, groveconstants.ConditionTypeMinAvailableBreached)
	if minAvailable != nil && minAvailable.Status == metav1.ConditionFalse &&
		(minAvailable.Reason == groveconstants.ConditionReasonScheduledReplicasBelowMinAvailable ||
			minAvailable.Reason == legacyConditionReasonInsufficientScheduledPCSGReplicas) {
		return false, fmt.Sprintf("min-available breached (%s): %s", minAvailable.Reason, minAvailable.Message), v1beta1.DGDReadyReasonInsufficientCapacity
	}
	if scheduled := pcsg.Status.ScheduledReplicas; scheduled > 0 && scheduled < desiredReplicas {
		return false, fmt.Sprintf("insufficient scheduled replicas: scheduled=%d/%d", scheduled, desiredReplicas), v1beta1.DGDReadyReasonInsufficientCapacity
	}
	if pcsg.Status.UpdatedReplicas != desiredReplicas {
		return false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, pcsg.Status.UpdatedReplicas), v1beta1.DGDReadyReasonUpdating
	}
	if pcsg.Status.Replicas != desiredReplicas {
		return false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, pcsg.Status.Replicas), v1beta1.DGDReadyReasonUpdating
	}
	return false, fmt.Sprintf("scheduled but available=%d/%d", pcsg.Status.AvailableReplicas, desiredReplicas), v1beta1.DGDReadyReasonPodsNotReady
}
