// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"fmt"
	"slices"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// EvaluateLPXGroveReadiness evaluates every observed role and replica of one component group.
// source is non-nil and admitted; groupName and componentNames identify one entry
// from ComponentGroups. pcs and pcsg may be nil while materializing. The caller
// verifies ownership and deletion state before passing pclqs.
// Missing cliques are pending. Inputs remain read-only.
func EvaluateLPXGroveReadiness(ctx context.Context, source *v1beta1.DynamoGraphDeployment, groupName string, componentNames []string, pcs *grovev1alpha1.PodCliqueSet, pcsg *grovev1alpha1.PodCliqueScalingGroup, pclqs map[string]*grovev1alpha1.PodClique) GroveReadiness {
	component := source.GetComponentByName(groupName)
	status := v1beta1.ComponentReplicaStatus{ComponentKind: v1beta1.ComponentKindPodCliqueScalingGroup, RuntimeNamespace: source.GetDynamoNamespaceForComponent(component)}
	// Draft instances are counted from their own complete Agent cliques.
	statuses := make(map[string]v1beta1.ComponentReplicaStatus)
	for _, name := range componentNames {
		member := source.GetComponentByName(name)
		if member.ComponentName != component.ComponentName {
			statuses[member.ComponentName] = v1beta1.ComponentReplicaStatus{
				ComponentKind: v1beta1.ComponentKindPodClique, RuntimeNamespace: source.GetDynamoNamespaceForComponent(member),
			}
		}
	}
	verifiedAvailable := int32(0)
	result := func(ready bool, classification, message string) GroveReadiness {
		if status.AvailableReplicas != nil {
			status.AvailableReplicas = ptr.To(min(*status.AvailableReplicas, verifiedAvailable))
		}
		statuses[component.ComponentName] = status
		return GroveReadiness{Ready: ready, Classification: classification, Message: message, ComponentStatuses: statuses}
	}
	pending := func(message string) GroveReadiness {
		return result(false, v1beta1.DGDReadyReasonSomeResourcesNotReady, message)
	}
	if pcs == nil {
		return pending("Waiting for the exact LPX PodCliqueSet")
	}
	hash := getAcceptedPCSRevisionHash(pcs)
	if hash == nil {
		return pending("Waiting for Grove to accept the LPX PodCliqueSet revision")
	}
	if pcsg == nil {
		return pending("Waiting for the LPX scaling group")
	}
	status.ComponentNames = []string{pcsg.Name}
	status.Replicas, status.UpdatedReplicas = pcsg.Status.Replicas, pcsg.Status.UpdatedReplicas
	status.AvailableReplicas = ptr.To(pcsg.Status.AvailableReplicas)
	if pcsg.Status.ObservedGeneration == nil || *pcsg.Status.ObservedGeneration != pcsg.Generation {
		return pending("Waiting for the exact observed LPX scaling group")
	}
	status.ScheduledReplicas = ptr.To(pcsg.Status.ScheduledReplicas)
	replicas := ptr.Deref(component.Replicas, pcsg.Spec.Replicas)
	if pcsg.Spec.Replicas != replicas || pcsg.Status.CurrentPodCliqueSetGenerationHash == nil || *pcsg.Status.CurrentPodCliqueSetGenerationHash != *hash {
		return result(false, v1beta1.DGDReadyReasonUpdating, "LPX scaling group has not applied the desired revision and capacity")
	}
	// Observe every member before returning so partial draft readiness remains visible.
	unreadyClassification, unreadyMessage := "", ""
	noteUnready := func(classification, message string) {
		if unreadyMessage == "" {
			unreadyClassification, unreadyMessage = classification, message
		}
	}
	for replica := range replicas {
		replicaReady := true
		for _, template := range pcs.Spec.Template.Cliques {
			if !slices.Contains(pcsg.Spec.CliqueNames, template.Name) {
				continue
			}
			name := grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: pcsg.Name, Replica: int(replica)}, template.Name)
			memberName := template.Labels[commonconsts.KubeLabelDynamoComponent]

			// Revision checks precede the pure per-clique readiness calculation.
			pclq := pclqs[name]
			readiness := groveComponentReadiness{}
			switch {
			case pclq == nil:
				readiness = readiness.withResult(false, fmt.Sprintf("Waiting for LPX role %s", name), v1beta1.DGDReadyReasonSomeResourcesNotReady)
			case pclq.Status.CurrentPodCliqueSetGenerationHash == nil || *pclq.Status.CurrentPodCliqueSetGenerationHash != *hash:
				readiness = readiness.withResult(false, fmt.Sprintf("LPX role %s has not applied the desired revision", name), v1beta1.DGDReadyReasonUpdating)
			default:
				readiness = observeLPXRole(ctx, pclq)
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
		return result(false, unreadyClassification, unreadyMessage)
	}
	ready, message, classification := pcsgStatusReady(pcsg, replicas)
	if ready {
		classification = v1beta1.DGDReadyReasonAllResourcesReady
	}
	return result(ready, classification, message)
}

// observeLPXRole calculates readiness and complete-instance counts from a non-nil
// PodClique at the accepted PCS revision. Capacity reconciliation has already
// applied explicit counts; omitted counts retain the observed external capacity.
func observeLPXRole(ctx context.Context, pclq *grovev1alpha1.PodClique) groveComponentReadiness {
	role := groveComponentReadiness{}
	replicas := pclq.Spec.Replicas

	// A model instance is counted only after the complete build width is observed.
	readiness := podCliqueReadiness(pclq, log.FromContext(ctx))
	role = role.withResult(readiness.ready, readiness.reason, readiness.classification)
	if pclq.Status.ObservedGeneration == nil || *pclq.Status.ObservedGeneration != pclq.Generation {
		return role
	}
	if pclq.Status.Replicas >= replicas {
		role.status.Replicas = 1
	}
	if pclq.Status.UpdatedReplicas >= replicas {
		role.status.UpdatedReplicas = 1
	}
	scheduled, ready := int32(0), int32(0)
	if pclq.Status.ScheduledReplicas >= replicas || readiness.ready {
		scheduled = 1
	}
	if readiness.ready {
		ready = 1
	}
	role.status.ScheduledReplicas, role.status.ReadyReplicas = &scheduled, &ready
	return role
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
