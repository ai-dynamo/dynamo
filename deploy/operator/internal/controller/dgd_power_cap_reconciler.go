// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	powerCapPropagationCondition = "PowerCapPropagationReady"
	// powerCapRetry re-observes expected convergence: repairs, cache lag, conflicts, and pending Grove updates.
	powerCapRetry = 2 * time.Second
	// powerCapFailureRetry re-attempts failed Pod observation or repair without failing workload readiness.
	powerCapFailureRetry = 10 * time.Second
)

// powerCapConditionRank orders outcomes so one Pod's result never hides a worse one in the same pass.
var powerCapConditionRank = map[string]int{
	"PodAnnotationsMatch":   0,
	"PodAnnotationMismatch": 1,
	"PodObservationFailed":  2,
	"PodPatchFailed":        3,
}

func powerCapRolloutBlocked(status *v1beta1.RollingUpdateStatus) bool {
	return rollingUpdateInProgress(status) || (status != nil && status.Phase == v1beta1.RollingUpdatePhaseFailed)
}

// desiredWorkerPowerCaps bounds all feature-specific reads to supported annotated graphs.
func desiredWorkerPowerCaps(dgd *v1beta1.DynamoGraphDeployment) map[string]string {
	if _, supported := dynamo.LivePowerCapSupport(dgd); !supported {
		return nil
	}
	caps := map[string]string{}
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if dynamo.IsWorkerComponent(string(component.ComponentType)) && component.PodTemplate != nil {
			if cap, ok := component.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]; ok {
				caps[component.ComponentName] = cap
			}
		}
	}
	return caps
}

// reconcilePowerCaps reports the observation before writes; a subsequent observation proves convergence.
// Propagation outcomes are reported only through its condition and a retry, so workload readiness,
// State, and the acknowledged generation stay owned by ordinary reconciliation. A waiting Grove update
// defers all Pod reads and writes. dgd and result must be non-nil; c may be nil only when caps is empty.
func reconcilePowerCaps(ctx context.Context, c client.Client, dgd *v1beta1.DynamoGraphDeployment, caps map[string]string, waiting bool, result *workloadProgramResult) {
	if len(caps) == 0 {
		meta.RemoveStatusCondition(&result.Status.Conditions, powerCapPropagationCondition)
		return
	}
	condition := metav1.Condition{Type: powerCapPropagationCondition, ObservedGeneration: dgd.Generation, Status: metav1.ConditionTrue, Reason: "PodAnnotationsMatch", Message: "No live workers were observed"}
	defer func() { meta.SetStatusCondition(&result.Status.Conditions, condition) }()
	if waiting {
		condition.Status, condition.Reason, condition.Message = metav1.ConditionFalse, "WaitingForWorkerUpdate", "Worker Pod annotations wait until Grove acknowledges and finishes the PodCliqueSet update"
		requeuePowerCaps(result, powerCapRetry)
		return
	}
	logger := log.FromContext(ctx)

	// Observe only the indexed Pods of each capped component, in a stable order.
	for _, component := range slices.Sorted(maps.Keys(caps)) {
		cap := caps[component]
		pods := &corev1.PodList{}
		if err := c.List(ctx, pods, client.InNamespace(dgd.Namespace), client.MatchingFields{dgdComponentPodIndex: dgdComponentPodIndexValue(dgd.Name, component)}); err != nil {
			logger.Error(err, "Failed to observe power-cap worker Pods", "component", component)
			degradePowerCapCondition(&condition, metav1.ConditionUnknown, "PodObservationFailed", err.Error())
			requeuePowerCaps(result, powerCapFailureRetry)
			continue
		}
		for i := range pods.Items {
			pod := &pods.Items[i]
			if isTerminalPhase(pod.Status.Phase) || !dynamo.IsWorkerComponent(pod.Labels[consts.KubeLabelDynamoComponentType]) {
				continue
			}
			owned, pending, err := powerCapPodOwned(ctx, c, dgd, pod)
			if err != nil {
				logger.Error(err, "Failed to observe power-cap worker Pod ownership", "pod", pod.Name)
				degradePowerCapCondition(&condition, metav1.ConditionUnknown, "PodObservationFailed", err.Error())
				requeuePowerCaps(result, powerCapFailureRetry)
				continue
			}
			if pending {
				degradePowerCapCondition(&condition, metav1.ConditionUnknown, "PodObservationFailed", "Waiting to observe the worker ownership chain")
				requeuePowerCaps(result, powerCapRetry)
				continue
			}
			if !owned {
				continue
			}
			if pod.Annotations[consts.KubeAnnotationGPUPowerLimit] == cap {
				if condition.Status == metav1.ConditionTrue {
					condition.Message = "All owned live worker Pod annotations match"
				}
				continue
			}
			degradePowerCapCondition(&condition, metav1.ConditionFalse, "PodAnnotationMismatch", "Worker Pod annotations are converging")

			// UID and resourceVersion protect against deletion/name reuse and concurrent edits.
			patch, err := json.Marshal(map[string]any{"metadata": map[string]any{"uid": pod.UID, "resourceVersion": pod.ResourceVersion, "annotations": map[string]string{consts.KubeAnnotationGPUPowerLimit: cap}}})
			if err == nil {
				err = c.Patch(ctx, pod, client.RawPatch(types.MergePatchType, patch))
			}
			switch {
			case err == nil || apierrors.IsNotFound(err):
				requeuePowerCaps(result, powerCapRetry)
			case apierrors.IsConflict(err):
				// A concurrent Pod update needs a fresh observation, not a slower failure retry.
				degradePowerCapCondition(&condition, metav1.ConditionFalse, "PodPatchFailed", err.Error())
				requeuePowerCaps(result, powerCapRetry)
			default:
				logger.Error(err, "Failed to repair power-cap worker Pod annotation", "pod", pod.Name)
				degradePowerCapCondition(&condition, metav1.ConditionFalse, "PodPatchFailed", err.Error())
				requeuePowerCaps(result, powerCapFailureRetry)
			}
		}
	}
}

// degradePowerCapCondition records an outcome only when it outranks the condition's current reason.
// condition must be non-nil.
func degradePowerCapCondition(condition *metav1.Condition, status metav1.ConditionStatus, reason, message string) {
	if powerCapConditionRank[reason] > powerCapConditionRank[condition.Reason] {
		condition.Status, condition.Reason, condition.Message = status, reason, message
	}
}

// requeuePowerCaps keeps the earliest retry when several causes, including the program's own, coincide.
func requeuePowerCaps(result *workloadProgramResult, after time.Duration) {
	if result.RequeueAfter == 0 || result.RequeueAfter > after {
		result.RequeueAfter = after
	}
}

// powerCapPodOwned follows only supported controller kinds and checks every owner UID.
// A missing cached parent is pending; a mismatched UID or foreign chain is not authorized.
func powerCapPodOwned(ctx context.Context, reader client.Reader, dgd *v1beta1.DynamoGraphDeployment, pod *corev1.Pod) (owned, pending bool, err error) {
	var object client.Object = pod
	for range 6 {
		owner := metav1.GetControllerOf(object)
		if owner == nil {
			return false, false, nil
		}
		if (owner.APIVersion == v1beta1.GroupVersion.String() || owner.APIVersion == "nvidia.com/v1alpha1") && owner.Kind == "DynamoGraphDeployment" {
			return owner.Name == dgd.Name && owner.UID == dgd.UID, false, nil
		}
		var parent client.Object
		switch owner.APIVersion + "/" + owner.Kind {
		case "apps/v1/ReplicaSet":
			parent = &metav1.PartialObjectMetadata{}
			parent.GetObjectKind().SetGroupVersionKind(appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))
		case "apps/v1/Deployment":
			parent = &appsv1.Deployment{}
		case v1beta1.GroupVersion.Group + "/v1beta1/DynamoComponentDeployment":
			parent = &v1beta1.DynamoComponentDeployment{}
		case "nvidia.com/v1alpha1/DynamoComponentDeployment":
			parent = &v1beta1.DynamoComponentDeployment{}
		case grovev1alpha1.SchemeGroupVersion.Group + "/v1alpha1/PodClique":
			parent = &grovev1alpha1.PodClique{}
		case grovev1alpha1.SchemeGroupVersion.Group + "/v1alpha1/PodCliqueScalingGroup":
			parent = &grovev1alpha1.PodCliqueScalingGroup{}
		case grovev1alpha1.SchemeGroupVersion.Group + "/v1alpha1/PodCliqueSet":
			parent = &grovev1alpha1.PodCliqueSet{}
		default:
			return false, false, nil
		}
		if err := reader.Get(ctx, client.ObjectKey{Namespace: pod.Namespace, Name: owner.Name}, parent); err != nil {
			if apierrors.IsNotFound(err) {
				return false, true, nil
			}
			return false, false, fmt.Errorf("observe power-cap Pod owner: %w", err)
		}
		if parent.GetUID() != owner.UID {
			return false, false, nil
		}
		object = parent
	}
	return false, false, nil
}
