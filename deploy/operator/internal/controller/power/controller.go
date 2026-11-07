// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package power propagates live GPU power caps from a DynamoGraphDeployment to its running worker Pods.
// The DGD controller keeps worker templates and identities for cap-only edits; this controller patches
// the Pods those templates already created and reports the PowerCapPropagationReady condition.
package power

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// ConditionType is the DGD condition this controller owns.
	ConditionType = "PowerCapPropagationReady"
	// retryAfter re-observes a Pod whose patch conflicted with a concurrent update.
	retryAfter = 2 * time.Second
	// failureRetryAfter re-attempts failed Pod observation or repair.
	failureRetryAfter = 10 * time.Second
)

// conditionRank orders outcomes so one Pod's result never hides a worse one in the same pass.
var conditionRank = map[string]int{
	"PodAnnotationsMatch":   0,
	"PodAnnotationMismatch": 1,
	"PodObservationFailed":  2,
	"PodPatchFailed":        3,
}

// Reconciler patches live worker Pod cap annotations for one DGD per request.
type Reconciler struct {
	client       client.Client
	groveEnabled bool
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;patch
// +kubebuilder:rbac:groups=grove.io,resources=podcliquesets,verbs=get;list;watch

// Reconcile observes the DGD's worker Pods, repairs their cap annotations and applies the condition.
func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	dgd := &v1beta1.DynamoGraphDeployment{}
	if err := r.client.Get(ctx, req.NamespacedName, dgd); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !dgd.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// A graph without supported explicit caps carries no condition.
	caps := dynamo.DesiredWorkerPowerCaps(dgd)
	if len(caps) == 0 {
		if meta.FindStatusCondition(dgd.Status.Conditions, ConditionType) == nil {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, applyCondition(ctx, r.client, dgd, nil)
	}

	// A stored Grove graph without the Grove integration has no observable update state; the DGD
	// program rejects it as unavailable, so its Pods and condition are left untouched as well.
	if dgd.Annotations[consts.KubeAnnotationWorkloadProvider] == "grove" && !r.groveEnabled {
		return ctrl.Result{}, nil
	}

	condition, requeueAfter := r.propagate(ctx, dgd, caps)
	if err := applyCondition(ctx, r.client, dgd, &condition); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: requeueAfter}, nil
}

// propagate reports the observation before writes; a later observation proves convergence.
// A Grove PodCliqueSet that Grove has not acknowledged or finished defers all Pod reads and writes.
// dgd must be non-nil and caps non-empty; a Grove graph requires the Grove integration to be enabled.
func (r *Reconciler) propagate(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, caps map[string]string) (metav1.Condition, time.Duration) {
	condition := metav1.Condition{Type: ConditionType, ObservedGeneration: dgd.Generation, Status: metav1.ConditionTrue, Reason: "PodAnnotationsMatch", Message: "No live workers were observed"}
	var requeueAfter time.Duration
	requeue := func(after time.Duration) {
		if requeueAfter == 0 || requeueAfter > after {
			requeueAfter = after
		}
	}
	logger := log.FromContext(ctx)

	// Wait for the observed Grove update to finish; the PodCliqueSet watch wakes the next pass.
	if dgd.Annotations[consts.KubeAnnotationWorkloadProvider] == "grove" {
		waiting, err := r.groveUpdatePending(ctx, dgd)
		if err != nil {
			logger.Error(err, "Failed to observe the power-cap PodCliqueSet")
			degrade(&condition, metav1.ConditionUnknown, "PodObservationFailed", err.Error())
			return condition, failureRetryAfter
		}
		if waiting {
			condition.Status, condition.Reason, condition.Message = metav1.ConditionFalse, "WaitingForWorkerUpdate", "Worker Pod annotations wait until Grove acknowledges and finishes the PodCliqueSet update"
			return condition, 0
		}
	}

	// The Dynamo graph and component labels associate a Pod with its DGD component; the lowering
	// hierarchy that created the Pod is irrelevant. Observe each capped component in a stable order.
	for _, component := range slices.Sorted(maps.Keys(caps)) {
		cap := caps[component]
		pods, err := podcontroller.ListComponentPods(ctx, r.client, dgd.Namespace, dgd.Name, component)
		if err != nil {
			logger.Error(err, "Failed to observe power-cap worker Pods", "component", component)
			degrade(&condition, metav1.ConditionUnknown, "PodObservationFailed", err.Error())
			requeue(failureRetryAfter)
			continue
		}
		for i := range pods {
			pod := &pods[i]
			if terminal(pod) || !dynamo.IsWorkerComponent(pod.Labels[consts.KubeLabelDynamoComponentType]) {
				continue
			}
			if pod.Annotations[consts.KubeAnnotationGPUPowerLimit] == cap {
				if condition.Status == metav1.ConditionTrue {
					condition.Message = "All live worker Pod annotations match"
				}
				continue
			}
			degrade(&condition, metav1.ConditionFalse, "PodAnnotationMismatch", "Worker Pod annotations are converging")

			// UID and resourceVersion protect against deletion/name reuse and concurrent edits.
			patch, err := json.Marshal(map[string]any{"metadata": map[string]any{"uid": pod.UID, "resourceVersion": pod.ResourceVersion, "annotations": map[string]string{consts.KubeAnnotationGPUPowerLimit: cap}}})
			if err == nil {
				err = r.client.Patch(ctx, pod, client.RawPatch(types.MergePatchType, patch))
			}
			switch {
			case err == nil || apierrors.IsNotFound(err):
				// The Pod update or delete event triggers the observation that proves convergence.
			case apierrors.IsConflict(err):
				// A concurrent Pod update needs a fresh observation, not a slower failure retry.
				degrade(&condition, metav1.ConditionFalse, "PodPatchFailed", err.Error())
				requeue(retryAfter)
			default:
				logger.Error(err, "Failed to repair power-cap worker Pod annotation", "pod", pod.Name)
				degrade(&condition, metav1.ConditionFalse, "PodPatchFailed", err.Error())
				requeue(failureRetryAfter)
			}
		}
	}
	return condition, requeueAfter
}

// groveUpdatePending reports whether the DGD's own PodCliqueSet has an update Grove has not
// acknowledged or finished. An absent or foreign PodCliqueSet has no update to wait for.
// dgd must be non-nil.
func (r *Reconciler) groveUpdatePending(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment) (bool, error) {
	pcs := &grovev1alpha1.PodCliqueSet{}
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.PCSNameForDGD(dgd, (*v1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController)}
	if err := r.client.Get(ctx, key, pcs); err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, fmt.Errorf("get PodCliqueSet %s: %w", key, err)
	}
	if !metav1.IsControlledBy(pcs, dgd) {
		return false, nil
	}
	if pcs.Status.ObservedGeneration == nil || *pcs.Status.ObservedGeneration != pcs.Generation {
		return true, nil
	}
	return pcs.Status.UpdateProgress != nil && pcs.Status.UpdateProgress.UpdateEndedAt == nil, nil
}

// degrade records an outcome only when it outranks the condition's current reason.
// condition must be non-nil.
func degrade(condition *metav1.Condition, status metav1.ConditionStatus, reason, message string) {
	if conditionRank[reason] > conditionRank[condition.Reason] {
		condition.Status, condition.Reason, condition.Message = status, reason, message
	}
}
