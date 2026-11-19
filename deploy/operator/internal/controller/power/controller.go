// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package power delivers GPU power caps from a DynamoGraphDeployment to its live worker Pods
// and reports annotation convergence through the PowerCapPropagationReady condition.
package power

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"slices"
	"strings"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ConditionType is the DGD condition this controller owns.
const ConditionType = "PowerCapPropagationReady"

// conditionRank orders outcomes so one Pod's result never hides a worse one in the same pass.
var conditionRank = map[string]int{
	"PodAnnotationsMatch":   0,
	"PodAnnotationMismatch": 1,
	"PodPatchFailed":        2,
	"PodObservationFailed":  3,
}

// reconciler patches live worker Pod cap annotations for one DGD per request.
type reconciler struct {
	client client.Client
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;patch

// Reconcile observes the DGD's worker Pods, repairs their cap annotations and applies the condition.
func (r *reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	dgd := &v1beta1.DynamoGraphDeployment{}
	if err := r.client.Get(ctx, req.NamespacedName, dgd); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !dgd.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// A graph without supported explicit caps carries no condition. Apply the removal even when the
	// cached DGD shows none: the cache can miss this controller's last write. Once the DGD has status,
	// an unchanged removal writes nothing.
	caps := dynamo.DesiredWorkerPowerCaps(dgd)
	if len(caps) == 0 {
		return ctrl.Result{}, applyCondition(ctx, r.client, dgd, nil)
	}

	condition, propagationErr := r.propagate(ctx, dgd, caps)
	applyErr := applyCondition(ctx, r.client, dgd, &condition)
	return ctrl.Result{}, errors.Join(propagationErr, applyErr)
}

// propagate reports the observation before writes; a later observation proves convergence.
// dgd must be non-nil and caps non-empty.
func (r *reconciler) propagate(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, caps map[string]string) (metav1.Condition, error) {
	condition := metav1.Condition{Type: ConditionType, ObservedGeneration: dgd.Generation, Status: metav1.ConditionTrue, Reason: "PodAnnotationsMatch", Message: "All live worker Pod annotations match"}
	var observedWorkers bool
	var failures []error
	logger := log.FromContext(ctx)

	// The Dynamo graph and component labels associate a Pod with its DGD component; the lowering
	// hierarchy that created the Pod is irrelevant. Observe each capped component in a stable order.
	for _, component := range slices.Sorted(maps.Keys(caps)) {
		cap := caps[component]
		pods, err := podcontroller.ListComponentPods(ctx, r.client, dgd.Namespace, dgd.Name, component)
		if err != nil {
			logger.Error(err, "Failed to observe power-cap worker Pods", "component", component)
			degrade(&condition, metav1.ConditionUnknown, "PodObservationFailed", err.Error())
			failures = append(failures, err)
			continue
		}

		// Visit Pods by name: the first failure's message names its Pod, and an indexed cache List has no order.
		slices.SortFunc(pods, func(a, b corev1.Pod) int { return strings.Compare(a.Name, b.Name) })
		for i := range pods {
			pod := &pods[i]
			if !liveWorkerPod(pod) {
				continue
			}
			observedWorkers = true
			if pod.Annotations[consts.KubeAnnotationGPUPowerLimit] == cap {
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
			default:
				logger.Error(err, "Failed to repair power-cap worker Pod annotation", "pod", pod.Name)
				degrade(&condition, metav1.ConditionFalse, "PodPatchFailed", err.Error())
				failures = append(failures, err)
			}
		}
	}
	if !observedWorkers && len(failures) == 0 {
		condition.Status, condition.Reason, condition.Message = metav1.ConditionUnknown, "NoWorkersObserved", "No live workers were observed"
	}
	return condition, errors.Join(failures...)
}

// liveWorkerPod reports annotation eligibility. pod must be non-nil.
func liveWorkerPod(pod *corev1.Pod) bool {
	return pod.DeletionTimestamp.IsZero() && pod.Status.Phase != corev1.PodSucceeded && pod.Status.Phase != corev1.PodFailed &&
		dynamo.IsWorkerComponent(pod.Labels[consts.KubeLabelDynamoComponentType])
}

// degrade records an outcome only when it outranks the condition's current reason.
// condition must be non-nil.
func degrade(condition *metav1.Condition, status metav1.ConditionStatus, reason, message string) {
	if conditionRank[reason] > conditionRank[condition.Reason] {
		condition.Status, condition.Reason, condition.Message = status, reason, message
	}
}
