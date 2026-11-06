// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"context"
	"slices"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// podMembershipLabels decide which DGD component, if any, a worker Pod belongs to.
var podMembershipLabels = []string{
	consts.KubeLabelDynamoGraphDeploymentName,
	consts.KubeLabelDynamoComponent,
	consts.KubeLabelDynamoComponentType,
	consts.KubeLabelDynamoSelector,
	consts.KubeLabelDynamoWorkerHash,
}

// Setup registers the power-cap propagation controller. The caller registers the
// podcontroller.ComponentIndex Pod index the controller queries. config and runtimeConfig must be non-nil.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig) error {
	r := &Reconciler{client: mgr.GetClient(), groveEnabled: runtimeConfig.Gate.Enabled(features.Grove)}

	// No timed retry covers waiting, so every input that can change the outcome wakes the DGD.
	b := ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeployment-power").
		For(&v1beta1.DynamoGraphDeployment{}, builder.WithPredicates(predicate.Or(
			predicate.GenerationChangedPredicate{},
			predicate.AnnotationChangedPredicate{},
		))).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(config, runtimeConfig)).
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(mapPodToDGD), builder.WithPredicates(podPredicate()))
	if r.groveEnabled {
		b = b.Owns(&grovev1alpha1.PodCliqueSet{}, builder.WithPredicates(podCliqueSetPredicate()))
	}
	return b.Complete(r)
}

// mapPodToDGD enqueues the DGD a Pod's labels name; update events map both the old and new Pod.
func mapPodToDGD(_ context.Context, obj client.Object) []ctrl.Request {
	name := obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
	if name == "" {
		return nil
	}
	return []ctrl.Request{{Namespace: obj.GetNamespace(), Name: name}}
}

// podPredicate admits Pod changes that can change membership, ownership, the cap or terminality.
func podPredicate() predicate.Predicate {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldPod, oldOK := e.ObjectOld.(*corev1.Pod)
			newPod, newOK := e.ObjectNew.(*corev1.Pod)
			if !oldOK || !newOK {
				return false
			}
			return oldPod.Annotations[consts.KubeAnnotationGPUPowerLimit] != newPod.Annotations[consts.KubeAnnotationGPUPowerLimit] ||
				slices.ContainsFunc(podMembershipLabels, func(key string) bool { return oldPod.Labels[key] != newPod.Labels[key] }) ||
				!apiequality.Semantic.DeepEqual(oldPod.OwnerReferences, newPod.OwnerReferences) ||
				terminal(oldPod) != terminal(newPod)
		},
	}
}

// podCliqueSetPredicate admits PodCliqueSet changes that can open or close the Grove update barrier.
func podCliqueSetPredicate() predicate.Predicate {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldPCS, oldOK := e.ObjectOld.(*grovev1alpha1.PodCliqueSet)
			newPCS, newOK := e.ObjectNew.(*grovev1alpha1.PodCliqueSet)
			if !oldOK || !newOK {
				return false
			}
			return oldPCS.Generation != newPCS.Generation ||
				!apiequality.Semantic.DeepEqual(oldPCS.OwnerReferences, newPCS.OwnerReferences) ||
				!apiequality.Semantic.DeepEqual(oldPCS.Status.ObservedGeneration, newPCS.Status.ObservedGeneration) ||
				!apiequality.Semantic.DeepEqual(oldPCS.Status.UpdateProgress, newPCS.Status.UpdateProgress)
		},
	}
}

// terminal reports a Pod that will not run again. pod must be non-nil.
func terminal(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed
}
