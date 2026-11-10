// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"slices"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// podMembershipLabels decide which DGD component, if any, a worker Pod belongs to.
var podMembershipLabels = []string{
	consts.KubeLabelDynamoGraphDeploymentName,
	consts.KubeLabelDynamoComponent,
	consts.KubeLabelDynamoComponentType,
}

// Setup registers the power-cap propagation controller. The caller registers the
// podcontroller.ComponentIndex Pod index the controller queries. config and runtimeConfig must be non-nil.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig) error {
	r := &reconciler{client: mgr.GetClient(), groveEnabled: runtimeConfig.Gate.Enabled(features.Grove)}

	// No timed retry covers waiting, so every input that can change the outcome wakes the DGD.
	b := ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeployment-power").
		For(&v1beta1.DynamoGraphDeployment{}, builder.WithPredicates(predicate.Or(
			predicate.GenerationChangedPredicate{},
			predicate.AnnotationChangedPredicate{},
		))).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(config, runtimeConfig)).
		Watches(&corev1.Pod{}, podcontroller.EnqueueOwners(podcontroller.DGDName), builder.WithPredicates(podPredicate()))
	if r.groveEnabled {
		b = b.Watches(&grovev1alpha1.PodCliqueSet{}, podcontroller.EnqueueOwners(func(obj client.Object) (string, bool) {
			name := obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
			return name, name != ""
		}), builder.WithPredicates(podCliqueSetPredicate()))
	}
	return b.Complete(r)
}

// podPredicate admits Pod changes that can change label membership, the cap or terminality.
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
				oldPCS.Labels[consts.KubeLabelDynamoGraphDeploymentName] != newPCS.Labels[consts.KubeLabelDynamoGraphDeploymentName] ||
				!apiequality.Semantic.DeepEqual(oldPCS.Status.ObservedGeneration, newPCS.Status.ObservedGeneration) ||
				!apiequality.Semantic.DeepEqual(oldPCS.Status.UpdateProgress, newPCS.Status.UpdateProgress)
		},
	}
}

// terminal reports a Pod that will not run again. pod must be non-nil.
func terminal(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed
}
