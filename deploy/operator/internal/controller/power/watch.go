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
	corev1 "k8s.io/api/core/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
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
	return ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeployment-power").
		For(&v1beta1.DynamoGraphDeployment{}, builder.WithPredicates(dgdPredicate())).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(config, runtimeConfig)).
		Watches(&corev1.Pod{}, podcontroller.EnqueueOwners(podcontroller.DGDName), builder.WithPredicates(podPredicate())).
		Complete(&reconciler{client: mgr.GetClient()})
}

// dgdPredicate admits DGD changes to desired inputs. A status-only update, including this
// controller's own condition, never wakes it.
func dgdPredicate() predicate.Predicate {
	return predicate.Or(predicate.GenerationChangedPredicate{}, predicate.AnnotationChangedPredicate{})
}

// podPredicate admits changes to Pod membership, the cap, or annotation eligibility.
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
				liveWorkerPod(oldPod) != liveWorkerPod(newPod)
		},
	}
}
