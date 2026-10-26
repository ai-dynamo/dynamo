/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/pods"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// dcdReasonContainerCrashLooping is the DCD Available=False reason that the
// DGD projects into a failed state.
const dcdReasonContainerCrashLooping = "ContainerCrashLooping"

// findDeploymentCrashLoop reports a crash-looping container among the Pods the
// Deployment selects, including Pods of older revisions during a rollout.
func (r *DynamoComponentDeploymentReconciler) findDeploymentCrashLoop(
	ctx context.Context,
	deployment *appsv1.Deployment,
) (pods.CrashLoop, bool, error) {
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return pods.CrashLoop{}, false, fmt.Errorf("parse Deployment %s selector: %w", deployment.Name, err)
	}
	podList := &corev1.PodList{}
	if err := r.List(ctx, podList,
		client.InNamespace(deployment.Namespace),
		client.MatchingLabelsSelector{Selector: selector},
	); err != nil {
		return pods.CrashLoop{}, false, fmt.Errorf("list Pods of Deployment %s: %w", deployment.Name, err)
	}
	crashLoop, found := pods.FindCrashLoop(podList.Items)
	return crashLoop, found, nil
}

// deploymentPodDCDName returns the DCD whose Deployment selects obj. A DCD's
// Deployment shares its name and selects Pods by that name.
func deploymentPodDCDName(obj client.Object) (string, bool) {
	name := obj.GetLabels()[commonconsts.KubeLabelDynamoSelector]
	return name, name != ""
}

// componentPodDGDName returns the DGD whose component obj is labeled for.
func componentPodDGDName(obj client.Object) (string, bool) {
	labels := obj.GetLabels()
	name := labels[commonconsts.KubeLabelDynamoGraphDeploymentName]
	return name, name != "" && labels[commonconsts.KubeLabelDynamoComponent] != ""
}

// findComponentPodCrashLoops returns the crash-loop diagnosis of each component
// that has a crash-looping container among the Pods labeled for it.
func findComponentPodCrashLoops(
	ctx context.Context,
	reader client.Reader,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	components []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (map[string]string, error) {
	crashLoops := map[string]string{}
	for i := range components {
		componentPods, err := listDGDComponentPods(ctx, reader, dgd, components[i].ComponentName)
		if err != nil {
			return nil, err
		}
		if crashLoop, found := pods.FindCrashLoop(componentPods); found {
			crashLoops[components[i].ComponentName] = crashLoop.String()
		}
	}
	return crashLoops, nil
}

// mapPodToRequests enqueues the object that requestName names for a Pod, in the Pod's namespace.
func mapPodToRequests(requestName func(client.Object) (string, bool)) handler.MapFunc {
	return func(_ context.Context, obj client.Object) []ctrl.Request {
		name, ok := requestName(obj)
		if !ok {
			return nil
		}
		return []ctrl.Request{{Namespace: obj.GetNamespace(), Name: name}}
	}
}

// podCrashLoopPredicate admits events that change the crash-loop diagnosis of a
// Pod that requestName maps. Workload status does not change when a container restarts.
func podCrashLoopPredicate(requestName func(client.Object) (string, bool)) predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(event.CreateEvent) bool {
			return false
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			_, ok := requestName(e.Object)
			return ok
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			if _, ok := requestName(e.ObjectNew); !ok {
				return false
			}
			oldPod, oldOK := e.ObjectOld.(*corev1.Pod)
			newPod, newOK := e.ObjectNew.(*corev1.Pod)
			if !oldOK || !newOK {
				return false
			}
			oldCrashLoop, oldFound := pods.CrashLoopOf(oldPod)
			newCrashLoop, newFound := pods.CrashLoopOf(newPod)
			return oldFound != newFound || oldCrashLoop != newCrashLoop
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}
