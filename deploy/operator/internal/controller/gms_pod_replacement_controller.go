/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// gmsPodReplacementReconciler replaces Pods that cannot safely recover after
// their native GMS sidecar restarts. The current eligibility policy is limited
// to owned Snapshot restore-target Pods.
type gmsPodReplacementReconciler struct {
	client.Client
	config        *configv1alpha1.OperatorConfiguration
	runtimeConfig *commoncontroller.RuntimeConfig
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;delete

func (r *gmsPodReplacementReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, fmt.Errorf("get GMS Pod replacement candidate %s: %w", req.NamespacedName, err)
	}

	if !pod.DeletionTimestamp.IsZero() || !isGMSPodReplacementEligible(&pod) {
		return ctrl.Result{}, nil
	}

	// The UID precondition prevents a stale reconcile from deleting a same-named replacement.
	uid := pod.UID
	if err := r.Delete(ctx, &pod, client.Preconditions{UID: &uid}); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, fmt.Errorf("delete GMS Pod replacement candidate %s/%s: %w", pod.Namespace, pod.Name, err)
	}
	return ctrl.Result{}, nil
}

func (r *gmsPodReplacementReconciler) setupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("gms-pod-replacement").
		For(&corev1.Pod{}, builder.WithPredicates(gmsPodReplacementPredicate())).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.config, r.runtimeConfig)).
		Complete(r)
}

func isGMSPodReplacementEligible(pod *corev1.Pod) bool {
	return pod != nil &&
		pod.Labels[snapshotprotocol.RestoreTargetLabel] == commonconsts.KubeLabelValueTrue &&
		metav1.GetControllerOf(pod) != nil &&
		hasRestartedNativeGMSServer(pod)
}

func hasRestartedNativeGMSServer(pod *corev1.Pod) bool {
	nativeSidecar := false
	for i := range pod.Spec.InitContainers {
		container := &pod.Spec.InitContainers[i]
		if container.Name == gms.ServerContainerName &&
			container.RestartPolicy != nil &&
			*container.RestartPolicy == corev1.ContainerRestartPolicyAlways {
			nativeSidecar = true
			break
		}
	}
	if !nativeSidecar {
		return false
	}

	for i := range pod.Status.InitContainerStatuses {
		status := &pod.Status.InitContainerStatuses[i]
		if status.Name == gms.ServerContainerName && status.RestartCount > 0 {
			return true
		}
	}
	return false
}

func gmsPodReplacementPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			pod, ok := e.Object.(*corev1.Pod)
			return ok && pod.DeletionTimestamp.IsZero() && isGMSPodReplacementEligible(pod)
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldPod, oldOK := e.ObjectOld.(*corev1.Pod)
			newPod, newOK := e.ObjectNew.(*corev1.Pod)
			if !oldOK || !newOK || !newPod.DeletionTimestamp.IsZero() {
				return false
			}
			return isGMSPodReplacementEligible(newPod) &&
				(!isGMSPodReplacementEligible(oldPod) || oldPod.UID != newPod.UID)
		},
		DeleteFunc:  func(event.DeleteEvent) bool { return false },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}
