/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
)

const topologyLabelMissingReason = "TopologyLabelMissing"

// TopologyLabelReconciler watches worker pods that have the
// nvidia.com/topology-label-key annotation. When a pod is scheduled
// (spec.nodeName is set) but the target label is missing, the controller
// reads the label from the node and patches it onto the pod. The Downward
// API volume then picks up the label value.
type TopologyLabelReconciler struct {
	client.Client
	NodeReader    client.Reader
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig
	Recorder      record.EventRecorder
}

// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;patch
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get

func (r *TopologyLabelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	labelKey, ok := topologyLabelCopyTarget(&pod)
	if !ok {
		return ctrl.Result{}, nil
	}

	var node corev1.Node
	if err := r.NodeReader.Get(ctx, types.NamespacedName{Name: pod.Spec.NodeName}, &node); err != nil {
		return ctrl.Result{}, fmt.Errorf("get node %s: %w", pod.Spec.NodeName, err)
	}

	labelValue, exists := node.Labels[labelKey]
	if !exists {
		logger.Info("Node missing topology label, skipping",
			"node", pod.Spec.NodeName, "labelKey", labelKey, "pod", req.NamespacedName)
		if r.Recorder != nil {
			r.Recorder.Eventf(&pod, corev1.EventTypeWarning, topologyLabelMissingReason,
				"Node %q does not have required topology label %q; topology metadata will remain unavailable",
				pod.Spec.NodeName, labelKey)
		}
		return ctrl.Result{}, nil
	}

	patch := client.MergeFrom(pod.DeepCopy())
	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}
	pod.Labels[labelKey] = labelValue
	if err := r.Patch(ctx, &pod, patch); err != nil {
		return ctrl.Result{}, fmt.Errorf("patch pod label: %w", err)
	}

	logger.Info("Copied node topology label to pod",
		"pod", req.NamespacedName, "node", pod.Spec.NodeName,
		"label", labelKey, "value", labelValue)
	return ctrl.Result{}, nil
}

func (r *TopologyLabelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		WithEventFilter(predicate.And(
			commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig),
			topologyLabelPredicate(),
		)).
		Complete(r)
}

// topologyLabelPredicate filters to annotated, scheduled pods that still need
// the target topology label copied from the node.
func topologyLabelPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return needsTopologyLabelCopy(e.Object)
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			return topologyLabelCopyBecameNeeded(e.ObjectOld, e.ObjectNew)
		},
		DeleteFunc: func(_ event.DeleteEvent) bool {
			return false
		},
		GenericFunc: func(_ event.GenericEvent) bool {
			return false
		},
	}
}

func topologyLabelCopyBecameNeeded(oldObj, newObj client.Object) bool {
	oldPod, ok := oldObj.(*corev1.Pod)
	if !ok || oldPod == nil {
		return false
	}
	newPod, ok := newObj.(*corev1.Pod)
	if !ok || newPod == nil {
		return false
	}

	if !needsTopologyLabelCopy(newPod) {
		return false
	}

	oldLabelKey := oldPod.GetAnnotations()[consts.KubeAnnotationTopologyLabelKey]
	newLabelKey := newPod.GetAnnotations()[consts.KubeAnnotationTopologyLabelKey]

	if oldPod.Spec.NodeName == "" && newPod.Spec.NodeName != "" {
		return true
	}
	if oldLabelKey != newLabelKey {
		return true
	}
	if oldLabelKey == "" {
		return false
	}

	_, oldHadLabel := oldPod.GetLabels()[oldLabelKey]
	_, newHasLabel := newPod.GetLabels()[oldLabelKey]
	return oldHadLabel && !newHasLabel
}

func needsTopologyLabelCopy(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return false
	}
	_, ok = topologyLabelCopyTarget(pod)
	return ok
}

func topologyLabelCopyTarget(pod *corev1.Pod) (string, bool) {
	if pod == nil || !isDynamoComponentPod(pod) {
		return "", false
	}

	labelKey := pod.GetAnnotations()[consts.KubeAnnotationTopologyLabelKey]
	if labelKey == "" {
		return "", false
	}

	if _, exists := pod.GetLabels()[labelKey]; exists {
		return "", false
	}

	if pod.Spec.NodeName == "" {
		return "", false
	}

	return labelKey, true
}

func isDynamoComponentPod(pod *corev1.Pod) bool {
	labels := pod.GetLabels()
	return labels[consts.KubeLabelDynamoGraphDeploymentName] != "" &&
		labels[consts.KubeLabelDynamoComponent] != ""
}
