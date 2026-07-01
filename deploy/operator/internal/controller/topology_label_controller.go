/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"strings"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
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
	Recorder      events.EventRecorder
}

// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;patch
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get
// +kubebuilder:rbac:groups=grove.io,resources=clustertopologies,verbs=get

func (r *TopologyLabelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	copyTargets, err := r.topologyLabelCopyTargets(ctx, &pod)
	if err != nil {
		return ctrl.Result{}, err
	}
	if len(copyTargets) == 0 {
		return ctrl.Result{}, nil
	}

	var node corev1.Node
	if err := r.NodeReader.Get(ctx, types.NamespacedName{Name: pod.Spec.NodeName}, &node); err != nil {
		return ctrl.Result{}, fmt.Errorf("get node %s: %w", pod.Spec.NodeName, err)
	}

	patch := client.MergeFrom(pod.DeepCopy())
	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}

	patchedLabels := 0
	for _, target := range copyTargets {
		labelValue, exists := node.Labels[target.sourceLabelKey]
		if !exists {
			logger.Info("Node missing topology label, skipping",
				"node", pod.Spec.NodeName, "sourceLabelKey", target.sourceLabelKey,
				"targetLabelKey", target.targetLabelKey, "pod", req.NamespacedName)
			if r.Recorder != nil {
				r.Recorder.Eventf(&pod, &node, corev1.EventTypeWarning, topologyLabelMissingReason, "Update",
					"Node %q does not have required topology label %q; topology metadata for %q will remain unavailable",
					pod.Spec.NodeName, target.sourceLabelKey, target.targetLabelKey)
			}
			continue
		}
		pod.Labels[target.targetLabelKey] = labelValue
		patchedLabels++
	}
	if patchedLabels == 0 {
		return ctrl.Result{}, nil
	}

	if err := r.Patch(ctx, &pod, patch); err != nil {
		return ctrl.Result{}, fmt.Errorf("patch pod label: %w", err)
	}

	logger.Info("Copied node topology label to pod",
		"pod", req.NamespacedName, "node", pod.Spec.NodeName,
		"labels", patchedLabels)
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

	// new Pod has become scheduled
	if oldPod.Spec.NodeName == "" && newPod.Spec.NodeName != "" {
		return true
	}

	// labelKey or clusterTopologyName changed
	if topologySourceAnnotationChanged(oldPod, newPod) {
		return true
	}

	// old Pod did not need the label copy, but new Pod does
	return !needsTopologyLabelCopy(oldPod)
}

func topologySourceAnnotationChanged(oldPod, newPod *corev1.Pod) bool {
	for _, annotationKey := range consts.KubeTopologySourceAnnotationKeys() {
		if oldPod.GetAnnotations()[annotationKey] != newPod.GetAnnotations()[annotationKey] {
			return true
		}
	}
	return false
}

func needsTopologyLabelCopy(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return false
	}
	if !isDynamoComponentPod(pod) {
		return false
	}

	labelKey := pod.GetAnnotations()[consts.KubeAnnotationTopologyLabelKey]
	clusterTopologyName := pod.GetAnnotations()[consts.KubeAnnotationTopologyClusterTopologyName]
	if labelKey == "" && clusterTopologyName == "" {
		return false
	}
	if pod.Spec.NodeName == "" {
		return false
	}
	if labelKey != "" {
		return !keyExists(pod, labelKey)
	}
	return missingDynamoTopologyLabel(pod)
}

func keyExists(pod *corev1.Pod, key string) bool {
	if pod == nil {
		return false
	}
	_, exists := pod.GetLabels()[key]
	return exists
}

type topologyLabelCopyTargetSpec struct {
	sourceLabelKey string
	targetLabelKey string
}

func (r *TopologyLabelReconciler) topologyLabelCopyTargets(ctx context.Context, pod *corev1.Pod) ([]topologyLabelCopyTargetSpec, error) {
	if pod == nil || !isDynamoComponentPod(pod) || pod.Spec.NodeName == "" {
		return nil, nil
	}

	annotations := pod.GetAnnotations()
	labelKey := annotations[consts.KubeAnnotationTopologyLabelKey]
	clusterTopologyName := annotations[consts.KubeAnnotationTopologyClusterTopologyName]

	targetsCapacity := 0
	if labelKey != "" {
		targetsCapacity++
	}

	var ct *grovev1alpha1.ClusterTopology
	if clusterTopologyName != "" {
		ct = &grovev1alpha1.ClusterTopology{}
		if err := r.Get(ctx, types.NamespacedName{Name: clusterTopologyName}, ct); err != nil {
			return nil, fmt.Errorf("get ClusterTopology %s: %w", clusterTopologyName, err)
		}
		targetsCapacity += len(ct.Spec.Levels)
	}

	targets := make([]topologyLabelCopyTargetSpec, 0, targetsCapacity)
	if labelKey != "" && !keyExists(pod, labelKey) {
		targets = append(targets, topologyLabelCopyTargetSpec{
			sourceLabelKey: labelKey,
			targetLabelKey: labelKey,
		})
	}

	if ct == nil {
		return targets, nil
	}

	for _, level := range ct.Spec.Levels {
		targetLabelKey := consts.DynamoTopologyLabelKey(string(level.Domain))
		if _, exists := pod.GetLabels()[targetLabelKey]; exists {
			continue
		}
		targets = append(targets, topologyLabelCopyTargetSpec{
			sourceLabelKey: level.Key,
			targetLabelKey: targetLabelKey,
		})
	}
	return targets, nil
}

func hasDynamoTopologyLabel(pod *corev1.Pod) bool {
	if pod == nil {
		return false
	}
	for labelKey := range pod.GetLabels() {
		if strings.HasPrefix(labelKey, consts.KubeLabelDynamoTopologyPrefix) {
			return true
		}
	}
	return false
}

func missingDynamoTopologyLabel(pod *corev1.Pod) bool {
	expectedLabels := expectedDynamoTopologyLabelKeys(pod)
	if len(expectedLabels) == 0 {
		return !hasDynamoTopologyLabel(pod)
	}
	labels := pod.GetLabels()
	for _, labelKey := range expectedLabels {
		if _, exists := labels[labelKey]; !exists {
			return true
		}
	}
	return false
}

func expectedDynamoTopologyLabelKeys(pod *corev1.Pod) []string {
	if pod == nil {
		return nil
	}
	const fieldPathPrefix = "metadata.labels['"
	const fieldPathSuffix = "']"

	var labelKeys []string
	for _, volume := range pod.Spec.Volumes {
		if volume.DownwardAPI == nil {
			continue
		}
		for _, item := range volume.DownwardAPI.Items {
			if item.FieldRef == nil {
				continue
			}
			fieldPath := item.FieldRef.FieldPath
			if !strings.HasPrefix(fieldPath, fieldPathPrefix) || !strings.HasSuffix(fieldPath, fieldPathSuffix) {
				continue
			}
			labelKey := strings.TrimSuffix(strings.TrimPrefix(fieldPath, fieldPathPrefix), fieldPathSuffix)
			if strings.HasPrefix(labelKey, consts.KubeLabelDynamoTopologyPrefix) {
				labelKeys = append(labelKeys, labelKey)
			}
		}
	}
	return labelKeys
}

func isDynamoComponentPod(pod *corev1.Pod) bool {
	labels := pod.GetLabels()
	return labels[consts.KubeLabelDynamoGraphDeploymentName] != "" &&
		labels[consts.KubeLabelDynamoComponent] != ""
}
