/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"slices"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

const (
	lpuEvictionEventReason                  = "LPUEviction"
	podDisruptionReasonTaintManagerDeletion = "DeletionByTaintManager"
	podDisruptionReasonEvictionAPI          = "EvictionByEvictionAPI"
)

type lpuEvictionReconciler struct {
	client.Client
	Recorder events.EventRecorder
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;delete;deletecollection
func (r *lpuEvictionReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	if !r.isTrigger(&pod) {
		return ctrl.Result{}, nil
	}

	pods, err := r.podsForTrigger(ctx, &pod)
	if err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, r.deletePods(ctx, &pod, pods)
}

// setupLPUEviction registers eviction of disrupted node-local LPX Agent cohorts.
func setupLPUEviction(mgr ctrl.Manager) error {
	r := &lpuEvictionReconciler{Client: mgr.GetClient(), Recorder: mgr.GetEventRecorder("lpu-eviction")}
	return ctrl.NewControllerManagedBy(mgr).
		Named("lpu-eviction").
		Watches(&corev1.Pod{}, &handler.EnqueueRequestForObject{},
			builder.WithPredicates(r.evictionPredicate()),
		).
		Complete(r)
}

func (r *lpuEvictionReconciler) podsForTrigger(ctx context.Context, trigger *corev1.Pod) ([]corev1.Pod, error) {
	owner := metav1.GetControllerOf(trigger)
	if owner == nil || owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() ||
		owner.Kind != groveconstants.KindPodClique || owner.Name == "" || owner.UID == "" {
		return nil, nil
	}
	clique := &grovev1alpha1.PodClique{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: trigger.Namespace, Name: owner.Name}, clique); err != nil {
		return nil, err
	}
	if clique.UID != owner.UID || !dynamolpx.OwnsPodClique(ctx, r.Client, clique) {
		return nil, fmt.Errorf("cannot verify ownership of LPU eviction trigger pod %s/%s", trigger.Namespace, trigger.Name)
	}

	mode := lpxv1alpha1.WorkloadMode(trigger.Annotations[dynamolpx.WorkloadModeAnnotation])
	if mode == "" {
		return nil, nil
	}
	pods, err := r.allModelPods(ctx, trigger)
	if err != nil {
		return nil, err
	}
	switch mode {
	case lpxv1alpha1.WorkloadModeV2LPUOnly, lpxv1alpha1.WorkloadModeV3HxLPUOnly:
	case lpxv1alpha1.WorkloadModeV2StrictHybrid, lpxv1alpha1.WorkloadModeV3HxStrictHybrid:
		triggerRow, err := lpxv1alpha1.ParsePodLogicalRow(trigger.Annotations)
		if err != nil {
			return nil, fmt.Errorf("parse LPU-GPU eviction trigger pod %s/%s: %w", trigger.Namespace, trigger.Name, err)
		}

		selected := pods[:0]
		for _, pod := range pods {
			row, err := lpxv1alpha1.ParsePodLogicalRow(pod.Annotations)
			if err != nil {
				return nil, fmt.Errorf("parse LPU-GPU eviction candidate pod %s/%s: %w", pod.Namespace, pod.Name, err)
			}
			if row.ModelPartitionID == triggerRow.ModelPartitionID {
				selected = append(selected, pod)
			}
		}
		pods = selected
	default:
		return nil, nil
	}
	return pods, nil
}

func (r *lpuEvictionReconciler) allModelPods(ctx context.Context, trigger *corev1.Pod) ([]corev1.Pod, error) {
	labels := trigger.GetLabels()
	dgdName := labels[commonconsts.KubeLabelDynamoGraphDeploymentName]
	component := labels[commonconsts.KubeLabelDynamoComponent]
	pcsg := labels[grovecommon.LabelPodCliqueScalingGroup]
	pcsgReplica := labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex]
	if dgdName == "" || component == "" || pcsg == "" || pcsgReplica == "" {
		return nil, fmt.Errorf("LPU eviction trigger pod %s/%s missing required labels", trigger.Namespace, trigger.Name)
	}

	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(trigger.Namespace), client.MatchingLabels{
		commonconsts.KubeLabelDynamoGraphDeploymentName:    dgdName,
		commonconsts.KubeLabelDynamoComponent:              component,
		grovecommon.LabelPodCliqueScalingGroup:             pcsg,
		grovecommon.LabelPodCliqueScalingGroupReplicaIndex: pcsgReplica,
	}); err != nil {
		return nil, fmt.Errorf("failed to list LPU pod candidates: %w", err)
	}

	model := trigger.Annotations[lpxv1alpha1.PodModelAnnotation]
	owner := metav1.GetControllerOf(trigger)

	return slices.DeleteFunc(pods.Items, func(pod corev1.Pod) bool {
		// Native updates can replace the clique or individual Pods within it.
		candidateOwner := metav1.GetControllerOf(&pod)
		return pod.DeletionTimestamp != nil || !r.isLPUAgentPod(&pod) ||
			owner == nil || candidateOwner == nil ||
			owner.APIVersion != candidateOwner.APIVersion || owner.Kind != candidateOwner.Kind ||
			owner.Name != candidateOwner.Name || owner.UID != candidateOwner.UID ||
			pod.Labels[grovecommon.LabelPodTemplateHash] != trigger.Labels[grovecommon.LabelPodTemplateHash] ||
			model != "" && pod.Annotations[lpxv1alpha1.PodModelAnnotation] != model
	}), nil
}

func (r *lpuEvictionReconciler) deletePods(ctx context.Context, trigger *corev1.Pod, pods []corev1.Pod) error {
	deleted := 0
	for i := range pods {
		pod := &pods[i]
		uid, resourceVersion := pod.UID, pod.ResourceVersion
		if err := r.Delete(ctx, pod, client.GracePeriodSeconds(0), client.Preconditions{UID: &uid, ResourceVersion: &resourceVersion}); err != nil {
			if errors.IsNotFound(err) {
				continue
			}
			return fmt.Errorf("failed to delete LPU pod %s/%s: %w", pod.Namespace, pod.Name, err)
		}
		deleted++
	}

	// Repeated observations and concurrent deletions must not report another eviction.
	if deleted == 0 {
		return nil
	}

	log.FromContext(ctx).Info("deleted LPU pod set",
		"trigger", trigger.Name,
		"deletedPods", deleted,
	)
	r.Recorder.Eventf(trigger, nil, corev1.EventTypeWarning, lpuEvictionEventReason, "Delete",
		"Pod %s is being disrupted; deleted %d LPU pods",
		trigger.Name, deleted,
	)
	return nil
}

func (r *lpuEvictionReconciler) isLPUAgentPod(pod *corev1.Pod) bool {
	componentType := pod.Labels[commonconsts.KubeLabelDynamoComponentType]
	return componentType == commonconsts.ComponentTypeLPX &&
		pod.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleAgent
}

func (r *lpuEvictionReconciler) isTrigger(pod *corev1.Pod) bool {
	if pod.Spec.SchedulerName != dynamolpx.SchedulerName || !r.isLPUAgentPod(pod) || pod.DeletionTimestamp == nil {
		return false
	}
	return slices.ContainsFunc(pod.Status.Conditions, func(condition corev1.PodCondition) bool {
		return condition.Type == corev1.DisruptionTarget &&
			condition.Status == corev1.ConditionTrue &&
			(condition.Reason == podDisruptionReasonTaintManagerDeletion || condition.Reason == podDisruptionReasonEvictionAPI)
	})
}

func (r *lpuEvictionReconciler) evictionPredicate() predicate.Predicate {
	shouldEnqueueDeletingPod := func(obj client.Object) bool {
		pod, ok := obj.(*corev1.Pod)
		return ok && r.isTrigger(pod)
	}

	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return shouldEnqueueDeletingPod(e.Object)
		},
		DeleteFunc:  func(event.DeleteEvent) bool { return false },
		GenericFunc: func(event.GenericEvent) bool { return false },
		UpdateFunc: func(e event.UpdateEvent) bool {
			return shouldEnqueueDeletingPod(e.ObjectNew)
		},
	}
}
