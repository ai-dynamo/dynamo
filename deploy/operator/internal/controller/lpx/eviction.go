/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"slices"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
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

	consts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
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
		Watches(&corev1.ConfigMap{}, handler.EnqueueRequestsFromMapFunc(r.evictionRequestsForConfigMap)).
		Complete(r)
}

func (r *lpuEvictionReconciler) evictionRequestsForConfigMap(ctx context.Context, object client.Object) []ctrl.Request {
	// Ignore ConfigMaps outside the hybrid runtime contract before listing Pods.
	config, ok := object.(*corev1.ConfigMap)
	if !ok || config.Data["partition_models"] == "" {
		return nil
	}

	// A late runtime table observation must wake disrupted Agents using that exact table.
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(config.GetNamespace()), client.MatchingLabels{
		consts.KubeLabelDynamoComponentType: consts.ComponentTypeLPX,
	}); err != nil {
		log.FromContext(ctx).Error(err, "list LPU eviction triggers for runtime ConfigMap")
		return nil
	}

	// Unrelated workloads and other immutable runtime revisions do not need another reconcile.
	var requests []ctrl.Request
	for i := range pods.Items {
		pod := &pods.Items[i]
		root := pod.Labels[grovecommon.LabelPartOfKey]
		hash := pod.Annotations[consts.AnnotationExtraResourcesHash]
		if r.isTrigger(pod) && root != "" && hash != "" && lpx.LPUConfigMapName(root, hash) == config.GetName() {
			requests = append(requests, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(pod)})
		}
	}
	return requests
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
	if clique.UID != owner.UID || !lpx.OwnsPodClique(ctx, r.Client, clique) {
		return nil, fmt.Errorf("cannot verify ownership of LPU eviction trigger pod %s/%s", trigger.Namespace, trigger.Name)
	}

	mode := lpxv1alpha1.WorkloadMode(trigger.Annotations[lpx.WorkloadModeAnnotation])
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
		// The Pod's content-addressed runtime table survives native template updates.
		root := trigger.Labels[grovecommon.LabelPartOfKey]
		hash := trigger.Annotations[consts.AnnotationExtraResourcesHash]
		if root == "" || hash == "" {
			return nil, fmt.Errorf("LPU-GPU eviction trigger pod %s/%s has no runtime ConfigMap identity", trigger.Namespace, trigger.Name)
		}
		var config corev1.ConfigMap
		key := client.ObjectKey{Namespace: trigger.Namespace, Name: lpx.LPUConfigMapName(root, hash)}
		if err := r.Get(ctx, key, &config); err != nil {
			return nil, fmt.Errorf("read LPU-GPU eviction runtime ConfigMap: %w", err)
		}
		if config.Immutable == nil || !*config.Immutable {
			return nil, fmt.Errorf("LPU-GPU eviction runtime ConfigMap %s is not immutable", key)
		}

		// Bind the table to the verified deployment and the Pod's full content hash.
		configOwner := metav1.GetControllerOf(&config)
		if configOwner == nil || configOwner.APIVersion != v1alpha1.GroupVersion.String() ||
			configOwner.Kind != v1alpha1.LPXGraphDeploymentGVK.Kind ||
			configOwner.Name != clique.Annotations[lpx.DeploymentNameAnnotation] ||
			string(configOwner.UID) != clique.Annotations[lpx.DeploymentUIDAnnotation] ||
			lpx.LPUConfigMapHash(&config) != hash {
			return nil, fmt.Errorf("cannot authenticate LPU-GPU eviction runtime ConfigMap %s", key)
		}
		triggerPartition, err := lpx.LPUAgentRuntimePartition(trigger, &config)
		if err != nil {
			return nil, fmt.Errorf("parse LPU-GPU eviction trigger pod %s/%s: %w", trigger.Namespace, trigger.Name, err)
		}

		// Select the full runtime partition before authorizing any sibling deletion.
		selected := pods[:0]
		for _, pod := range pods {
			partition, err := lpx.LPUAgentRuntimePartition(&pod, &config)
			if err != nil {
				return nil, fmt.Errorf("parse LPU-GPU eviction candidate pod %s/%s: %w", pod.Namespace, pod.Name, err)
			}
			if partition == triggerPartition {
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
	dgdName := labels[consts.KubeLabelDynamoGraphDeploymentName]
	component := labels[consts.KubeLabelDynamoComponent]
	pcsg := labels[grovecommon.LabelPodCliqueScalingGroup]
	pcsgReplica := labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex]
	if dgdName == "" || component == "" || pcsg == "" || pcsgReplica == "" {
		return nil, fmt.Errorf("LPU eviction trigger pod %s/%s missing required labels", trigger.Namespace, trigger.Name)
	}

	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(trigger.Namespace), client.MatchingLabels{
		consts.KubeLabelDynamoGraphDeploymentName:          dgdName,
		consts.KubeLabelDynamoComponent:                    component,
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
			pod.Annotations[consts.AnnotationExtraResourcesHash] != trigger.Annotations[consts.AnnotationExtraResourcesHash] ||
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
	componentType := pod.Labels[consts.KubeLabelDynamoComponentType]
	return componentType == consts.ComponentTypeLPX &&
		pod.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleAgent
}

func (r *lpuEvictionReconciler) isTrigger(pod *corev1.Pod) bool {
	if pod.Spec.SchedulerName != lpx.SchedulerName || !r.isLPUAgentPod(pod) || pod.DeletionTimestamp == nil {
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
