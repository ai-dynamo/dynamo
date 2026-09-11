/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
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
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get
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
		// Resolve the trigger's runtime partition from Dynamo's generated ConfigMap.
		partitionByPodIndex, err := r.runtimePartitionByPodIndex(ctx, trigger)
		if err != nil {
			return nil, err
		}
		triggerPartition, ok := partitionByPodIndex[trigger.Labels[grovecommon.LabelPodCliquePodIndex]]
		if !ok {
			return nil, fmt.Errorf("LPU-GPU eviction trigger pod %s/%s has no runtime partition for Grove pod index %q", trigger.Namespace, trigger.Name, trigger.Labels[grovecommon.LabelPodCliquePodIndex])
		}

		// Keep only Agent Pods in the trigger's Dynamo-authored runtime partition.
		selected := pods[:0]
		triggerConfigHash := trigger.Annotations[commonconsts.AnnotationExtraResourcesHash]
		for _, pod := range pods {
			// Ignore Pods from a concurrent rollout with a different runtime table.
			if pod.Annotations[commonconsts.AnnotationExtraResourcesHash] != triggerConfigHash {
				continue
			}
			partition, ok := partitionByPodIndex[pod.Labels[grovecommon.LabelPodCliquePodIndex]]
			if !ok {
				return nil, fmt.Errorf("LPU-GPU eviction candidate pod %s/%s has no runtime partition for Grove pod index %q", pod.Namespace, pod.Name, pod.Labels[grovecommon.LabelPodCliquePodIndex])
			}
			if partition == triggerPartition {
				selected = append(selected, pod)
			}
		}
		pods = selected
	default:
		return nil, nil
	}
	if len(pods) == 0 {
		return nil, fmt.Errorf("matched LPU eviction mode returned no pods for %s/%s", trigger.Namespace, trigger.Name)
	}
	return pods, nil
}

func (r *lpuEvictionReconciler) runtimePartitionByPodIndex(ctx context.Context, trigger *corev1.Pod) (map[string]int, error) {
	// Require the immutable runtime-table identity stamped on the trigger Pod.
	triggerConfigHash := trigger.Annotations[commonconsts.AnnotationExtraResourcesHash]
	if triggerConfigHash == "" {
		return nil, fmt.Errorf("LPU-GPU eviction trigger pod %s/%s has no runtime ConfigMap hash", trigger.Namespace, trigger.Name)
	}

	// Read the generated runtime partition table independently of authored volume overrides.
	pcsName := trigger.Labels[grovecommon.LabelPartOfKey]
	if pcsName == "" {
		return nil, fmt.Errorf("LPU-GPU eviction trigger pod %s/%s has no PodCliqueSet identity", trigger.Namespace, trigger.Name)
	}
	configName := dynamolpx.LPUConfigMapName(pcsName)
	var config corev1.ConfigMap
	if err := r.Get(ctx, client.ObjectKey{Namespace: trigger.Namespace, Name: configName}, &config); err != nil {
		return nil, fmt.Errorf("get LPU runtime ConfigMap %s/%s: %w", trigger.Namespace, configName, err)
	}

	// Fail closed when a rollout has already replaced the trigger's runtime table.
	configHash := dynamolpx.LPUConfigMapHash(&config)
	if configHash != triggerConfigHash {
		return nil, fmt.Errorf("LPU runtime ConfigMap %s/%s does not match trigger pod %s/%s", config.Namespace, config.Name, trigger.Namespace, trigger.Name)
	}

	// Require one node count and offset for every runtime partition row.
	counts := strings.Fields(config.Data["nodes_per_partition"])
	offsets := strings.Fields(config.Data["partition_node_offsets"])
	if len(counts) == 0 || len(counts) != len(offsets) {
		return nil, fmt.Errorf("LPU runtime ConfigMap %s/%s has inconsistent partition node counts and offsets", config.Namespace, config.Name)
	}

	// Expand each non-overlapping row into the Grove pod indexes it owns.
	partitionByPodIndex := make(map[string]int)
	for partition := range counts {
		count, countErr := strconv.Atoi(counts[partition])
		offset, offsetErr := strconv.Atoi(offsets[partition])
		if countErr != nil || count < 1 || offsetErr != nil || offset < 0 {
			return nil, fmt.Errorf("LPU runtime ConfigMap %s/%s has invalid partition row %d", config.Namespace, config.Name, partition)
		}
		for rank := 0; rank < count; rank++ {
			podIndex := strconv.Itoa(offset + rank)
			if _, duplicate := partitionByPodIndex[podIndex]; duplicate {
				return nil, fmt.Errorf("LPU runtime ConfigMap %s/%s maps Grove pod index %s more than once", config.Namespace, config.Name, podIndex)
			}
			partitionByPodIndex[podIndex] = partition
		}
	}
	return partitionByPodIndex, nil
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

	return slices.DeleteFunc(pods.Items, func(pod corev1.Pod) bool {
		return !r.isLPUAgentPod(&pod) ||
			model != "" && pod.Annotations[lpxv1alpha1.PodModelAnnotation] != model
	}), nil
}

func (r *lpuEvictionReconciler) deletePods(ctx context.Context, trigger *corev1.Pod, pods []corev1.Pod) error {
	deleted := 0
	for i := range pods {
		pod := &pods[i]
		if pod.DeletionTimestamp != nil {
			continue
		}
		uid := pod.UID
		if err := r.Delete(ctx, pod, client.GracePeriodSeconds(0), client.Preconditions{UID: &uid}); err != nil {
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
