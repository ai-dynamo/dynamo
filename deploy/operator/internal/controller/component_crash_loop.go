/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"sort"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

const (
	// crashLoopRestartThreshold is the restart count from which a container of a
	// not-Ready Pod that is down after a non-zero exit is crash-looping.
	crashLoopRestartThreshold = 2

	// dcdReasonContainerCrashLooping is the DCD Available=False reason that the
	// DGD projects into a failed state.
	dcdReasonContainerCrashLooping = "ContainerCrashLooping"

	// deploymentRevisionAnnotation is the revision the Deployment controller
	// stamps on a Deployment and on the ReplicaSet of each of its templates.
	deploymentRevisionAnnotation = "deployment.kubernetes.io/revision"
)

// containerCrashLoop identifies a crash-looping container and its last exit.
type containerCrashLoop struct {
	pod          string
	container    string
	restartCount int32
	exitCode     int32
	reason       string
}

func (c containerCrashLoop) String() string {
	message := fmt.Sprintf(
		"pod %s container %s restarted %d times, last exit code %d",
		c.pod, c.container, c.restartCount, c.exitCode,
	)
	if c.reason != "" {
		message += fmt.Sprintf(" (%s)", c.reason)
	}
	return message
}

// findCrashLoop returns the first crash-looping container, ordered by Pod name,
// among the non-terminal Pods labeled with the non-empty podTemplateHash.
func findCrashLoop(pods []corev1.Pod, podTemplateHash string) (containerCrashLoop, bool) {
	candidates := make([]*corev1.Pod, 0, len(pods))
	for i := range pods {
		pod := &pods[i]
		if pod.Labels[appsv1.DefaultDeploymentUniqueLabelKey] != podTemplateHash ||
			isTerminalPhase(pod.Status.Phase) ||
			podReady(pod) {
			continue
		}
		candidates = append(candidates, pod)
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Name < candidates[j].Name
	})

	for _, pod := range candidates {
		for _, statuses := range [][]corev1.ContainerStatus{
			pod.Status.InitContainerStatuses,
			pod.Status.ContainerStatuses,
		} {
			for _, status := range statuses {
				last := status.LastTerminationState.Terminated
				if status.RestartCount < crashLoopRestartThreshold || last == nil || last.ExitCode == 0 || !crashing(status) {
					continue
				}
				return containerCrashLoop{
					pod:          pod.Name,
					container:    status.Name,
					restartCount: status.RestartCount,
					exitCode:     last.ExitCode,
					reason:       last.Reason,
				}, true
			}
		}
	}
	return containerCrashLoop{}, false
}

// currentPodTemplateHash returns the pod-template-hash of the ReplicaSet at the
// Deployment's current revision. It reports false until the Deployment
// controller has observed the latest spec and that ReplicaSet is cached.
func currentPodTemplateHash(
	deployment *appsv1.Deployment,
	replicaSets []metav1.PartialObjectMetadata,
) (string, bool) {
	revision := deployment.Annotations[deploymentRevisionAnnotation]
	if deployment.Status.ObservedGeneration < deployment.Generation || revision == "" {
		return "", false
	}
	for i := range replicaSets {
		replicaSet := &replicaSets[i]
		if !metav1.IsControlledBy(replicaSet, deployment) ||
			replicaSet.Annotations[deploymentRevisionAnnotation] != revision {
			continue
		}
		hash := replicaSet.Labels[appsv1.DefaultDeploymentUniqueLabelKey]
		return hash, hash != ""
	}
	return "", false
}

// findDeploymentCrashLoop reports a crash-looping container among the Pods of
// the Deployment's current ReplicaSet.
func (r *DynamoComponentDeploymentReconciler) findDeploymentCrashLoop(
	ctx context.Context,
	deployment *appsv1.Deployment,
) (containerCrashLoop, bool, error) {
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return containerCrashLoop{}, false, fmt.Errorf("parse Deployment %s selector: %w", deployment.Name, err)
	}
	listOptions := []client.ListOption{
		client.InNamespace(deployment.Namespace),
		client.MatchingLabelsSelector{Selector: selector},
	}

	replicaSets := &metav1.PartialObjectMetadataList{}
	replicaSets.SetGroupVersionKind(appsv1.SchemeGroupVersion.WithKind("ReplicaSetList"))
	if err := r.List(ctx, replicaSets, listOptions...); err != nil {
		return containerCrashLoop{}, false, fmt.Errorf("list ReplicaSets of Deployment %s: %w", deployment.Name, err)
	}
	podTemplateHash, ok := currentPodTemplateHash(deployment, replicaSets.Items)
	if !ok {
		return containerCrashLoop{}, false, nil
	}

	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, listOptions...); err != nil {
		return containerCrashLoop{}, false, fmt.Errorf("list Pods of Deployment %s: %w", deployment.Name, err)
	}
	crashLoop, found := findCrashLoop(pods.Items, podTemplateHash)
	return crashLoop, found, nil
}

// crashing reports whether a container is down after a failed run: waiting to
// restart, or terminated non-zero. A completed init container is not.
func crashing(status corev1.ContainerStatus) bool {
	if terminated := status.State.Terminated; terminated != nil {
		return terminated.ExitCode != 0
	}
	return status.State.Waiting != nil
}

func podReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

// deploymentPodDCDName returns the DCD whose Deployment manages obj. A DCD's
// Deployment shares its name and selects Pods by that name.
func deploymentPodDCDName(obj client.Object) (string, bool) {
	owner := metav1.GetControllerOf(obj)
	name := obj.GetLabels()[commonconsts.KubeLabelDynamoSelector]
	return name, owner != nil && owner.Kind == "ReplicaSet" && name != ""
}

func mapDeploymentPodToDCDRequests(_ context.Context, obj client.Object) []ctrl.Request {
	name, ok := deploymentPodDCDName(obj)
	if !ok {
		return nil
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: name}}}
}

type podCrashLoopSignal struct {
	ready    bool
	terminal bool
	restarts int32
	crashing int
}

func crashLoopSignalOf(pod *corev1.Pod) podCrashLoopSignal {
	signal := podCrashLoopSignal{
		ready:    podReady(pod),
		terminal: isTerminalPhase(pod.Status.Phase),
	}
	for _, statuses := range [][]corev1.ContainerStatus{
		pod.Status.InitContainerStatuses,
		pod.Status.ContainerStatuses,
	} {
		for _, status := range statuses {
			signal.restarts += status.RestartCount
			if crashing(status) {
				signal.crashing++
			}
		}
	}
	return signal
}

// dcdPodCrashLoopPredicate admits Deployment Pod events that can start or end
// a crash loop. Deployment status does not change when a container restarts.
func dcdPodCrashLoopPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(event.CreateEvent) bool {
			return false
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			_, ok := deploymentPodDCDName(e.Object)
			return ok
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			if _, ok := deploymentPodDCDName(e.ObjectNew); !ok {
				return false
			}
			oldPod, oldOK := e.ObjectOld.(*corev1.Pod)
			newPod, newOK := e.ObjectNew.(*corev1.Pod)
			return oldOK && newOK && crashLoopSignalOf(oldPod) != crashLoopSignalOf(newPod)
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}
