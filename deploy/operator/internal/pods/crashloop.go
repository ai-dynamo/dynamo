/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package pods interprets Pod status independently of the workload that manages the Pods.
package pods

import (
	"fmt"
	"sort"

	corev1 "k8s.io/api/core/v1"
)

// crashLoopRestartThreshold is the restart count from which a container of a
// not-Ready Pod that is down after a non-zero exit is crash-looping.
const crashLoopRestartThreshold = 2

// CrashLoop identifies a crash-looping container and its last exit.
type CrashLoop struct {
	Pod          string
	Container    string
	RestartCount int32
	ExitCode     int32
	Reason       string
}

// String describes the crash loop for a status message.
func (c CrashLoop) String() string {
	message := fmt.Sprintf(
		"pod %s container %s restarted %d times, last exit code %d",
		c.Pod, c.Container, c.RestartCount, c.ExitCode,
	)
	if c.Reason != "" {
		message += fmt.Sprintf(" (%s)", c.Reason)
	}
	return message
}

// FindCrashLoop returns the first crash-looping container among pods, ordered
// by Pod name.
func FindCrashLoop(pods []corev1.Pod) (CrashLoop, bool) {
	sorted := make([]*corev1.Pod, len(pods))
	for i := range pods {
		sorted[i] = &pods[i]
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})

	for _, pod := range sorted {
		if crashLoop, found := CrashLoopOf(pod); found {
			return crashLoop, true
		}
	}
	return CrashLoop{}, false
}

// CrashLoopOf returns pod's first crash-looping container, init containers
// first. Ready, terminal and terminating Pods have none.
func CrashLoopOf(pod *corev1.Pod) (CrashLoop, bool) {
	if pod.DeletionTimestamp != nil ||
		pod.Status.Phase == corev1.PodFailed ||
		pod.Status.Phase == corev1.PodSucceeded ||
		ready(pod) {
		return CrashLoop{}, false
	}

	for _, statuses := range [][]corev1.ContainerStatus{
		pod.Status.InitContainerStatuses,
		pod.Status.ContainerStatuses,
	} {
		for _, status := range statuses {
			last := status.LastTerminationState.Terminated
			if status.RestartCount < crashLoopRestartThreshold || last == nil || last.ExitCode == 0 || !crashing(status) {
				continue
			}
			return CrashLoop{
				Pod:          pod.Name,
				Container:    status.Name,
				RestartCount: status.RestartCount,
				ExitCode:     last.ExitCode,
				Reason:       last.Reason,
			}, true
		}
	}
	return CrashLoop{}, false
}

// CrashLoopChanged reports whether newPod's crash-loop diagnosis differs from
// oldPod's, including one starting or ending.
func CrashLoopChanged(oldPod, newPod *corev1.Pod) bool {
	oldCrashLoop, oldFound := CrashLoopOf(oldPod)
	newCrashLoop, newFound := CrashLoopOf(newPod)
	return oldFound != newFound || oldCrashLoop != newCrashLoop
}

// crashing reports whether a container is down after a failed run: waiting to
// restart, or terminated non-zero. A completed init container is not.
func crashing(status corev1.ContainerStatus) bool {
	if terminated := status.State.Terminated; terminated != nil {
		return terminated.ExitCode != 0
	}
	return status.State.Waiting != nil
}

// ready reports whether the Pod's Ready condition is true.
func ready(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
