/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"strings"
	"sync"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

const powerGateTerminationPrefix = "dynamo-power-gate: "

type powerGateTermination struct {
	Message     string
	ContainerID string
	FinishedAt  time.Time
}

type trackedPowerGateTermination struct {
	Scope      string
	Occurrence powerGateTermination
}

type powerGateEventTracker struct {
	mu         sync.Mutex
	observed   map[string]trackedPowerGateTermination
	successful map[string]string
}

func (tracker *powerGateEventTracker) forgetScope(scope string) {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	for podID, previous := range tracker.observed {
		if previous.Scope == scope {
			delete(tracker.observed, podID)
		}
	}
	for podID, previousScope := range tracker.successful {
		if previousScope == scope {
			delete(tracker.successful, podID)
		}
	}
}

func powerGateTerminationOccurrence(pod *corev1.Pod) powerGateTermination {
	if pod == nil {
		return powerGateTermination{}
	}
	for i := range pod.Status.ContainerStatuses {
		status := &pod.Status.ContainerStatuses[i]
		if status.Name != consts.MainContainerName {
			continue
		}

		// A current termination is authoritative. Never fall through from a
		// non-gate current failure to an older gate failure.
		if status.State.Terminated != nil {
			return newPowerGateTermination(status.State.Terminated)
		}
		// A healthy running container has recovered; its last termination is
		// history rather than an ongoing gate failure.
		if status.State.Running != nil {
			return powerGateTermination{}
		}
		// During CrashLoopBackOff kubelet exposes the failure only as the last
		// termination while current state is waiting.
		if status.LastTerminationState.Terminated != nil {
			return newPowerGateTermination(status.LastTerminationState.Terminated)
		}
	}
	return powerGateTermination{}
}

func newPowerGateTermination(terminated *corev1.ContainerStateTerminated) powerGateTermination {
	message := strings.TrimSpace(terminated.Message)
	if !strings.HasPrefix(message, powerGateTerminationPrefix) {
		return powerGateTermination{}
	}
	return powerGateTermination{
		Message:     message,
		ContainerID: terminated.ContainerID,
		FinishedAt:  terminated.FinishedAt.Time,
	}
}

func powerGateEventReason(message string) string {
	kind, _, _ := strings.Cut(strings.TrimPrefix(message, powerGateTerminationPrefix), ":")
	switch kind {
	case "configuration_error":
		return "PowerGateConfigurationError"
	case "enforcement_timeout":
		return "PowerGateEnforcementTimeout"
	case "exec_failed":
		return "PowerGateExecFailed"
	default:
		return "PowerGateFailed"
	}
}

func (tracker *powerGateEventTracker) unseen(scope string, pods []corev1.Pod) []int {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	if tracker.observed == nil {
		tracker.observed = make(map[string]trackedPowerGateTermination)
	}
	livePodIDs := make(map[string]struct{}, len(pods))
	unseen := make([]int, 0)
	for i := range pods {
		podID := powerGatePodID(&pods[i])
		livePodIDs[podID] = struct{}{}
		occurrence := powerGateTerminationOccurrence(&pods[i])
		if occurrence.Message == "" {
			delete(tracker.observed, podID)
			continue
		}
		if previous, exists := tracker.observed[podID]; exists && previous.Scope == scope && previous.Occurrence == occurrence {
			continue
		}
		tracker.observed[podID] = trackedPowerGateTermination{Scope: scope, Occurrence: occurrence}
		unseen = append(unseen, i)
	}
	for podID, previous := range tracker.observed {
		if previous.Scope == scope {
			if _, exists := livePodIDs[podID]; !exists {
				delete(tracker.observed, podID)
			}
		}
	}
	for podID, previousScope := range tracker.successful {
		if previousScope == scope {
			if _, exists := livePodIDs[podID]; !exists {
				delete(tracker.successful, podID)
			}
		}
	}
	return unseen
}

func (tracker *powerGateEventTracker) observeSuccess(
	scope string,
	pod *corev1.Pod,
	finishedAt time.Time,
) (time.Duration, bool, bool) {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	if tracker.successful == nil {
		tracker.successful = make(map[string]string)
	}
	podID := powerGatePodID(pod)
	if previousScope, exists := tracker.successful[podID]; exists && previousScope == scope {
		return 0, false, false
	}
	tracker.successful[podID] = scope
	elapsed, known := powerGateWaitElapsed(pod, finishedAt)
	return elapsed, known, true
}

func powerGatePodID(pod *corev1.Pod) string {
	if pod == nil {
		return ""
	}
	if pod.UID != "" {
		return string(pod.UID)
	}
	return pod.Namespace + "/" + pod.Name
}

func powerGateWaitElapsed(pod *corev1.Pod, finishedAt time.Time) (time.Duration, bool) {
	if pod == nil || finishedAt.IsZero() {
		return 0, false
	}
	startedAt := pod.CreationTimestamp.Time
	if pod.Status.StartTime != nil && !pod.Status.StartTime.IsZero() {
		startedAt = pod.Status.StartTime.Time
	}
	if startedAt.IsZero() || finishedAt.Before(startedAt) {
		return 0, false
	}
	return finishedAt.Sub(startedAt), true
}

func (r *DynamoGraphDeploymentReconciler) emitPowerGateFailureEvents(scope string, pods []corev1.Pod) {
	for _, i := range r.powerGateEvents.unseen(scope, pods) {
		occurrence := powerGateTerminationOccurrence(&pods[i])
		message := occurrence.Message
		reason := powerGateEventReason(message)
		elapsed, known := powerGateWaitElapsed(&pods[i], occurrence.FinishedAt)
		recordPowerGateWait(reason, elapsed, known)
		if r.Recorder == nil {
			continue
		}
		// The client-go events recorder aggregates identical regarding/reason/
		// action/note tuples into one Event series across repeated reconciliation.
		r.Recorder.Eventf(
			&pods[i],
			nil,
			corev1.EventTypeWarning,
			reason,
			"Start",
			"%s",
			message,
		)
	}
}
