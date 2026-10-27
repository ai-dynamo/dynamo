/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pods

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// crashLoopTestPod describes a running Pod whose main container restarted
// restarts times, last exiting with exitCode, and waits to restart.
func crashLoopTestPod(name string, ready bool, restarts, exitCode int32) corev1.Pod {
	readyStatus := corev1.ConditionFalse
	if ready {
		readyStatus = corev1.ConditionTrue
	}
	pod := corev1.Pod{
		Name: name,
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: readyStatus}},
			ContainerStatuses: []corev1.ContainerStatus{{
				Name:         "main",
				RestartCount: restarts,
			}},
		},
	}
	if restarts > 0 {
		pod.Status.ContainerStatuses[0].LastTerminationState.Terminated = &corev1.ContainerStateTerminated{
			ExitCode: exitCode,
			Reason:   "Error",
		}
		pod.Status.ContainerStatuses[0].State.Waiting = &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}
	}
	return pod
}

func TestFindCrashLoop(t *testing.T) {
	initCrashLoop := crashLoopTestPod("worker-a", false, 0, 0)
	initCrashLoop.Status.InitContainerStatuses = []corev1.ContainerStatus{{
		Name:                 "setup",
		RestartCount:         3,
		State:                corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 2}},
		LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 2}},
	}}
	initCompleted := crashLoopTestPod("worker-a", false, 0, 0)
	initCompleted.Status.InitContainerStatuses = []corev1.ContainerStatus{{
		Name:                 "setup",
		RestartCount:         3,
		State:                corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 0}},
		LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 2}},
	}}
	runningAgain := crashLoopTestPod("worker-a", false, 2, 1)
	runningAgain.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	evicted := crashLoopTestPod("worker-a", false, 4, 1)
	evicted.Status.Phase = corev1.PodFailed
	terminating := crashLoopTestPod("worker-a", false, 4, 1)
	terminating.DeletionTimestamp = &metav1.Time{}

	tests := []struct {
		name  string
		pods  []corev1.Pod
		want  CrashLoop
		found bool
	}{
		{
			name:  "not-Ready Pod at the restart threshold with a non-zero exit",
			pods:  []corev1.Pod{crashLoopTestPod("worker-a", false, 2, 1)},
			want:  CrashLoop{Pod: "worker-a", Container: "main", RestartCount: 2, ExitCode: 1, Reason: "Error"},
			found: true,
		},
		{
			name:  "crash-looping init container",
			pods:  []corev1.Pod{initCrashLoop},
			want:  CrashLoop{Pod: "worker-a", Container: "setup", RestartCount: 3, ExitCode: 2},
			found: true,
		},
		{
			name: "first crash-looping Pod by name",
			pods: []corev1.Pod{
				crashLoopTestPod("worker-b", false, 5, 137),
				crashLoopTestPod("worker-a", false, 2, 1),
			},
			want:  CrashLoop{Pod: "worker-a", Container: "main", RestartCount: 2, ExitCode: 1, Reason: "Error"},
			found: true,
		},
		{
			name: "restarts below the threshold",
			pods: []corev1.Pod{crashLoopTestPod("worker-a", false, 1, 1)},
		},
		{
			name: "last run exited cleanly",
			pods: []corev1.Pod{crashLoopTestPod("worker-a", false, 2, 0)},
		},
		{
			name: "init container that completed after its crashes",
			pods: []corev1.Pod{initCompleted},
		},
		{
			name: "container running again after its crashes",
			pods: []corev1.Pod{runningAgain},
		},
		{
			name: "Ready Pod",
			pods: []corev1.Pod{crashLoopTestPod("worker-a", true, 2, 1)},
		},
		{
			name: "terminal Pod",
			pods: []corev1.Pod{evicted},
		},
		{
			name: "terminating Pod",
			pods: []corev1.Pod{terminating},
		},
		{
			name: "no Pods",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Search the Pods for a crash loop")
			got, found := FindCrashLoop(tt.pods)

			t.Log("Verify the reported container")
			assert.Equal(t, tt.found, found)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCrashLoopString(t *testing.T) {
	tests := []struct {
		name      string
		crashLoop CrashLoop
		want      string
	}{
		{
			name:      "with termination reason",
			crashLoop: CrashLoop{Pod: "worker-a", Container: "main", RestartCount: 3, ExitCode: 1, Reason: "Error"},
			want:      "pod worker-a container main restarted 3 times, last exit code 1 (Error)",
		},
		{
			name:      "without termination reason",
			crashLoop: CrashLoop{Pod: "worker-a", Container: "main", RestartCount: 2, ExitCode: 139},
			want:      "pod worker-a container main restarted 2 times, last exit code 139",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Verify the status message of the case's crash loop")
			assert.Equal(t, tt.want, tt.crashLoop.String())
		})
	}
}

func TestCrashLoopChanged(t *testing.T) {
	recovered := crashLoopTestPod("worker-a", false, 2, 1)
	recovered.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	terminal := crashLoopTestPod("worker-a", false, 2, 1)
	terminal.Status.Phase = corev1.PodFailed
	terminating := crashLoopTestPod("worker-a", false, 2, 1)
	terminating.DeletionTimestamp = &metav1.Time{}

	t.Log("Describe a Pod whose crash loop moves between two containers while one container stays waiting")
	firstCrashing := crashLoopTestPod("worker-a", false, 2, 1)
	firstCrashing.Status.ContainerStatuses = append(firstCrashing.Status.ContainerStatuses, corev1.ContainerStatus{
		Name:  "sidecar",
		State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}},
	})
	secondCrashing := crashLoopTestPod("worker-a", false, 0, 0)
	secondCrashing.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	secondCrashing.Status.ContainerStatuses = append(secondCrashing.Status.ContainerStatuses, corev1.ContainerStatus{
		Name:                 "sidecar",
		RestartCount:         2,
		State:                corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}},
		LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1}},
	})

	tests := []struct {
		name   string
		oldPod corev1.Pod
		newPod corev1.Pod
		want   bool
	}{
		{
			name:   "crash loop starts",
			oldPod: crashLoopTestPod("worker-a", false, 1, 1),
			newPod: crashLoopTestPod("worker-a", false, 2, 1),
			want:   true,
		},
		{
			name:   "container recovers",
			oldPod: crashLoopTestPod("worker-a", false, 2, 1),
			newPod: recovered,
			want:   true,
		},
		{
			name:   "restart count changes",
			oldPod: crashLoopTestPod("worker-a", false, 2, 1),
			newPod: crashLoopTestPod("worker-a", false, 3, 1),
			want:   true,
		},
		{
			name:   "exit code changes",
			oldPod: crashLoopTestPod("worker-a", false, 2, 1),
			newPod: crashLoopTestPod("worker-a", false, 2, 137),
			want:   true,
		},
		{
			name:   "crash loop moves to another container",
			oldPod: firstCrashing,
			newPod: secondCrashing,
			want:   true,
		},
		{
			name:   "Pod becomes Ready",
			oldPod: crashLoopTestPod("worker-a", false, 2, 1),
			newPod: crashLoopTestPod("worker-a", true, 2, 1),
			want:   true,
		},
		{
			name:   "Pod becomes terminal",
			oldPod: crashLoopTestPod("worker-a", false, 2, 1),
			newPod: terminal,
			want:   true,
		},
		{
			name:   "Pod starts terminating",
			oldPod: crashLoopTestPod("worker-a", false, 2, 1),
			newPod: terminating,
			want:   true,
		},
		{
			name:   "unchanged crash loop",
			oldPod: crashLoopTestPod("worker-a", false, 2, 1),
			newPod: crashLoopTestPod("worker-a", false, 2, 1),
		},
		{
			name:   "unchanged healthy Pod",
			oldPod: crashLoopTestPod("worker-a", true, 0, 0),
			newPod: crashLoopTestPod("worker-a", true, 0, 0),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Compare the diagnoses of the two Pod states")
			assert.Equal(t, tt.want, CrashLoopChanged(&tt.oldPod, &tt.newPod))
		})
	}
}
