/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/event"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// crashLoopTestPod describes a running Pod of one ReplicaSet template whose main
// container restarted restarts times, last exiting with exitCode, and waits to
// restart.
func crashLoopTestPod(name, podTemplateHash string, ready bool, restarts, exitCode int32) corev1.Pod {
	readyStatus := corev1.ConditionFalse
	if ready {
		readyStatus = corev1.ConditionTrue
	}
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{appsv1.DefaultDeploymentUniqueLabelKey: podTemplateHash},
		},
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: readyStatus}},
			ContainerStatuses: []corev1.ContainerStatus{{
				Name:         commonconsts.MainContainerName,
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
	initCrashLoop := crashLoopTestPod("worker-a", "current", false, 0, 0)
	initCrashLoop.Status.InitContainerStatuses = []corev1.ContainerStatus{{
		Name:         "setup",
		RestartCount: 3,
		State:        corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 2}},
		LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
			ExitCode: 2,
		}},
	}}
	initCompleted := crashLoopTestPod("worker-a", "current", false, 0, 0)
	initCompleted.Status.InitContainerStatuses = []corev1.ContainerStatus{{
		Name:                 "setup",
		RestartCount:         3,
		State:                corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 0}},
		LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 2}},
	}}
	runningAgain := crashLoopTestPod("worker-a", "current", false, 2, 1)
	runningAgain.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	evicted := crashLoopTestPod("worker-a", "current", false, 4, 1)
	evicted.Status.Phase = corev1.PodFailed

	tests := []struct {
		name  string
		pods  []corev1.Pod
		want  containerCrashLoop
		found bool
	}{
		{
			name:  "current not-Ready Pod at the restart threshold with a non-zero exit",
			pods:  []corev1.Pod{crashLoopTestPod("worker-a", "current", false, 2, 1)},
			want:  containerCrashLoop{pod: "worker-a", container: commonconsts.MainContainerName, restartCount: 2, exitCode: 1, reason: "Error"},
			found: true,
		},
		{
			name:  "crash-looping init container",
			pods:  []corev1.Pod{initCrashLoop},
			want:  containerCrashLoop{pod: "worker-a", container: "setup", restartCount: 3, exitCode: 2},
			found: true,
		},
		{
			name: "first crash-looping Pod by name",
			pods: []corev1.Pod{
				crashLoopTestPod("worker-b", "current", false, 5, 137),
				crashLoopTestPod("worker-a", "current", false, 2, 1),
			},
			want:  containerCrashLoop{pod: "worker-a", container: commonconsts.MainContainerName, restartCount: 2, exitCode: 1, reason: "Error"},
			found: true,
		},
		{
			name: "restarts below the threshold",
			pods: []corev1.Pod{crashLoopTestPod("worker-a", "current", false, 1, 1)},
		},
		{
			name: "last run exited cleanly",
			pods: []corev1.Pod{crashLoopTestPod("worker-a", "current", false, 2, 0)},
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
			pods: []corev1.Pod{crashLoopTestPod("worker-a", "current", true, 2, 1)},
		},
		{
			name: "Pod of an older template",
			pods: []corev1.Pod{crashLoopTestPod("worker-a", "previous", false, 9, 1)},
		},
		{
			name: "terminal Pod",
			pods: []corev1.Pod{evicted},
		},
		{
			name: "no Pods",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Search the current template's Pods for a crash loop")
			got, found := findCrashLoop(tt.pods, "current")

			t.Log("Verify the reported container")
			assert.Equal(t, tt.found, found)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestContainerCrashLoopString(t *testing.T) {
	tests := []struct {
		name      string
		crashLoop containerCrashLoop
		want      string
	}{
		{
			name:      "with termination reason",
			crashLoop: containerCrashLoop{pod: "worker-a", container: "main", restartCount: 3, exitCode: 1, reason: "Error"},
			want:      "pod worker-a container main restarted 3 times, last exit code 1 (Error)",
		},
		{
			name:      "without termination reason",
			crashLoop: containerCrashLoop{pod: "worker-a", container: "main", restartCount: 2, exitCode: 139},
			want:      "pod worker-a container main restarted 2 times, last exit code 139",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.crashLoop.String())
		})
	}
}

// crashLoopTestReplicaSet describes the cached metadata of a ReplicaSet
// controlled by the Deployment with UID owner.
func crashLoopTestReplicaSet(owner types.UID, revision, hash string) metav1.PartialObjectMetadata {
	return metav1.PartialObjectMetadata{ObjectMeta: metav1.ObjectMeta{
		Name:            "worker-" + hash,
		Labels:          map[string]string{appsv1.DefaultDeploymentUniqueLabelKey: hash},
		Annotations:     map[string]string{deploymentRevisionAnnotation: revision},
		OwnerReferences: []metav1.OwnerReference{{Kind: "Deployment", UID: owner, Controller: ptr.To(true)}},
	}}
}

// crashLoopTestDeploymentPod describes a Pod that the Deployment of DCD
// graph-worker manages.
func crashLoopTestDeploymentPod(restarts int32, ready bool, phase corev1.PodPhase) *corev1.Pod {
	pod := crashLoopTestPod("worker-a", "current", ready, restarts, 1)
	pod.Namespace = "inference"
	pod.Labels[commonconsts.KubeLabelDynamoSelector] = "graph-worker"
	pod.OwnerReferences = []metav1.OwnerReference{{Kind: "ReplicaSet", Name: "graph-worker-current", Controller: ptr.To(true)}}
	pod.Status.Phase = phase
	return &pod
}

func TestCurrentPodTemplateHash(t *testing.T) {
	deploymentUID := types.UID("deployment-uid")

	tests := []struct {
		name               string
		revision           string
		observedGeneration int64
		replicaSets        []metav1.PartialObjectMetadata
		wantHash           string
		wantOK             bool
	}{
		{
			name:               "ReplicaSet at the current revision",
			revision:           "3",
			observedGeneration: 2,
			replicaSets: []metav1.PartialObjectMetadata{
				crashLoopTestReplicaSet(deploymentUID, "2", "previous"),
				crashLoopTestReplicaSet(deploymentUID, "3", "current"),
			},
			wantHash: "current",
			wantOK:   true,
		},
		{
			name:               "Deployment controller has not observed the latest spec",
			revision:           "3",
			observedGeneration: 1,
			replicaSets:        []metav1.PartialObjectMetadata{crashLoopTestReplicaSet(deploymentUID, "3", "current")},
		},
		{
			name:               "current ReplicaSet not cached yet",
			revision:           "3",
			observedGeneration: 2,
			replicaSets:        []metav1.PartialObjectMetadata{crashLoopTestReplicaSet(deploymentUID, "2", "previous")},
		},
		{
			name:               "ReplicaSet controlled by another Deployment",
			revision:           "3",
			observedGeneration: 2,
			replicaSets:        []metav1.PartialObjectMetadata{crashLoopTestReplicaSet("other-uid", "3", "current")},
		},
		{
			name:               "Deployment without a revision",
			observedGeneration: 2,
			replicaSets:        []metav1.PartialObjectMetadata{crashLoopTestReplicaSet(deploymentUID, "3", "current")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a Deployment at generation 2")
			deployment := &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker", UID: deploymentUID, Generation: 2},
				Status:     appsv1.DeploymentStatus{ObservedGeneration: tt.observedGeneration},
			}
			if tt.revision != "" {
				deployment.Annotations = map[string]string{deploymentRevisionAnnotation: tt.revision}
			}

			t.Log("Resolve the current template hash")
			hash, ok := currentPodTemplateHash(deployment, tt.replicaSets)

			t.Log("Verify the resolved hash")
			assert.Equal(t, tt.wantOK, ok)
			assert.Equal(t, tt.wantHash, hash)
		})
	}
}

func TestDCDPodCrashLoopPredicate(t *testing.T) {
	unmanaged := crashLoopTestDeploymentPod(2, false, corev1.PodRunning)
	unmanaged.OwnerReferences[0].Kind = "LeaderWorkerSet"
	restarted := crashLoopTestDeploymentPod(2, false, corev1.PodRunning)
	restarted.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}

	tests := []struct {
		name   string
		oldPod *corev1.Pod
		newPod *corev1.Pod
		want   bool
	}{
		{
			name:   "container restart",
			oldPod: crashLoopTestDeploymentPod(1, false, corev1.PodRunning),
			newPod: crashLoopTestDeploymentPod(2, false, corev1.PodRunning),
			want:   true,
		},
		{
			name:   "restarted container crashes again",
			oldPod: restarted,
			newPod: crashLoopTestDeploymentPod(2, false, corev1.PodRunning),
			want:   true,
		},
		{
			name:   "Pod becomes Ready",
			oldPod: crashLoopTestDeploymentPod(2, false, corev1.PodRunning),
			newPod: crashLoopTestDeploymentPod(2, true, corev1.PodRunning),
			want:   true,
		},
		{
			name:   "Pod becomes terminal",
			oldPod: crashLoopTestDeploymentPod(2, false, corev1.PodRunning),
			newPod: crashLoopTestDeploymentPod(2, false, corev1.PodFailed),
			want:   true,
		},
		{
			name:   "unrelated status change",
			oldPod: crashLoopTestDeploymentPod(2, false, corev1.PodRunning),
			newPod: crashLoopTestDeploymentPod(2, false, corev1.PodRunning),
		},
		{
			name:   "Pod not managed by a DCD Deployment",
			oldPod: crashLoopTestDeploymentPod(1, false, corev1.PodRunning),
			newPod: unmanaged,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dcdPodCrashLoopPredicate().Update(event.UpdateEvent{ObjectOld: tt.oldPod, ObjectNew: tt.newPod})
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMapDeploymentPodToDCDRequests(t *testing.T) {
	unmanaged := crashLoopTestDeploymentPod(2, false, corev1.PodRunning)
	unmanaged.OwnerReferences = nil

	tests := []struct {
		name string
		pod  *corev1.Pod
		want []ctrl.Request
	}{
		{
			name: "Pod of a DCD Deployment",
			pod:  crashLoopTestDeploymentPod(2, false, corev1.PodRunning),
			want: []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "inference", Name: "graph-worker"}}},
		},
		{
			name: "Pod without a ReplicaSet controller",
			pod:  unmanaged,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, mapDeploymentPodToDCDRequests(t.Context(), tt.pod))
		})
	}
}
