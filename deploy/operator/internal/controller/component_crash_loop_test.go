/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// crashLoopTestPod describes a running Pod labeled for DCD dcdName whose main
// container restarted restarts times, last exiting with code 1, and waits to restart.
func crashLoopTestPod(namespace, name, dcdName string, ready bool, restarts int32) *corev1.Pod {
	readyStatus := corev1.ConditionFalse
	if ready {
		readyStatus = corev1.ConditionTrue
	}
	pod := &corev1.Pod{
		Name:      name,
		Namespace: namespace,
		Labels:    map[string]string{commonconsts.KubeLabelDynamoSelector: dcdName},
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
			ExitCode: 1,
			Reason:   "Error",
		}
		pod.Status.ContainerStatuses[0].State.Waiting = &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}
	}
	return pod
}

func TestPodCrashLoopPredicate(t *testing.T) {
	restarted := crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2)
	restarted.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	terminal := crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2)
	terminal.Status.Phase = corev1.PodFailed
	terminating := crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2)
	terminating.DeletionTimestamp = &metav1.Time{}

	tests := []struct {
		name   string
		oldPod *corev1.Pod
		newPod *corev1.Pod
		want   bool
	}{
		{
			name:   "container reaches the restart threshold",
			oldPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 1),
			newPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			want:   true,
		},
		{
			name:   "crash-looping container restarts again",
			oldPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			newPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 3),
			want:   true,
		},
		{
			name:   "restarted container crashes again",
			oldPod: restarted,
			newPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			want:   true,
		},
		{
			name:   "Pod becomes Ready",
			oldPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			newPod: crashLoopTestPod("inference", "worker-a", "graph-worker", true, 2),
			want:   true,
		},
		{
			name:   "Pod becomes terminal",
			oldPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			newPod: terminal,
			want:   true,
		},
		{
			name:   "Pod starts terminating",
			oldPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			newPod: terminating,
			want:   true,
		},
		{
			name:   "unrelated status change",
			oldPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			newPod: crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
		},
		{
			name:   "Pod without the selector label",
			oldPod: crashLoopTestPod("inference", "worker-a", "", false, 1),
			newPod: crashLoopTestPod("inference", "worker-a", "", false, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := podCrashLoopPredicate(deploymentPodDCDName).Update(event.UpdateEvent{ObjectOld: tt.oldPod, ObjectNew: tt.newPod})
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMapPodToRequests(t *testing.T) {
	dgdComponentPod := crashLoopTestPod("inference", "graph-frontend-a", "graph-frontend", false, 2)
	dgdComponentPod.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = "graph"
	dgdComponentPod.Labels[commonconsts.KubeLabelDynamoComponent] = "frontend"
	graphOnlyPod := crashLoopTestPod("inference", "graph-frontend-a", "", false, 2)
	graphOnlyPod.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = "graph"

	tests := []struct {
		name        string
		requestName func(client.Object) (string, bool)
		pod         *corev1.Pod
		want        []ctrl.Request
	}{
		{
			name:        "Pod labeled for a DCD",
			requestName: deploymentPodDCDName,
			pod:         crashLoopTestPod("inference", "worker-a", "graph-worker", false, 2),
			want:        []ctrl.Request{{Namespace: "inference", Name: "graph-worker"}},
		},
		{
			name:        "Pod without the selector label",
			requestName: deploymentPodDCDName,
			pod:         crashLoopTestPod("inference", "worker-a", "", false, 2),
		},
		{
			name:        "Pod labeled for a DGD component",
			requestName: componentPodDGDName,
			pod:         dgdComponentPod,
			want:        []ctrl.Request{{Namespace: "inference", Name: "graph"}},
		},
		{
			name:        "Pod labeled for a DGD without a component",
			requestName: componentPodDGDName,
			pod:         graphOnlyPod,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, mapPodToRequests(tt.requestName)(t.Context(), tt.pod))
		})
	}
}

func TestFindComponentPodCrashLoops(t *testing.T) {
	t.Log("Label Pods for two components of graph and one component of another graph")
	componentPod := func(name, dgdName, component string, restarts int32) *corev1.Pod {
		pod := crashLoopTestPod("inference", name, "", false, restarts)
		pod.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = dgdName
		pod.Labels[commonconsts.KubeLabelDynamoComponent] = component
		return pod
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithIndex(&corev1.Pod{}, dgdComponentPodIndex, dgdComponentPodIndexValues).
		WithObjects(
			componentPod("graph-frontend-a", "graph", "frontend", 3),
			componentPod("graph-worker-a", "graph", "worker", 1),
			componentPod("other-worker-a", "other", "worker", 4),
		).
		Build()
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "inference"}
	components := []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "frontend"}, {ComponentName: "worker"}}

	t.Log("Diagnose each component from the Pods labeled for it")
	crashLoops, err := findComponentPodCrashLoops(t.Context(), kubeClient, dgd, components)

	t.Log("Verify only the crash-looping component of graph is reported")
	require.NoError(t, err)
	assert.Equal(t, map[string]string{
		"frontend": "pod graph-frontend-a container main restarted 3 times, last exit code 1 (Error)",
	}, crashLoops)
}
