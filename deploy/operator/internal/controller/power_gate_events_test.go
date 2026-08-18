/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestPowerGateTerminationEmitsStableWarningEvent(t *testing.T) {
	message := "dynamo-power-gate: enforcement_timeout: pod_identity_mismatch"
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "test", UID: "pod-uid"},
		Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{{
			Name: consts.MainContainerName,
			LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
				ExitCode: 1,
				Message:  message + "\n",
			}},
		}}},
	}
	recorder := events.NewFakeRecorder(1)
	reconciler := &DynamoGraphDeploymentReconciler{Recorder: recorder}

	reconciler.emitPowerGateFailureEvents("dgd-uid", []corev1.Pod{pod})

	select {
	case recorded := <-recorder.Events:
		if !strings.Contains(recorded, corev1.EventTypeWarning) ||
			!strings.Contains(recorded, "PowerGateEnforcementTimeout") ||
			!strings.Contains(recorded, message) {
			t.Fatalf("event = %q, want stable power-gate warning", recorded)
		}
	default:
		t.Fatal("power-gate failure event was not emitted")
	}

	t.Log("Do not re-emit an unchanged termination on an unrelated reconciliation")
	reconciler.emitPowerGateFailureEvents("dgd-uid", []corev1.Pod{pod})
	select {
	case duplicate := <-recorder.Events:
		t.Fatalf("duplicate event = %q", duplicate)
	default:
	}

	t.Log("Do not treat an old gate failure as ongoing after the container recovers")
	recovered := pod.DeepCopy()
	recovered.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	reconciler.emitPowerGateFailureEvents("dgd-uid", []corev1.Pod{*recovered})
	select {
	case stale := <-recorder.Events:
		t.Fatalf("recovered-container event = %q", stale)
	default:
	}

	t.Log("Do not fall through from a current backend failure to an old gate failure")
	nonGateFailure := pod.DeepCopy()
	nonGateFailure.Status.ContainerStatuses[0].State = corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
		ExitCode: 23,
		Message:  "backend exited after successful startup",
	}}
	if occurrence := powerGateTerminationOccurrence(nonGateFailure); occurrence.Message != "" {
		t.Fatalf("non-gate current termination resolved to stale occurrence %#v", occurrence)
	}
}

func TestPowerGateTerminationMessageTransitionIsWatched(t *testing.T) {
	labels := map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: "graph",
		consts.KubeLabelDynamoComponent:           "worker",
		consts.KubeLabelDynamoSelector:            "worker",
		consts.KubeLabelDynamoComponentType:       "worker",
	}
	oldPod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels}}
	newPod := oldPod.DeepCopy()
	newPod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name: consts.MainContainerName,
		State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
			ExitCode: 1,
			Message:  "dynamo-power-gate: enforcement_timeout: report_not_fresh",
		}},
	}}

	if !dgdWorkerPodEventPredicate().Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}) {
		t.Fatal("power-gate termination-message transition was not observed")
	}
}

func TestPowerGateEventTrackerForgetsDeletedDGDScope(t *testing.T) {
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "test", UID: "pod-uid"},
		Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{{
			Name: consts.MainContainerName,
			State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
				Message: "dynamo-power-gate: enforcement_timeout: report_missing",
			}},
		}}},
	}
	tracker := powerGateEventTracker{}
	if unseen := tracker.unseen("dgd-uid", []corev1.Pod{pod}); len(unseen) != 1 {
		t.Fatalf("initial unseen indexes = %v, want one", unseen)
	}
	if unseen := tracker.unseen("dgd-uid", []corev1.Pod{pod}); len(unseen) != 0 {
		t.Fatalf("repeated unseen indexes = %v, want none", unseen)
	}

	tracker.forgetScope("dgd-uid")
	if unseen := tracker.unseen("dgd-uid", []corev1.Pod{pod}); len(unseen) != 1 {
		t.Fatalf("post-delete unseen indexes = %v, want one after scope reset", unseen)
	}
}
