/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	corev1 "k8s.io/api/core/v1"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// crashLoopTestPod describes a not-Ready Pod labeled for DCD dcdName whose main
// container restarted three times, last exiting with code 1, and waits to restart.
func crashLoopTestPod(name, dcdName string) *corev1.Pod {
	return &corev1.Pod{
		Name:      name,
		Namespace: crashLoopTestNamespace,
		Labels:    map[string]string{commonconsts.KubeLabelDynamoSelector: dcdName},
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}},
			ContainerStatuses: []corev1.ContainerStatus{{
				Name:                 commonconsts.MainContainerName,
				RestartCount:         3,
				State:                corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"}},
				LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1, Reason: "Error"}},
			}},
		},
	}
}
