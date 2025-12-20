/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// WorkerDefaults implements ComponentDefaults for Worker components
type WorkerDefaults struct {
	*BaseComponentDefaults
}

func NewWorkerDefaults() *WorkerDefaults {
	return &WorkerDefaults{&BaseComponentDefaults{}}
}

func (w *WorkerDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	container := w.getCommonContainer(context)

	// Add system port
	container.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          commonconsts.DynamoSystemPortName,
			ContainerPort: int32(commonconsts.DynamoSystemPort),
		},
	}

	container.LivenessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/live",
				Port: intstr.FromString(commonconsts.DynamoSystemPortName),
			},
		},
		PeriodSeconds:    5,
		TimeoutSeconds:   4, // TimeoutSeconds should be < PeriodSeconds
		FailureThreshold: 1, // Note this default FailureThreshold is 3, with 1 a single failure will restart Pod
	}

	// ReadinessProbe in Dynamo worker context doesn't determine that the worker is ready to receive traffic
	// Since worker registration is done through external KvStore and Transport does not use Kubernetes Service
	// Still important for external depencies that rely on Pod Readiness
	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/health",
				Port: intstr.FromString(commonconsts.DynamoSystemPortName),
			},
		},
		PeriodSeconds:    10,
		TimeoutSeconds:   4,
		FailureThreshold: 3,
	}

	container.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/live",
				Port: intstr.FromString(commonconsts.DynamoSystemPortName),
			},
		},
		PeriodSeconds:    10,
		TimeoutSeconds:   5,
		FailureThreshold: 720, // 10s * 720 = 7200s = 2h
	}

	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  "DYN_SYSTEM_ENABLED",
			Value: "true",
		},
		{
			Name:  "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
			Value: "[\"generate\"]",
		},
		{
			Name:  "DYN_SYSTEM_PORT",
			Value: fmt.Sprintf("%d", commonconsts.DynamoSystemPort),
		},
		{
			Name:  "DYN_HEALTH_CHECK_ENABLED",
			Value: "false",
		},
	}...)

	return container, nil
}
