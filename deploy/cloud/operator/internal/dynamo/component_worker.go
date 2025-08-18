/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
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

	// Add worker base defaults
	container.Resources = corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("10"),
			corev1.ResourceMemory: resource.MustParse("20Gi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("10"),
			corev1.ResourceMemory: resource.MustParse("20Gi"),
			"nvidia.com/gpu":      resource.MustParse("1"),
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
		TimeoutSeconds:   30,
		FailureThreshold: 1,
	}

	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/health",
				Port: intstr.FromString(commonconsts.DynamoSystemPortName),
			},
		},
		PeriodSeconds:    10,
		TimeoutSeconds:   30,
		FailureThreshold: 60,
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
		FailureThreshold: 60,
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
	}...)

	return container, nil
}
