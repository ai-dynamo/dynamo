/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// PlannerDefaults implements ComponentDefaults for Planner components
type PlannerDefaults struct {
	*BaseComponentDefaults
}

func NewPlannerDefaults() *PlannerDefaults {
	return &PlannerDefaults{&BaseComponentDefaults{}}
}

func (p *PlannerDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	container := p.getCommonContainer(context)
	container.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          commonconsts.DynamoMetricsPortName,
			ContainerPort: int32(commonconsts.DynamoPlannerMetricsPort),
		},
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
		TimeoutSeconds:   4,
		FailureThreshold: 1,
	}

	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/health",
				Port: intstr.FromString(commonconsts.DynamoSystemPortName),
			},
		},
		PeriodSeconds:    1,
		TimeoutSeconds:   4,
		FailureThreshold: 3,
	}

	// Startup probe with generous timeout: the planner waits for worker
	// services to become ready before it can initialise, so it needs more
	// time than a typical worker.
	container.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/live",
				Port: intstr.FromString(commonconsts.DynamoSystemPortName),
			},
		},
		PeriodSeconds:    1,
		TimeoutSeconds:   5,
		FailureThreshold: 7200, // 1s * 7200 = 7200s = 2h
	}

	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  "PLANNER_PROMETHEUS_PORT",
			Value: fmt.Sprintf("%d", commonconsts.DynamoPlannerMetricsPort),
		},
		{
			Name:  "DYN_SYSTEM_PORT",
			Value: fmt.Sprintf("%d", commonconsts.DynamoSystemPort),
		},
	}...)
	return container, nil
}

func (p *PlannerDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec := p.getCommonPodSpec()
	podSpec.ServiceAccountName = commonconsts.PlannerServiceAccountName
	return podSpec, nil
}
