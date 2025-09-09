/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
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
	}
	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  "PROMETHEUS_PORT",
			Value: fmt.Sprintf("%d", commonconsts.DynamoPlannerMetricsPort),
		},
	}...)
	return container, nil
}

func (p *PlannerDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec := p.getCommonPodSpec()
	podSpec.ServiceAccountName = commonconsts.PlannerServiceAccountName
	return podSpec, nil
}
