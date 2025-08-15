/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
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

	// Add planner-specific defaults
	container.Resources = corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("2"),
			corev1.ResourceMemory: resource.MustParse("2Gi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("2"),
			corev1.ResourceMemory: resource.MustParse("2Gi"),
		},
	}

	return container, nil
}

func (p *PlannerDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec := corev1.PodSpec{
		ServiceAccountName: commonconsts.PlannerServiceAccountName,
	}
	return podSpec, nil
}
