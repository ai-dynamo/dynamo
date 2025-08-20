/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
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
	return container, nil
}

func (p *PlannerDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec := p.getCommonPodSpec()
	podSpec.ServiceAccountName = commonconsts.PlannerServiceAccountName
	return podSpec, nil
}
