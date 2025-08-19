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

// FrontendDefaults implements ComponentDefaults for Frontend components
type FrontendDefaults struct {
	*BaseComponentDefaults
}

func NewFrontendDefaults() *FrontendDefaults {
	return &FrontendDefaults{&BaseComponentDefaults{}}
}

func (f *FrontendDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	// Frontend doesn't need backend-specific config
	container := f.getCommonContainer(context)

	// Set default command and args
	container.Command = []string{"python3"}
	container.Args = []string{"-m", "dynamo.frontend"}

	// Add HTTP port
	container.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          commonconsts.DynamoContainerPortName,
			ContainerPort: int32(commonconsts.DynamoServicePort),
		},
	}

	// Add frontend-specific defaults
	container.LivenessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/health",
				Port: intstr.FromString(commonconsts.DynamoContainerPortName),
			},
		},
		InitialDelaySeconds: 60,
		PeriodSeconds:       60,
		TimeoutSeconds:      30,
		FailureThreshold:    10,
	}

	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{
					"/bin/sh",
					"-c",
					"curl -s http://localhost:${DYNAMO_PORT}/health | jq -e \".status == \\\"healthy\\\"\"",
				},
			},
		},
		InitialDelaySeconds: 60,
		PeriodSeconds:       60,
		TimeoutSeconds:      30,
		FailureThreshold:    10,
	}

	// Add standard environment variables
	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  commonconsts.EnvDynamoServicePort,
			Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
		},
		{
			Name:  "DYN_HTTP_PORT", // TODO: need to reconcile DYNAMO_PORT and DYN_HTTP_PORT
			Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
		},
	}...)

	return container, nil
}
