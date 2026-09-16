/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	// LPUSSHPort is the SSH port used by Nova and LPU Agent workers.
	LPUSSHPort = commonconsts.MpiRunSshPort
	// LPUReadinessPort is the HTTP readiness port used by the LPU runtime.
	LPUReadinessPort = 19878
	lpuRDMAPort      = 19877
)

const (
	lpuV2HTTPStartupFailureThreshold int32 = 60
	lpuV2ExecStartupFailureThreshold int32 = 240
)

func lpuV2StartupProbe(useHTTP bool) *corev1.Probe {
	probe := lpuV2ReadinessProbe(useHTTP)
	if useHTTP {
		probe.FailureThreshold = lpuV2HTTPStartupFailureThreshold
	} else {
		probe.FailureThreshold = lpuV2ExecStartupFailureThreshold
	}
	return probe
}

func lpuV2ReadinessProbe(useHTTP bool) *corev1.Probe {
	probe := &corev1.Probe{
		FailureThreshold: 6,
		PeriodSeconds:    5,
		SuccessThreshold: 1,
		TimeoutSeconds:   1,
	}
	if useHTTP {
		probe.HTTPGet = &corev1.HTTPGetAction{
			Path:   "/readyz",
			Port:   intstr.FromInt(LPUReadinessPort),
			Scheme: corev1.URISchemeHTTP,
		}
	} else {
		probe.Exec = &corev1.ExecAction{
			Command: []string{"/bin/hydra-entrypoint", "ready"},
		}
	}
	return probe
}

func lpuSSHProbe() *corev1.Probe {
	return &corev1.Probe{
		FailureThreshold:    3,
		InitialDelaySeconds: 10,
		PeriodSeconds:       5,
		SuccessThreshold:    1,
		TimeoutSeconds:      1,
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: &corev1.TCPSocketAction{
				Port: intstr.FromInt(LPUSSHPort),
			},
		},
	}
}
