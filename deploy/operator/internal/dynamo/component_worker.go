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
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          commonconsts.DynamoNixlPortName,
			ContainerPort: int32(commonconsts.DynamoNixlPort),
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

	// ReadinessProbe gates external (Kubernetes Service / EndpointSlice) routing,
	// including the Frontend's KubeDiscoveryClient path in
	// lib/runtime/src/discovery/kube/daemon.rs:246, which filters EndpointSlices
	// by endpoint.conditions.ready==true before correlating with DynamoWorkerMetadata
	// CRs. Without a passing ReadinessProbe on the worker, the Service endpoint
	// stays not-ready and Frontend's KubeDiscoveryClient returns 0 instances for
	// AllEndpoints / AllModels even though the worker has registered its CR and
	// is serving traffic over the internal NATS/etcd KvStore transport.
	//
	// Prior wording ("doesn't determine that the worker is ready to receive traffic")
	// was only correct for Dynamo's internal KvStore-based routing. The clarifying
	// comment below ("Still important for external dependencies") was ambiguous
	// and caused multiple downstream deployments to omit the probe and see
	// KubeDiscoveryClient returning 0 instances.
	//
	// See: https://github.com/ai-dynamo/dynamo/issues/9200 for the reproducing case.
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
		{
			Name:  "NIXL_TELEMETRY_ENABLE",
			Value: "n",
		},
		{
			Name:  "NIXL_TELEMETRY_EXPORTER",
			Value: "prometheus",
		},
		{
			Name:  "NIXL_TELEMETRY_PROMETHEUS_PORT",
			Value: fmt.Sprintf("%d", commonconsts.DynamoNixlPort),
		},
		{
			Name:  "DYN_FORWARDPASS_METRIC_PORT",
			Value: fmt.Sprintf("%d", commonconsts.DynamoFPMBasePort),
		},
	}...)

	if context.WorkerHashSuffix != "" {
		container.Env = append(container.Env, []corev1.EnvVar{
			{
				Name:  commonconsts.DynamoNamespaceWorkerSuffixEnvVar,
				Value: context.WorkerHashSuffix,
			},
		}...)
	}

	return container, nil
}
