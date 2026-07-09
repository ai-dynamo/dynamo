/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// EPPDefaults implements ComponentDefaults for EPP (Endpoint Picker Plugin) components
type EPPDefaults struct {
	*BaseComponentDefaults
}

func NewEPPDefaults() *EPPDefaults {
	return &EPPDefaults{&BaseComponentDefaults{}}
}

func (e *EPPDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	container := e.getCommonContainer(context)

	// EPP uses gRPC, so we need gRPC probes (not HTTP)
	// Port 9002: gRPC endpoint for InferencePool communication
	// Port 9003: gRPC health check endpoint
	// Port 9090: Metrics endpoint
	container.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          commonconsts.EPPGRPCPortName,
			ContainerPort: commonconsts.EPPGRPCPort,
		},
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          "grpc-health",
			ContainerPort: 9003,
		},
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          commonconsts.DynamoMetricsPortName,
			ContainerPort: 9090,
		},
	}

	// gRPC-based probes
	container.LivenessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			GRPC: &corev1.GRPCAction{
				Port:    9003,
				Service: ptr.To("inference-extension"),
			},
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       10,
	}

	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			GRPC: &corev1.GRPCAction{
				Port:    9003,
				Service: ptr.To("inference-extension"),
			},
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       10,
	}

	// Startup probe allows long initialization while waiting for workers to register.
	// EPP waits indefinitely for discovery to find workers, so this probe is the
	// only timeout mechanism. Default: 30 minutes (10s × 180 = 1800s).
	container.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			GRPC: &corev1.GRPCAction{
				Port:    9003,
				Service: ptr.To("inference-extension"),
			},
		},
		PeriodSeconds:    10,
		FailureThreshold: 180,
	}

	// EPP-specific environment variables
	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  "USE_STREAMING",
			Value: "true",
		},
		{
			Name:  "RUST_LOG",
			Value: "debug,dynamo_llm::kv_router=trace",
		},
		{
			Name:  "DYN_ENFORCE_DISAGG",
			Value: "false",
		},
		{
			Name:  commonconsts.DynamoNamespacePrefixEnvVar,
			Value: context.DynamoNamespace,
		},
	}...)

	// The Dynamo EPP is a native binary configured entirely through
	// environment variables (DYN_NAMESPACE_PREFIX, DYN_COMPONENT_NAME, ...)
	// and serves ext_proc/health on fixed ports. It takes no CLI flags and
	// reads no config file, so leave Command/Args empty and let the image
	// ENTRYPOINT run. Users can still override args via
	// extraPodSpec.mainContainer.args (mergo.WithOverride).
	container.Command = []string{}
	container.Args = []string{}

	// Mount HuggingFace cache directory for model config downloads
	hfCacheMount := corev1.VolumeMount{
		Name:      "hf-cache",
		MountPath: "/home/nonroot/.cache",
	}
	container.VolumeMounts = append(container.VolumeMounts, hfCacheMount)

	return container, nil
}

func (e *EPPDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec := e.getCommonPodSpec()

	// EPP uses global service account (like planner)
	podSpec.ServiceAccountName = commonconsts.EPPServiceAccountName

	// EPP needs longer grace period for graceful shutdown
	podSpec.TerminationGracePeriodSeconds = ptr.To(int64(130))

	// Add emptyDir volume for HuggingFace cache (needed for downloading model config files)
	hfCacheVolume := corev1.Volume{
		Name: "hf-cache",
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}
	podSpec.Volumes = append(podSpec.Volumes, hfCacheVolume)

	return podSpec, nil
}
