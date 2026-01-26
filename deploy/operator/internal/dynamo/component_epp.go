/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
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

	// EPP-specific environment variables
	// Note: Platform-specific env vars (NATS_SERVER, ETCD_ENDPOINTS) are added by the controller
	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  "DYNAMO_KV_BLOCK_SIZE",
			Value: "16",
		},
		{
			Name:  "USE_STREAMING",
			Value: "true",
		},
		{
			Name:  "RUST_LOG",
			Value: "debug,dynamo_llm::kv_router=trace",
		},
	}...)

	// EPP default args
	// These can be overridden via extraPodSpec.mainContainer.args (mergo.WithOverride)
	poolName := epp.GetPoolName(context.ParentGraphDeploymentName, context.EPPConfig)
	poolNamespace := epp.GetPoolNamespace(context.ParentGraphDeploymentNamespace, context.EPPConfig)
	configFilePath := epp.GetConfigFilePath()

	container.Command = []string{}

	container.Args = []string{
		"-pool-name", poolName,
		"-pool-namespace", poolNamespace,
		"-pool-group", epp.InferencePoolGroup,
		"-v", "4",
		"--zap-encoder", "json",
		"-grpc-port", fmt.Sprintf("%d", commonconsts.EPPGRPCPort),
		"-grpc-health-port", "9003",
		"-config-file", configFilePath,
	}

	// Mount EPP config
	_, volumeMount := epp.GetConfigMapVolumeMount(context.ParentGraphDeploymentName, context.EPPConfig)
	container.VolumeMounts = append(container.VolumeMounts, volumeMount)

	return container, nil
}

func (e *EPPDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec := e.getCommonPodSpec()

	// EPP uses global service account (like planner)
	podSpec.ServiceAccountName = commonconsts.EPPServiceAccountName

	// EPP needs longer grace period for graceful shutdown
	podSpec.TerminationGracePeriodSeconds = ptr.To(int64(130))

	// Add EPP config volume
	volume, _ := epp.GetConfigMapVolumeMount(context.ParentGraphDeploymentName, context.EPPConfig)
	podSpec.Volumes = append(podSpec.Volumes, volume)

	return podSpec, nil
}
