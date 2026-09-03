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

const (
	// HOME in the frontend image, where the native Rust EPP ships: that image
	// does `useradd -m ... dynamo` + `ENV HOME=/home/dynamo` + `USER dynamo`.
	nativeRustEPPHome = "/home/dynamo"
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
			Value: "info",
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

	container.Command = []string{}

	// Presence of eppConfig keeps the legacy Go EPP launch contract so existing
	// DGDs survive operator upgrades unchanged until migration clears it.
	if epp.IsLegacyGoEPP(context.EPPConfig) {
		poolName := epp.GetPoolName(context.ParentGraphDeploymentName)
		poolNamespace := epp.GetPoolNamespace(context.ParentGraphDeploymentNamespace)
		configFilePath := epp.GetConfigFilePath()

		container.Args = []string{
			"--pool-name", poolName,
			"--pool-namespace", poolNamespace,
			"--pool-group", epp.InferencePoolGroup,
			"-v", "4",
			"--zap-encoder", "json",
			"--grpc-port", fmt.Sprintf("%d", commonconsts.EPPGRPCPort),
			"--grpc-health-port", "9003",
			"--config-file", configFilePath,
		}

		_, volumeMount := epp.GetConfigMapVolumeMount(context.ParentGraphDeploymentName, context.EPPConfig)
		container.VolumeMounts = append(container.VolumeMounts, volumeMount)
	} else {
		// Native Rust EPP: configured through DYN_* env vars, serves
		// ext_proc/health on fixed ports, takes no CLI flags, and reads no
		// config file. Leave Args empty and let the image ENTRYPOINT run.
		// Users can still override args via extraPodSpec.mainContainer.args.
		container.Args = []string{}

		// Pin HOME rather than inheriting whatever the image sets. The native
		// EPP's default image (frontend) uses /home/dynamo while the standalone
		// dynamo-epp image runs as nonroot, so without this the mount below can
		// only be right for one of them.
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "HOME",
			Value: nativeRustEPPHome,
		})

		// Mount the model-config cache the EPP writes while downloading model
		// configs. The path must track the container's HOME: the Rust EPP
		// resolves its MDC cache root from $HOME (lib/llm/src/model_card.rs), so
		// a mount that does not match is silently inert -- no error, but the
		// blobs land on the container's ephemeral writable layer, which grows
		// unbounded toward node DiskPressure and fails outright under
		// readOnlyRootFilesystem: true.
		//
		// Only the native EPP gets this mount. The legacy Go EPP downloads no
		// model configs, and its HOME cannot be derived from the launch contract
		// alone: eppConfig selects the contract but not the image, so a legacy
		// component on the frontend image (HOME=/home/dynamo) would take the
		// nonroot path and mount the cache somewhere nothing ever reads.
		//
		// Withholding it from the legacy path re-renders every existing legacy
		// EPP Pod, so those components roll once on an operator upgrade even
		// though their DGD did not change. That is a deliberate exception to the
		// rule that an operator-only upgrade must not roll unchanged workloads
		// (deploy/operator/internal/AGENTS.md): the volume being withdrawn was
		// never read on this path, and the Go EPP it was shaped for is
		// deprecated, with its source removed in this release. The enumerated
		// legacy Pod contract that upgrade compatibility actually promises --
		// CLI flags, config volume, Service selector -- is untouched.
		//
		// A HOME supplied through podTemplate still wins over the one set above
		// (MergeEnvs gives user env precedence), which would re-break the
		// pairing; that is the caller's choice, the same as overriding args.
		container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
			Name:      "hf-cache",
			MountPath: nativeRustEPPHome + "/.cache",
		})
	}

	return container, nil
}

func (e *EPPDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec := e.getCommonPodSpec()

	// EPP uses global service account (like planner)
	podSpec.ServiceAccountName = commonconsts.EPPServiceAccountName

	// EPP needs longer grace period for graceful shutdown
	podSpec.TerminationGracePeriodSeconds = ptr.To(int64(130))

	if epp.IsLegacyGoEPP(context.EPPConfig) {
		volume, _ := epp.GetConfigMapVolumeMount(context.ParentGraphDeploymentName, context.EPPConfig)
		podSpec.Volumes = append(podSpec.Volumes, volume)
	} else {
		// Backs the model-config cache mounted in GetBaseContainer. Native EPP
		// only: the legacy Go EPP downloads no model configs, so the volume
		// would be dead weight and its mount path unknowable from the contract.
		// See GetBaseContainer for why withholding it from the legacy path is a
		// deliberate exception to the no-roll-on-operator-upgrade rule.
		podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
			Name: "hf-cache",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		})
	}

	return podSpec, nil
}
