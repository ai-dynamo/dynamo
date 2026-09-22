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
		// Rust EPP's default image (frontend) uses /home/dynamo while the
		// dedicated dynamo-epp image runs as nonroot, so without this the mount
		// below can only be right for one of them.
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "HOME",
			Value: nativeRustEPPHome,
		})

		// Mount the model-config cache under the pinned HOME. The Rust EPP
		// resolves its MDC cache root from $HOME (lib/llm/src/model_card.rs), so
		// a mount that does not match is silently inert and the blobs grow on
		// the container's writable layer instead.
		//
		// Native Rust EPP only. Recipes render the native path on
		// dynamo-frontend, but the image is the user's to set, and the images an
		// EPP component can carry disagree on HOME:
		//
		//   contract  image                     image HOME      mounted
		//   native    dynamo-frontend:1.5.0+    /home/dynamo    yes, HOME pinned
		//   native    dynamo-epp:1.5.0          /home/nonroot   yes, HOME pinned
		//   legacy    epp-image:1.4.x           /home/nonroot   no
		//   legacy    dynamo-frontend:1.4.x     /home/dynamo    no
		//
		// A DYN_EPP_MODE=standalone EPP is absent from that list on both counts:
		// it is applied as hand-written YAML rather than rendered here, and it
		// needs no cache anyway, since runner.rs takes the
		// EppRouter::from_selector branch and never reaches download_config.
		//
		// The two native images disagree just as the legacy pair does, but the
		// operator sets HOME itself above, so both converge on /home/dynamo and
		// one mount path is right for both. The legacy rows get no such pin, and
		// nothing available here picks between them: eppConfig names the launch
		// contract, not the image, and the resolved runtime version only bounds
		// it to "below 1.5.0", which both legacy images satisfy. A mount at the
		// wrong HOME is worse than none -- nothing errors, the blobs go to the
		// writable layer anyway, and the volume merely looks correct.
		//
		// Going without costs little there in any case: the legacy Go EPP does
		// download the same files, calling the same download_config through
		// libdynamo_llm_capi, but only once at startup over a fixed file set --
		// so the loss is a few MB and a re-fetch on restart, on a component
		// being removed.
		//
		// Withholding it re-renders existing legacy EPP Pods, so they roll once
		// on an operator upgrade. That is a deliberate exception to the no-roll
		// rule in deploy/operator/internal/AGENTS.md, taken on deprecation
		// grounds; the promised legacy contract (CLI flags, config volume,
		// Service selector) is untouched.
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
		// Backs the model-config cache mounted in GetBaseContainer, which is
		// native Rust EPP only; see there for why the legacy path goes without.
		podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
			Name: "hf-cache",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		})
	}

	return podSpec, nil
}
