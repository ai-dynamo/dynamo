/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

const (
	lpuModelConfigFilepath              = lpuConfigMountPath + "/model_config.toml"
	lpuExpandedDatacenterConfigFilepath = runtimeTemporaryStorageMountPath + "/datacenter.toml"
)

func updateLPUConductorContainer(
	container *corev1.Container,
	allocation string,
) {
	// Expand this engine's topology before the default or explicitly authored command.
	container.Name = "conductor"
	wrapRuntimeStartup(container, "/bin/nova", "datacenter.toml", "")

	// Nova and OpenMPI require root in the supported direct-DGD runtime.
	if container.SecurityContext == nil {
		container.SecurityContext = &corev1.SecurityContext{}
	}
	container.SecurityContext.RunAsUser = ptr.To(int64(0))
	container.SecurityContext.RunAsGroup = ptr.To(int64(0))
	container.SecurityContext.RunAsNonRoot = ptr.To(false)

	// Append the complete operator-owned Nova launch contract after user arguments.
	container.Args = append(container.Args,
		"--agent-path",
		"/bin/agent",
		"--agent-wd",
		runtimeTemporaryStorageMountPath,
		"--conductor-bind-host",
		"0.0.0.0",
		"--conductor-hostname",
		"$(POD_IP)",
		"--control-server-addr",
		"$(POD_IP):51045",

		// configmap
		"--datacenter-config-filepath",
		lpuExpandedDatacenterConfigFilepath,
		"--model-config-filepath",
		lpuModelConfigFilepath,

		// logs
		"--json-logs",
		"--disable-tracing",

		// mpirun
		"--mpirun-path",
		"/bin/mpirun",
		"--mpi-ssh-cmd",
		fmt.Sprintf(
			"ssh -p %d -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i %s",
			LPUSSHPort,
			conductorSSHPrivateKeyPath,
		),
		"--mpi-use-tcp",

		// alloc
		"--allocation", allocation,

		// conductor/agent lifecycle
		"--agent-disable-auto-restart",
		"--agent-server-max-retries",
		"60",
		"--agent-server-retry-sleep-dur",
		"2s",
		"--controlplane-start-port",
		"12345",
		"--graceful-shutdown-timeout",
		"0s",

		// misc
		"--response-timeout",
		"120s",

		// dyn
		"--frontend-type",
		"dynamo",
		"--endpoint-id",
		"dyn://$(DYN_NAMESPACE).backend.generate",
		"--request-plane",
		"tcp",
		"--store-backend",
		"kubernetes",
	)

	// Allow root OpenMPI to reach the SSH workers with the copied private key.
	container.Env = append(container.Env,
		corev1.EnvVar{Name: "OMPI_ALLOW_RUN_AS_ROOT", Value: "1"},
		corev1.EnvVar{Name: "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", Value: "1"},
		corev1.EnvVar{
			Name: "OMPI_MCA_plm_rsh_args",
			Value: fmt.Sprintf(
				"-p %d -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentityFile=%s",
				LPUSSHPort,
				conductorSSHPrivateKeyPath,
			),
		},
	)
	container.VolumeMounts = setVolumeMount(container.VolumeMounts, conductorSSHKeyVolumeMount())

	// Default readiness to Nova's health endpoint while preserving an explicit probe.
	if container.ReadinessProbe == nil {
		container.ReadinessProbe = &corev1.Probe{
			FailureThreshold:    3,
			InitialDelaySeconds: 10,
			PeriodSeconds:       5,
			SuccessThreshold:    1,
			TimeoutSeconds:      1,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path:   "/readyz",
					Port:   intstr.FromInt(51045),
					Scheme: corev1.URISchemeHTTP,
				},
			},
		}
	}
}
