/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package trtllm declares the Kubernetes and launch-command mutations used by
// the TensorRT-LLM backend.
package trtllm

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// MultinodeWiring returns the operator-owned container wiring shared by the
// TensorRT-LLM leader and workers.
func MultinodeWiring(secretName string) mutation.EngineMutations {
	return mutation.EngineMutations{
		{
			Applies: mutation.WorkerRole,
			Mutation: mutation.RemoveProbesMutation{
				ContainerName: commonconsts.MainContainerName,
				Liveness:      true,
				Startup:       true,
			},
		},
		{
			Applies: mutation.WorkerRole,
			Mutation: mutation.SetReadinessProbeMutation{
				ContainerName: commonconsts.MainContainerName,
				Probe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						TCPSocket: &corev1.TCPSocketAction{
							Port: intstr.FromInt(commonconsts.MpiRunSshPort),
						},
					},
					InitialDelaySeconds: 20,
					PeriodSeconds:       20,
					TimeoutSeconds:      5,
					FailureThreshold:    10,
				},
			},
		},
		{
			Mutation: SSHVolumeMount(secretName),
		},
		{
			Mutation: mutation.AppendEnvMutation{
				ContainerName: commonconsts.MainContainerName,
				Env:           MPIEnvironment(),
			},
		},
	}
}

// MPIEnvironment returns the environment mutation value that must also be
// visible while resolving the leader's exported mpirun environment.
func MPIEnvironment() corev1.EnvVar {
	return corev1.EnvVar{Name: "OMPI_MCA_orte_keep_fqdn_hostnames", Value: "1"}
}

// LeaderLaunch returns the resolved TensorRT-LLM leader launch mutation.
func LeaderLaunch(command, args []string) mutation.EngineMutations {
	return mutation.EngineMutations{{
		Applies: mutation.LeaderRole,
		Mutation: mutation.SetCommandMutation{
			ContainerName: commonconsts.MainContainerName,
			Command:       command,
			Args:          args,
		},
	}}
}

// WorkerLaunch returns the resolved TensorRT-LLM worker launch mutation.
func WorkerLaunch(command, args []string) mutation.EngineMutations {
	return mutation.EngineMutations{{
		Applies: mutation.WorkerRole,
		Mutation: mutation.SetCommandMutation{
			ContainerName: commonconsts.MainContainerName,
			Command:       command,
			Args:          args,
		},
	}}
}

// SSHVolumeMount returns the standard TensorRT-LLM SSH key volume mount.
func SSHVolumeMount(secretName string) mutation.AppendVolumeMountMutation {
	return mutation.AppendVolumeMountMutation{
		ContainerName: commonconsts.MainContainerName,
		VolumeMount: corev1.VolumeMount{
			Name: secretName, MountPath: "/ssh-pk", ReadOnly: true,
		},
	}
}

// PodSpec returns the TensorRT-LLM SSH secret volume mutation.
func PodSpec(secretName string) mutation.PodSpecMutations {
	mode := int32(0644)
	return mutation.PodSpecMutations{{
		Mutation: mutation.AppendVolumeMutation{
			Volume: corev1.Volume{
				Name: secretName,
				VolumeSource: corev1.VolumeSource{
					Secret: &corev1.SecretVolumeSource{
						SecretName:  secretName,
						DefaultMode: &mode,
					},
				},
			},
		},
	}}
}
