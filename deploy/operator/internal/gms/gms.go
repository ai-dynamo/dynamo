/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package gms provides GMS (GPU Memory Service) server container building
// for both steady-state DGD pods and checkpoint/restore flows.
package gms

import (
	"path/filepath"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const (
	// ServerContainerName is the name of the GMS server init sidecar.
	ServerContainerName = "gms-server"

	// SharedVolumeName is the emptyDir volume shared between the GMS server
	// sidecar and the main workload container for UDS sockets. The name
	// disambiguates it from the snapshot-control volume, which carries
	// checkpoint/restore lifecycle sentinels written by the snapshot agent.
	SharedVolumeName = "gms-intrapod-control"

	// SharedMountPath is the mount path for the GMS intra-pod IPC directory.
	SharedMountPath = "/gms-intrapod-control"

	// EnvSocketDir is the environment variable name for the GMS UDS socket directory.
	EnvSocketDir = "GMS_SOCKET_DIR"

	// ServerModule is the Python module for the GMS server entry point.
	ServerModule = "gpu_memory_service.cli.server"

	readyFile = "gms-ready"
)

// EnsureServerSidecar adds the GMS server as a restartable init sidecar with a
// startup probe. Idempotent — safe to call from both the DGD and checkpoint paths.
func EnsureServerSidecar(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	EnsureSharedVolume(podSpec, mainContainer)

	sidecar := Container(ServerContainerName, ServerModule, mainContainer.Image)
	sidecar.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
	sidecar.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{"test", "-f", filepath.Join(SharedMountPath, readyFile)},
			},
		},
		PeriodSeconds:    1,
		FailureThreshold: 300, // 1s * 300 = 5 min
	}
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == sidecar.Name {
			return
		}
	}
	podSpec.InitContainers = append(podSpec.InitContainers, sidecar)
}

// EnsureSharedVolume adds the GMS UDS socket volume, mount, and GMS_SOCKET_DIR
// env var to the main container. Idempotent.
func EnsureSharedVolume(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
	hasVolume := false
	for _, v := range podSpec.Volumes {
		if v.Name == SharedVolumeName {
			hasVolume = true
			break
		}
	}
	if !hasVolume {
		podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
			Name:         SharedVolumeName,
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		})
	}

	hasMount := false
	for _, m := range mainContainer.VolumeMounts {
		if m.Name == SharedVolumeName {
			hasMount = true
			break
		}
	}
	if !hasMount {
		mainContainer.VolumeMounts = append(mainContainer.VolumeMounts, corev1.VolumeMount{Name: SharedVolumeName, MountPath: SharedMountPath})
	}

	hasEnv := false
	for _, e := range mainContainer.Env {
		if e.Name == EnvSocketDir {
			hasEnv = true
			break
		}
	}
	if !hasEnv {
		mainContainer.Env = append(mainContainer.Env, corev1.EnvVar{Name: EnvSocketDir, Value: SharedMountPath})
	}
}

// Container builds a GMS container with the shared socket volume, env, and
// DRA claim. Used for the server, loader, and saver.
func Container(name, module, image string) corev1.Container {
	return corev1.Container{
		Name:    name,
		Image:   image,
		Command: []string{"python3", "-m", module},
		Env: []corev1.EnvVar{
			{Name: EnvSocketDir, Value: SharedMountPath},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: SharedVolumeName, MountPath: SharedMountPath},
		},
		Resources: corev1.ResourceRequirements{
			Claims: []corev1.ResourceClaim{{Name: dra.ClaimName}},
		},
	}
}
