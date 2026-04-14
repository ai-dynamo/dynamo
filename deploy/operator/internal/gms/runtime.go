/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package gms

import (
	"path/filepath"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const (
	// ServerContainerName is the name of the GMS server init sidecar.
	ServerContainerName = "gms-server"

	// SharedVolumeName is the emptyDir volume shared between the GMS server
	// sidecar and the main workload container for UDS sockets.
	SharedVolumeName = "gms-shared"

	// SharedMountPath is the mount path for the shared GMS socket directory.
	SharedMountPath = "/shared"

	// DRAClaimName is the pod-level DRA ResourceClaim name used by both the
	// main container and GMS sidecars.
	DRAClaimName = "shared-gpu"

	// ControlVolumeName is the checkpoint-specific control volume name.
	ControlVolumeName = "gms-control"

	// ControlDir is the mount path for the checkpoint control volume.
	ControlDir = "/tmp/gms-control"

	readyFile = "gms-ready"

	serverSidecarModule = "gpu_memory_service.cli.gms_server_sidecar"
)

// EnsureServerSidecar adds the GMS server as a restartable init sidecar with a
// startup probe. Used for checkpoint jobs and steady-state pods where the main
// container needs GMS sockets before starting.
func EnsureServerSidecar(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	ensureSharedVolume(podSpec, mainContainer)

	sidecar := serverContainer(mainContainer.Image)
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
	copyDeviceClaims(mainContainer, &sidecar)
	ensureInitContainer(podSpec, sidecar)
}

// BuildServerContainer prepares the shared GMS volume/env and returns a GMS
// server container suitable for use as a regular sidecar. The caller must
// append the returned container to podSpec.Containers.
//
// Used for restore pods where the main container is CRIU-restored and does not
// need GMS sockets at startup. The gms-loader polls for sockets internally.
func BuildServerContainer(podSpec *corev1.PodSpec, mainContainer *corev1.Container) corev1.Container {
	ensureSharedVolume(podSpec, mainContainer)
	sidecar := serverContainer(mainContainer.Image)
	copyDeviceClaims(mainContainer, &sidecar)
	return sidecar
}

// FindServerContainer returns a pointer to the GMS server container, checking
// both init containers and regular containers. Returns nil if not present.
func FindServerContainer(podSpec *corev1.PodSpec) *corev1.Container {
	if podSpec == nil {
		return nil
	}
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == ServerContainerName {
			return &podSpec.InitContainers[i]
		}
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == ServerContainerName {
			return &podSpec.Containers[i]
		}
	}
	return nil
}

// ensureSharedVolume adds the shared GMS socket volume and env vars to the pod.
func ensureSharedVolume(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
	ensureVolume(podSpec, corev1.Volume{
		Name:         SharedVolumeName,
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	})
	ensureVolumeMount(mainContainer, corev1.VolumeMount{Name: SharedVolumeName, MountPath: SharedMountPath})
	setEnv(mainContainer, "TMPDIR", SharedMountPath)
	setEnv(mainContainer, "GMS_SOCKET_DIR", SharedMountPath)
}

// serverContainer builds the base GMS server container without init-specific
// fields (RestartPolicy, StartupProbe). Callers add those as needed.
func serverContainer(image string) corev1.Container {
	return corev1.Container{
		Name:    ServerContainerName,
		Image:   image,
		Command: []string{"python3", "-m", serverSidecarModule},
		Env: []corev1.EnvVar{
			{Name: "TMPDIR", Value: SharedMountPath},
			{Name: "GMS_SOCKET_DIR", Value: SharedMountPath},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: SharedVolumeName, MountPath: SharedMountPath},
		},
	}
}

func copyDeviceClaims(src *corev1.Container, dst *corev1.Container) {
	if src == nil || dst == nil || len(src.Resources.Claims) == 0 {
		return
	}
	dst.Resources.Claims = append(dst.Resources.Claims, src.Resources.Claims...)
}

func ensureInitContainer(podSpec *corev1.PodSpec, container corev1.Container) {
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == container.Name {
			return
		}
	}
	podSpec.InitContainers = append(podSpec.InitContainers, container)
}

func ensureVolume(podSpec *corev1.PodSpec, volume corev1.Volume) {
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == volume.Name {
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, volume)
}

func ensureVolumeMount(container *corev1.Container, mount corev1.VolumeMount) {
	for i := range container.VolumeMounts {
		if container.VolumeMounts[i].Name == mount.Name && container.VolumeMounts[i].MountPath == mount.MountPath {
			return
		}
	}
	container.VolumeMounts = append(container.VolumeMounts, mount)
}

func setEnv(container *corev1.Container, name string, value string) {
	for i := range container.Env {
		if container.Env[i].Name != name {
			continue
		}
		container.Env[i].Value = value
		container.Env[i].ValueFrom = nil
		return
	}
	container.Env = append(container.Env, corev1.EnvVar{Name: name, Value: value})
}
