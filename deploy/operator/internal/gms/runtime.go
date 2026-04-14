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

	serverSidecarModule = "gpu_memory_service.cli.server"
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
	// Idempotent — EnsureServerSidecar may be called by both the
	// steady-state operator path and the checkpoint overlay.
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == sidecar.Name {
			return
		}
	}
	podSpec.InitContainers = append(podSpec.InitContainers, sidecar)
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

// ensureSharedVolume adds the shared GMS socket volume, mounts, and env vars.
// Idempotent — may be called by both steady-state and checkpoint paths.
func ensureSharedVolume(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
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

	// Mount and env injection checked independently of volume existence —
	// another code path may have added the volume without configuring main.
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
		if e.Name == "GMS_SOCKET_DIR" {
			hasEnv = true
			break
		}
	}
	if !hasEnv {
		mainContainer.Env = append(mainContainer.Env,
			corev1.EnvVar{Name: "TMPDIR", Value: SharedMountPath},
			corev1.EnvVar{Name: "GMS_SOCKET_DIR", Value: SharedMountPath},
		)
	}
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
	claims := make([]corev1.ResourceClaim, len(src.Resources.Claims))
	copy(claims, src.Resources.Claims)
	dst.Resources.Claims = append(dst.Resources.Claims, claims...)
}


