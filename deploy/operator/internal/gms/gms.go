/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package gms provides GMS (GPU Memory Service) server container building
// for both steady-state DGD pods and checkpoint/restore flows.
package gms

import (
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

	// EnvUseV1 selects the GMS V1 protocol in engine and GMS processes.
	EnvUseV1 = "DYN_GMS_USE_V1"

	// ServerModule is the Python module for the GMS server entry point.
	ServerModule = "gpu_memory_service.cli.server"
)

// ClientEnvironment returns the owned bindings independently of resource setup,
// so renderers can compose them before installing the GMS sidecar and DRA claim.
func ClientEnvironment(useV1 bool) []corev1.EnvVar {
	bindings := []corev1.EnvVar{socketEnvironment()}
	if useV1 {
		bindings = append(bindings, v1Environment())
	}
	return bindings
}

func socketEnvironment() corev1.EnvVar {
	return corev1.EnvVar{Name: EnvSocketDir, Value: SharedMountPath}
}

func v1Environment() corev1.EnvVar {
	return corev1.EnvVar{Name: EnvUseV1, Value: "true"}
}

// EnsureServerSidecar adds the GMS server as a native sidecar (init +
// restartPolicy=Always). With no StartupProbe, kubelet considers it Started
// as soon as the process is running — clients ride out the sub-second
// socket-bind via connect-retry. Native sidecar status is kept so kubelet
// terminates the server when the Job's regular containers exit; a regular
// container here would keep the Pod in Running forever. Idempotent.
//
// Snapshot-coupled GMS uses the V1 protocol: sidecar, extra GMS clients,
// and the engine container all receive DYN_GMS_USE_V1=true. Intra-pod
// GMS without snapshot keeps the V0 sidecar and does not set that env.
func EnsureServerSidecar(podSpec *corev1.PodSpec, mainContainer *corev1.Container, useV1 bool) []corev1.EnvVar {
	if podSpec == nil || mainContainer == nil {
		return nil
	}

	bindings := EnsureClient(podSpec, mainContainer)
	if useV1 {
		bindings = append(bindings, EnableV1(mainContainer)...)
	}
	ensureServerSidecar(podSpec, mainContainer, useV1)
	return bindings
}

// EnsureServerSidecarResources attaches the server and client resources without
// changing the client's environment. podSpec and mainContainer must not be nil.
func EnsureServerSidecarResources(podSpec *corev1.PodSpec, mainContainer *corev1.Container, useV1 bool) {
	EnsureClientResources(podSpec, mainContainer)
	ensureServerSidecar(podSpec, mainContainer, useV1)
}

// ensureServerSidecar leaves client env composition to the selected caller.
func ensureServerSidecar(podSpec *corev1.PodSpec, mainContainer *corev1.Container, useV1 bool) {
	sidecar := Container(ServerContainerName, ServerModule, mainContainer.Image)
	if useV1 {
		EnableV1(&sidecar)
	}
	sidecar.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == sidecar.Name {
			return
		}
	}
	podSpec.InitContainers = append(podSpec.InitContainers, sidecar)
}

// EnableV1 sets DYN_GMS_USE_V1=true on a GMS process container. Idempotent.
func EnableV1(container *corev1.Container) []corev1.EnvVar {
	if container == nil {
		return nil
	}
	binding := v1Environment()
	for i := range container.Env {
		if container.Env[i].Name == EnvUseV1 {
			container.Env[i] = binding
			return []corev1.EnvVar{binding}
		}
	}
	container.Env = append(container.Env, binding)
	return []corev1.EnvVar{binding}
}

// EnsureClient adds the GMS UDS socket volume, mount, GMS_SOCKET_DIR env var,
// and DRA claim to a GMS client container. Idempotent.
func EnsureClient(podSpec *corev1.PodSpec, container *corev1.Container) []corev1.EnvVar {
	if podSpec == nil || container == nil {
		return nil
	}
	EnsureClientResources(podSpec, container)

	// Legacy callers replace the first binding in place, preserving list order.
	sharedEnv := socketEnvironment()
	for i := range container.Env {
		if container.Env[i].Name == EnvSocketDir {
			container.Env[i] = sharedEnv
			return []corev1.EnvVar{sharedEnv}
		}
	}
	container.Env = append(container.Env, sharedEnv)
	return []corev1.EnvVar{sharedEnv}
}

// EnsureClientResources attaches the socket volume, mount and DRA claim without
// mutating environment sources. podSpec and container must not be nil.
func EnsureClientResources(podSpec *corev1.PodSpec, container *corev1.Container) {
	sharedVolume := corev1.Volume{
		Name:     SharedVolumeName,
		EmptyDir: &corev1.EmptyDirVolumeSource{},
	}
	foundVolume := false
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == SharedVolumeName {
			podSpec.Volumes[i] = sharedVolume
			foundVolume = true
			break
		}
	}
	if !foundVolume {
		podSpec.Volumes = append(podSpec.Volumes, sharedVolume)
	}

	sharedMount := corev1.VolumeMount{Name: SharedVolumeName, MountPath: SharedMountPath}
	foundMount := false
	for i := range container.VolumeMounts {
		if container.VolumeMounts[i].Name == SharedVolumeName {
			container.VolumeMounts[i] = sharedMount
			foundMount = true
			break
		}
	}
	if !foundMount {
		container.VolumeMounts = append(container.VolumeMounts, sharedMount)
	}

	dra.RemoveGPUResources(container.Resources.Limits)
	dra.RemoveGPUResources(container.Resources.Requests)
	foundClaim := false
	for i := range container.Resources.Claims {
		if container.Resources.Claims[i].Name == dra.ClaimName {
			foundClaim = true
			break
		}
	}
	if !foundClaim {
		container.Resources.Claims = append(container.Resources.Claims, corev1.ResourceClaim{Name: dra.ClaimName})
	}
}

// Container builds a GMS container with the shared socket volume, env, and DRA claim.
func Container(name, module, image string) corev1.Container {
	return corev1.Container{
		Name:    name,
		Image:   image,
		Command: []string{"python3", "-m", module},
		Env:     ClientEnvironment(false),
		VolumeMounts: []corev1.VolumeMount{
			{Name: SharedVolumeName, MountPath: SharedMountPath},
		},
		Resources: corev1.ResourceRequirements{
			Claims: []corev1.ResourceClaim{{Name: dra.ClaimName}},
		},
	}
}
