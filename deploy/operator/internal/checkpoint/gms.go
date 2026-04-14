/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"context"
	"fmt"
	"path/filepath"

	gmsruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	GMSLoaderContainer = "gms-loader"
	GMSSaverContainer  = "gms-saver"

	gmsCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	gmsCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"
)

func ResolveGMSCheckpointStorage(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	checkpointID string,
	artifactVersion string,
) (snapshotprotocol.Storage, error) {
	if reader == nil {
		return snapshotprotocol.Storage{}, fmt.Errorf("checkpoint client is required")
	}

	daemonSets := &appsv1.DaemonSetList{}
	if err := reader.List(
		ctx,
		daemonSets,
		ctrlclient.InNamespace(namespace),
		ctrlclient.MatchingLabels{snapshotprotocol.SnapshotAgentLabelKey: snapshotprotocol.SnapshotAgentLabelValue},
	); err != nil {
		return snapshotprotocol.Storage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}

	storage, err := snapshotprotocol.DiscoverStorageFromDaemonSets(namespace, daemonSets.Items)
	if err != nil {
		return snapshotprotocol.Storage{}, err
	}
	return snapshotprotocol.ResolveCheckpointStorage(checkpointID, artifactVersion, storage)
}

// BuildGMSRestoreSidecars prepares GMS infrastructure for a restore pod and
// returns the additional containers the caller must append to podSpec.Containers.
//
// The GMS server runs as a regular container (not init) because the CRIU-restored
// main process already has GPU memory mapped and does not need sockets at
// startup. The gms-loader polls for sockets internally via wait_for_weights_socket.
func BuildGMSRestoreSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) []corev1.Container {
	if podSpec == nil || mainContainer == nil {
		return nil
	}

	// Remove gms-server from initContainers if the DGD-level
	// applyGPUMemoryService already placed it there. For restore pods the
	// server runs as a regular container so that all containers start in
	// parallel — the restored main process does not need sockets at startup.
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == gmsruntime.ServerContainerName {
			podSpec.InitContainers = append(podSpec.InitContainers[:i], podSpec.InitContainers[i+1:]...)
			break
		}
	}

	server := gmsruntime.BuildServerContainer(podSpec, mainContainer)

	loader := gmsCheckpointLoaderContainer(mainContainer.Image)
	copyGMSDeviceClaims(mainContainer, &loader)
	ensureCheckpointVolume(podSpec, storage.PVCName)
	loader.VolumeMounts = append(loader.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	loader.Env = append(loader.Env, corev1.EnvVar{Name: "GMS_CHECKPOINT_DIR", Value: resolveGMSArtifactDir(storage)})

	return []corev1.Container{server, loader}
}

// BuildGMSCheckpointJobSidecars prepares GMS infrastructure for a checkpoint
// job and returns the additional containers the caller must append to
// podSpec.Containers.
func BuildGMSCheckpointJobSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) ([]corev1.Container, error) {
	if podSpec == nil || mainContainer == nil {
		return nil, nil
	}
	if len(mainContainer.Resources.Claims) == 0 {
		return nil, fmt.Errorf("gms sidecars require main container resource claims")
	}
	if storage.PVCName == "" || storage.BasePath == "" || storage.Location == "" {
		return nil, fmt.Errorf("gms checkpoint jobs require resolved checkpoint storage")
	}

	gmsruntime.EnsureServerSidecar(podSpec, mainContainer)
	ensureGMSCheckpointControl(podSpec)

	saver := gmsCheckpointSaverContainer(mainContainer.Image)
	copyGMSDeviceClaims(mainContainer, &saver)
	ensureCheckpointVolume(podSpec, storage.PVCName)
	saver.VolumeMounts = append(saver.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	saver.Env = append(saver.Env, corev1.EnvVar{Name: "GMS_CHECKPOINT_DIR", Value: resolveGMSArtifactDir(storage)})

	return []corev1.Container{saver}, nil
}

func resolveGMSArtifactDir(storage snapshotprotocol.Storage) string {
	checkpointRoot := filepath.Dir(filepath.Dir(storage.Location))
	artifactVersion := filepath.Base(storage.Location)
	return filepath.Join(checkpointRoot, "gms", "versions", artifactVersion)
}

func gmsCheckpointLoaderContainer(image string) corev1.Container {
	container := corev1.Container{
		Name:    GMSLoaderContainer,
		Image:   image,
		Command: []string{"python3", "-m", gmsCheckpointLoaderModule},
		Env: []corev1.EnvVar{
			{Name: "TMPDIR", Value: gmsruntime.SharedMountPath},
			{Name: "GMS_SOCKET_DIR", Value: gmsruntime.SharedMountPath},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: gmsruntime.SharedVolumeName, MountPath: gmsruntime.SharedMountPath},
		},
	}
	return container
}

func gmsCheckpointSaverContainer(image string) corev1.Container {
	container := corev1.Container{
		Name:    GMSSaverContainer,
		Image:   image,
		Command: []string{"python3", "-m", gmsCheckpointSaverModule},
		Env: []corev1.EnvVar{
			{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
			{Name: "TMPDIR", Value: gmsruntime.SharedMountPath},
			{Name: "GMS_SOCKET_DIR", Value: gmsruntime.SharedMountPath},
			{Name: "GMS_CONTROL_DIR", Value: gmsruntime.ControlDir},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: gmsruntime.SharedVolumeName, MountPath: gmsruntime.SharedMountPath},
			{Name: gmsruntime.ControlVolumeName, MountPath: gmsruntime.ControlDir},
		},
	}
	return container
}

// ensureGMSCheckpointControl adds the control volume and injects
// GMS_CONTROL_DIR into the GMS server container for checkpoint coordination.
func ensureGMSCheckpointControl(podSpec *corev1.PodSpec) {
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name:         gmsruntime.ControlVolumeName,
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	})
	server := gmsruntime.FindServerContainer(podSpec)
	if server != nil {
		server.VolumeMounts = append(server.VolumeMounts, corev1.VolumeMount{Name: gmsruntime.ControlVolumeName, MountPath: gmsruntime.ControlDir})
		server.Env = append(server.Env, corev1.EnvVar{Name: "GMS_CONTROL_DIR", Value: gmsruntime.ControlDir})
	}
}

func copyGMSDeviceClaims(mainContainer *corev1.Container, container *corev1.Container) {
	if mainContainer == nil || container == nil || len(mainContainer.Resources.Claims) == 0 {
		return
	}
	container.Resources.Claims = append([]corev1.ResourceClaim{}, mainContainer.Resources.Claims...)
}

func ensureCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
	if pvcName == "" {
		return
	}
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == snapshotprotocol.CheckpointVolumeName {
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: snapshotprotocol.CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: pvcName},
		},
	})
}


