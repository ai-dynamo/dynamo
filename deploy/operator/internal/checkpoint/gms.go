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

	gmsCheckpointLoaderModule = "gpu_memory_service.cli.gms_checkpoint_loader"
	gmsCheckpointSaverModule  = "gpu_memory_service.cli.gms_checkpoint_saver"
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

// EnsureGMSRestoreSidecars adds the GMS server sidecar and loader container
// for a restore pod. Uses the shared runtime helper for server/volume setup.
func EnsureGMSRestoreSidecars(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	gmsruntime.EnsureServerSidecar(podSpec, mainContainer)

	loader := gmsCheckpointLoaderContainer(mainContainer.Image)
	copyGMSDeviceClaims(mainContainer, &loader)
	ensureGMSContainer(podSpec, loader)
}

// EnsureGMSRestoreHelperMounts adds the checkpoint storage volume and mount
// to the GMS loader container and sets GMS_CHECKPOINT_DIR.
func EnsureGMSRestoreHelperMounts(podSpec *corev1.PodSpec, storage snapshotprotocol.Storage) {
	loader := findContainer(podSpec, GMSLoaderContainer)
	if loader == nil {
		return
	}
	ensureCheckpointVolume(podSpec, storage.PVCName)
	ensureVolumeMount(loader, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	setEnv(loader, "GMS_CHECKPOINT_DIR", resolveGMSArtifactDir(storage))
}

// EnsureGMSCheckpointJobSidecars adds the GMS server sidecar, saver container,
// and checkpoint control volume for a checkpoint job pod.
func EnsureGMSCheckpointJobSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) error {
	if podSpec == nil || mainContainer == nil {
		return nil
	}
	if len(mainContainer.Resources.Claims) == 0 {
		return fmt.Errorf("gms sidecars require main container resource claims")
	}
	if storage.PVCName == "" || storage.BasePath == "" || storage.Location == "" {
		return fmt.Errorf("gms checkpoint jobs require resolved checkpoint storage")
	}

	gmsruntime.EnsureServerSidecar(podSpec, mainContainer)
	ensureGMSCheckpointControl(podSpec)

	saver := gmsCheckpointSaverContainer(mainContainer.Image)
	copyGMSDeviceClaims(mainContainer, &saver)
	ensureCheckpointVolume(podSpec, storage.PVCName)
	ensureVolumeMount(&saver, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	setEnv(&saver, "GMS_CHECKPOINT_DIR", resolveGMSArtifactDir(storage))
	ensureGMSContainer(podSpec, saver)
	return nil
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
	ensureVolume(podSpec, corev1.Volume{
		Name:         gmsruntime.ControlVolumeName,
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	})
	server := gmsruntime.FindServerContainer(podSpec)
	if server != nil {
		ensureVolumeMount(server, corev1.VolumeMount{Name: gmsruntime.ControlVolumeName, MountPath: gmsruntime.ControlDir})
		setEnv(server, "GMS_CONTROL_DIR", gmsruntime.ControlDir)
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

func ensureGMSContainer(podSpec *corev1.PodSpec, container corev1.Container) {
	if findContainer(podSpec, container.Name) != nil {
		return
	}
	podSpec.Containers = append(podSpec.Containers, container)
}

func findContainer(podSpec *corev1.PodSpec, name string) *corev1.Container {
	if podSpec == nil {
		return nil
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == name {
			return &podSpec.Containers[i]
		}
	}
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == name {
			return &podSpec.InitContainers[i]
		}
	}
	return nil
}
