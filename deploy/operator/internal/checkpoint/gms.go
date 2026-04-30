/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"fmt"
	"path/filepath"

	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const (
	GMSLoaderContainer = "gms-loader"
	GMSSaverContainer  = "gms-saver"

	GMSCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	GMSCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"

	// EnvCheckpointDir is the environment variable name for the GMS
	// checkpoint artifact directory on the snapshot PVC.
	EnvCheckpointDir = "GMS_CHECKPOINT_DIR"
	// EnvPodUID exposes metadata.uid to GMS saver/loader sidecars so their
	// completion files are unique per pod attempt.
	EnvPodUID = "GMS_POD_UID"
)

// EnsureGMSRestoreSidecars appends GMS server + loader containers to the pod
// spec for a checkpoint restore. They are regular containers so Kubernetes can
// start them in parallel with the restore target; snapshot-agent only needs the
// restore target's container ID before it can launch nsrestore.
func EnsureGMSRestoreSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	gms.EnsureSharedVolume(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	server := gms.Container(gms.ServerContainerName, gms.ServerModule, mainContainer.Image)

	loader := gms.Container(GMSLoaderContainer, GMSCheckpointLoaderModule, mainContainer.Image)
	loader.VolumeMounts = append(loader.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	loader.Env = append(loader.Env,
		corev1.EnvVar{Name: EnvCheckpointDir, Value: ResolveGMSArtifactDir(storage)},
		PodUIDEnvVar(),
	)

	podSpec.InitContainers = removeGMSRestoreSidecars(podSpec.InitContainers)
	podSpec.Containers = removeGMSRestoreSidecars(podSpec.Containers)
	podSpec.Containers = append(podSpec.Containers, server, loader)
}

// EnsureGMSCheckpointJobSidecars adds GMS server (init) + saver containers
// to the pod spec for a checkpoint job.
func EnsureGMSCheckpointJobSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) error {
	if podSpec == nil || mainContainer == nil {
		return nil
	}
	if len(mainContainer.Resources.Claims) == 0 {
		return fmt.Errorf("gms sidecars require main container resource claims (DRA must be enabled)")
	}
	if storage.PVCName == "" || storage.BasePath == "" || storage.Location == "" {
		return fmt.Errorf("gms checkpoint jobs require resolved checkpoint storage")
	}

	gmsArtifactDir := ResolveGMSArtifactDir(storage)

	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	saver := gms.Container(GMSSaverContainer, GMSCheckpointSaverModule, mainContainer.Image)
	saver.VolumeMounts = append(saver.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	saver.Env = append(saver.Env,
		corev1.EnvVar{Name: EnvCheckpointDir, Value: gmsArtifactDir},
		PodUIDEnvVar(),
	)
	// The saver is an init sidecar (restartPolicy=Always) so it doesn't
	// affect pod Ready (only the worker's probe matters) and doesn't block
	// Job completion. It saves, then sleeps until the pod terminates.
	saver.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
	podSpec.InitContainers = append(podSpec.InitContainers, saver)
	return nil
}

func PodUIDEnvVar() corev1.EnvVar {
	return corev1.EnvVar{
		Name: EnvPodUID,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: "metadata.uid",
			},
		},
	}
}

func ResolveGMSArtifactDir(storage snapshotprotocol.Storage) string {
	// GMS data lives under /checkpoints/gms/<hash>/versions/<version>
	// separate from the CRIU tree (/checkpoints/<hash>/versions/<version>)
	// so the non-root saver can create directories at the PVC root.
	artifactVersion := filepath.Base(storage.Location)
	checkpointID := filepath.Base(filepath.Dir(filepath.Dir(storage.Location)))
	return filepath.Join(storage.BasePath, "gms", checkpointID, "versions", artifactVersion)
}

func removeGMSRestoreSidecars(containers []corev1.Container) []corev1.Container {
	filtered := containers[:0]
	for _, container := range containers {
		if container.Name == gms.ServerContainerName || container.Name == GMSLoaderContainer {
			continue
		}
		filtered = append(filtered, container)
	}
	return filtered
}
