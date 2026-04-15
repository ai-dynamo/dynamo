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
)

const (
	GMSLoaderContainer = "gms-loader"
	GMSSaverContainer  = "gms-saver"

	gmsCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	gmsCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"

	// envCheckpointDir is the environment variable name for the GMS
	// checkpoint artifact directory on the snapshot PVC.
	envCheckpointDir = "GMS_CHECKPOINT_DIR"


)

// EnsureGMSRestoreSidecars adds GMS server + loader containers to the pod spec
// for a checkpoint restore. The server runs as a regular container (not init)
// because the CRIU-restored main process already has GPU memory mapped and
// all containers must start in parallel.
func EnsureGMSRestoreSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	// The DGD path adds the GMS server as an init sidecar (blocks until
	// sockets are ready). For restore, move it to a regular container so
	// all containers start in parallel.
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == gms.ServerContainerName {
			podSpec.InitContainers = append(podSpec.InitContainers[:i], podSpec.InitContainers[i+1:]...)
			break
		}
	}
	gms.EnsureSharedVolume(podSpec, mainContainer)

	loader := gms.Container(GMSLoaderContainer, gmsCheckpointLoaderModule, mainContainer.Image)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)
	loader.VolumeMounts = append(loader.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	loader.Env = append(loader.Env, corev1.EnvVar{Name: envCheckpointDir, Value: resolveGMSArtifactDir(storage)})

	podSpec.Containers = append(podSpec.Containers, gms.Container(gms.ServerContainerName, gms.ServerModule, mainContainer.Image), loader)
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

	gmsArtifactDir := resolveGMSArtifactDir(storage)

	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	// Tell the GMS server where to look for the checkpoint-done stop file.
	// The saver writes it to gmsArtifactDir after saving completes.
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == gms.ServerContainerName {
			podSpec.InitContainers[i].VolumeMounts = append(podSpec.InitContainers[i].VolumeMounts,
				corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
			podSpec.InitContainers[i].Env = append(podSpec.InitContainers[i].Env,
				corev1.EnvVar{Name: envCheckpointDir, Value: gmsArtifactDir})
			break
		}
	}

	saver := gms.Container(GMSSaverContainer, gmsCheckpointSaverModule, mainContainer.Image)
	saver.VolumeMounts = append(saver.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	saver.Env = append(saver.Env, corev1.EnvVar{Name: envCheckpointDir, Value: gmsArtifactDir})

	podSpec.Containers = append(podSpec.Containers, saver)
	return nil
}

func resolveGMSArtifactDir(storage snapshotprotocol.Storage) string {
	// GMS data lives under /checkpoints/gms/<hash>/versions/<version>
	// separate from the CRIU tree (/checkpoints/<hash>/versions/<version>)
	// so the non-root saver can create directories at the PVC root.
	artifactVersion := filepath.Base(storage.Location)
	checkpointID := filepath.Base(filepath.Dir(filepath.Dir(storage.Location)))
	return filepath.Join(storage.BasePath, "gms", checkpointID, "versions", artifactVersion)
}




