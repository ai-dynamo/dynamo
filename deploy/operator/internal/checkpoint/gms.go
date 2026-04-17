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

	gmsCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	gmsCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"

	// envCheckpointDir is the environment variable name for the GMS
	// checkpoint artifact directory on the snapshot PVC.
	envCheckpointDir = "GMS_CHECKPOINT_DIR"
)

// EnsureGMSRestoreSidecars adds GMS server + loader containers to the pod
// spec for a checkpoint restore.
//
// Precondition: the upstream DGD path has already run EnsureServerSidecar,
// which adds the GMS server as a blocking init container (so socket
// readiness gates the workload). The restore path does not need the
// server to block startup because the CRIU-restored workload brings its
// GPU memory back mapped and all init sidecars must start in parallel.
// This function therefore:
//
//  1. Removes the existing init gms-server added by EnsureServerSidecar.
//  2. Re-adds it as an always-restarting init sidecar together with the
//     gms-loader, which hydrates the server from the checkpoint artifact
//     directory on the snapshot PVC.
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

	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	server := gms.Container(gms.ServerContainerName, gms.ServerModule, mainContainer.Image)
	server.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)

	loader := gms.Container(GMSLoaderContainer, gmsCheckpointLoaderModule, mainContainer.Image)
	loader.VolumeMounts = append(loader.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	loader.Env = append(loader.Env, corev1.EnvVar{Name: envCheckpointDir, Value: resolveGMSArtifactDir(storage)})
	loader.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)

	podSpec.InitContainers = append(podSpec.InitContainers, server, loader)
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

	saver := gms.Container(GMSSaverContainer, gmsCheckpointSaverModule, mainContainer.Image)
	saver.VolumeMounts = append(saver.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	saver.Env = append(saver.Env, corev1.EnvVar{Name: envCheckpointDir, Value: gmsArtifactDir})
	// The saver is an init sidecar (restartPolicy=Always) so it doesn't
	// affect pod Ready (only the worker's probe matters) and doesn't block
	// Job completion. It saves, then sleeps until the pod terminates.
	saver.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
	podSpec.InitContainers = append(podSpec.InitContainers, saver)
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
