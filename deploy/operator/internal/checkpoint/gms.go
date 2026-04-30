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

	GMSCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	GMSCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"

	// EnvCheckpointDir is the environment variable name for the GMS
	// checkpoint artifact directory on the snapshot PVC.
	EnvCheckpointDir = "GMS_CHECKPOINT_DIR"
	// EnvPodUID exposes metadata.uid to GMS saver/loader sidecars so their
	// completion files are unique per pod attempt.
	EnvPodUID = "GMS_POD_UID"
	// EnvTransferBackend selects the GMS loader byte-transfer backend.
	EnvTransferBackend = "GMS_TRANSFER_BACKEND"
	// EnvLoadWorkers controls the GMS loader's per-device transfer concurrency.
	EnvLoadWorkers = "GMS_LOAD_WORKERS"
)

// EnsureGMSRestoreSidecars adds the GMS server init sidecar and restore loader.
// The server startup probe gates socket readiness before the regular containers
// start; the loader then overlaps GMS weight loading with snapshot restore.
func EnsureGMSRestoreSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	podSpec.InitContainers = removeGMSManagedContainers(podSpec.InitContainers, gms.ServerContainerName, GMSLoaderContainer)
	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	loader := gms.Container(GMSLoaderContainer, GMSCheckpointLoaderModule, mainContainer.Image)
	loader.VolumeMounts = append(loader.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	loader.Env = append(loader.Env,
		corev1.EnvVar{Name: EnvCheckpointDir, Value: ResolveGMSArtifactDir(storage)},
		PodUIDEnvVar(),
	)
	loader.Env = append(loader.Env, loaderPassThroughEnvVars(mainContainer)...)

	podSpec.Containers = removeGMSManagedContainers(podSpec.Containers, gms.ServerContainerName, GMSLoaderContainer)
	podSpec.Containers = append(podSpec.Containers, loader)
}

// EnsureGMSCheckpointJobSidecars adds the GMS server init sidecar and checkpoint
// saver. The saver is a regular Job container so the Job completes after save.
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

	podSpec.InitContainers = removeGMSManagedContainers(podSpec.InitContainers, gms.ServerContainerName, GMSSaverContainer)
	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	saver := gms.Container(GMSSaverContainer, GMSCheckpointSaverModule, mainContainer.Image)
	saver.VolumeMounts = append(saver.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	saver.Env = append(saver.Env,
		corev1.EnvVar{Name: EnvCheckpointDir, Value: gmsArtifactDir},
		PodUIDEnvVar(),
	)
	podSpec.Containers = removeGMSManagedContainers(podSpec.Containers, gms.ServerContainerName, GMSSaverContainer)
	podSpec.Containers = append(podSpec.Containers, saver)
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

func loaderPassThroughEnvVars(mainContainer *corev1.Container) []corev1.EnvVar {
	var result []corev1.EnvVar
	for _, env := range mainContainer.Env {
		switch env.Name {
		case EnvTransferBackend, EnvLoadWorkers:
			result = append(result, env)
		}
	}
	return result
}

func ResolveGMSArtifactDir(storage snapshotprotocol.Storage) string {
	// GMS data lives under /checkpoints/gms/<hash>/versions/<version>
	// separate from the CRIU tree (/checkpoints/<hash>/versions/<version>)
	// so the non-root saver can create directories at the PVC root.
	artifactVersion := filepath.Base(storage.Location)
	checkpointID := filepath.Base(filepath.Dir(filepath.Dir(storage.Location)))
	return filepath.Join(storage.BasePath, "gms", checkpointID, "versions", artifactVersion)
}

func removeGMSManagedContainers(containers []corev1.Container, names ...string) []corev1.Container {
	managed := make(map[string]struct{}, len(names))
	for _, name := range names {
		managed[name] = struct{}{}
	}

	filtered := containers[:0]
	for _, container := range containers {
		if _, ok := managed[container.Name]; ok {
			continue
		}
		filtered = append(filtered, container)
	}
	return filtered
}
