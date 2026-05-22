/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"fmt"

	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
)

const (
	GMSLoaderContainer = "gms-loader"
	GMSSaverContainer  = "gms-saver"

	gmsCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	gmsCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"
)

// EnsureGMSRestoreSidecars adds the GMS server init sidecar and loader.
// The loader is a regular sidecar; the GMS RO lock — not init-phase ordering —
// gates the restored engine on weight load. Idempotent.
func EnsureGMSRestoreSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	for _, c := range podSpec.Containers {
		if c.Name == GMSLoaderContainer {
			return
		}
	}

	loader := gms.Container(GMSLoaderContainer, gmsCheckpointLoaderModule, mainContainer.Image)
	loader.VolumeMounts = append(loader.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	podSpec.Containers = append(podSpec.Containers, loader)
}

// EnsureGMSCheckpointJobSidecars adds the GMS server init sidecar and saver
// as a regular Job container. Saver is a regular container (not init+sleep)
// so Job completion gates on tensor write.
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

	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	saver := gms.Container(GMSSaverContainer, gmsCheckpointSaverModule, mainContainer.Image)
	saver.VolumeMounts = append(saver.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	podSpec.Containers = append(podSpec.Containers, saver)
	return nil
}
