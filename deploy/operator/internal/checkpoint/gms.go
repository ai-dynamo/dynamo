/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"fmt"
	"path/filepath"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const (
	GMSLoaderContainer = "gms-loader"
	GMSSaverContainer  = "gms-saver"
	GMSArtifactVolume  = "gms-artifact-storage"

	GMSCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	GMSCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"

	// EnvCheckpointDir is the snapshot-agent-visible GMS control directory
	// where saver/loader sidecars write completion sentinels.
	EnvCheckpointDir = "GMS_CHECKPOINT_DIR"
	// EnvWeightsCheckpointDir optionally points GMS saver/loader sidecars at
	// the PVC path that stores weight bytes. EnvCheckpointDir remains the
	// snapshot-agent-visible control/sentinel directory.
	EnvWeightsCheckpointDir = "GMS_WEIGHTS_CHECKPOINT_DIR"
	// EnvPodUID exposes metadata.uid to GMS saver/loader sidecars so their
	// completion files are unique per pod attempt.
	EnvPodUID = "GMS_POD_UID"
	// EnvTransferBackend selects the GMS loader byte-transfer backend.
	EnvTransferBackend = "GMS_TRANSFER_BACKEND"
	// EnvLoadWorkers controls the GMS loader's per-device transfer concurrency.
	EnvLoadWorkers = "GMS_LOAD_WORKERS"
	// EnvSaveWorkers controls the GMS saver's per-device write concurrency.
	EnvSaveWorkers = "GMS_SAVE_WORKERS"
	// EnvLocalSSDRoots lists host-local SSD roots used by test-only local SSD
	// backends. Checkpoint and restore must land on the same node.
	EnvLocalSSDRoots = "GMS_LOCAL_SSD_ROOTS"
	// EnvShardSizeBytes controls GMS save shard size / local SSD stripe size.
	EnvShardSizeBytes = "GMS_SHARD_SIZE_BYTES"
)

type GMSCheckpointStorage struct {
	Control     snapshotprotocol.Storage
	Artifacts   snapshotprotocol.Storage
	ControlDir  string
	ArtifactDir string
}

// EnsureGMSRestoreSidecars adds the GMS server init sidecar and restore loader.
// The server is a restartable init container without a startup probe; clients
// wait for its socket so restore and GMS load can start without kubelet gating.
func EnsureGMSRestoreSidecars(
	podSpec *corev1.PodSpec,
	targetContainers []*corev1.Container,
	storage GMSCheckpointStorage,
) {
	if podSpec == nil || len(targetContainers) == 0 {
		return
	}

	var sidecarSource *corev1.Container
	for _, targetContainer := range targetContainers {
		if targetContainer == nil {
			continue
		}
		if sidecarSource == nil {
			sidecarSource = targetContainer
		}
		gms.EnsureSharedVolume(podSpec, targetContainer)
	}
	if sidecarSource == nil {
		return
	}
	podSpec.InitContainers = removeGMSManagedContainers(podSpec.InitContainers, gms.ServerContainerName, GMSLoaderContainer)
	ensureGMSCheckpointVolumes(podSpec, storage)
	gms.EnsureServerSidecar(podSpec, sidecarSource)

	loader := gms.Container(GMSLoaderContainer, GMSCheckpointLoaderModule, sidecarSource.Image)
	addGMSStorageMounts(&loader, storage)
	addGMSLocalSSDVolumeMounts(&loader, sidecarSource)
	loader.Env = append(loader.Env,
		corev1.EnvVar{Name: EnvCheckpointDir, Value: storage.ControlDir},
		PodUIDEnvVar(),
	)
	if storage.ArtifactDir != storage.ControlDir {
		loader.Env = append(loader.Env, corev1.EnvVar{Name: EnvWeightsCheckpointDir, Value: storage.ArtifactDir})
	}
	loader.Env = append(loader.Env, gmsCheckpointPassThroughEnvVars(sidecarSource)...)

	podSpec.Containers = removeGMSManagedContainers(podSpec.Containers, gms.ServerContainerName, GMSLoaderContainer)
	podSpec.Containers = append(podSpec.Containers, loader)
}

// EnsureGMSCheckpointJobSidecars adds the GMS server and checkpoint saver as
// restartable init sidecars so Job completion is driven by the main container.
func EnsureGMSCheckpointJobSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage GMSCheckpointStorage,
) error {
	if podSpec == nil || mainContainer == nil {
		return nil
	}
	if len(mainContainer.Resources.Claims) == 0 {
		return fmt.Errorf("gms sidecars require main container resource claims (DRA must be enabled)")
	}
	if storage.Control.PVCName == "" || storage.Control.BasePath == "" || storage.ControlDir == "" {
		return fmt.Errorf("gms checkpoint jobs require resolved checkpoint storage")
	}

	podSpec.InitContainers = removeGMSManagedContainers(podSpec.InitContainers, gms.ServerContainerName, GMSSaverContainer)
	podSpec.Containers = removeGMSManagedContainers(podSpec.Containers, gms.ServerContainerName, GMSSaverContainer)
	gms.EnsureServerSidecar(podSpec, mainContainer)
	ensureGMSCheckpointVolumes(podSpec, storage)

	saver := gms.Container(GMSSaverContainer, GMSCheckpointSaverModule, mainContainer.Image)
	addGMSStorageMounts(&saver, storage)
	addGMSLocalSSDVolumeMounts(&saver, mainContainer)
	saver.Env = append(saver.Env,
		corev1.EnvVar{Name: EnvCheckpointDir, Value: storage.ControlDir},
		PodUIDEnvVar(),
	)
	if storage.ArtifactDir != storage.ControlDir {
		saver.Env = append(saver.Env, corev1.EnvVar{Name: EnvWeightsCheckpointDir, Value: storage.ArtifactDir})
	}
	saver.Env = append(saver.Env, gmsCheckpointPassThroughEnvVars(mainContainer)...)
	// The saver is an init sidecar (restartPolicy=Always) so it doesn't
	// affect pod Ready (only the worker's probe matters) and doesn't block
	// Job completion. It saves, then sleeps until the pod terminates.
	saver.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
	podSpec.InitContainers = append(podSpec.InitContainers, saver)
	return nil
}

func ResolveGMSCheckpointStorage(
	snapshotStorage snapshotprotocol.Storage,
	gmsSpec *nvidiacomv1alpha1.GPUMemoryServiceSpec,
) (GMSCheckpointStorage, error) {
	controlDir := ResolveGMSArtifactDir(snapshotStorage)
	resolved := GMSCheckpointStorage{
		Control:     snapshotStorage,
		Artifacts:   snapshotStorage,
		ControlDir:  controlDir,
		ArtifactDir: controlDir,
	}
	resolved.Control.Location = controlDir
	resolved.Artifacts.Location = controlDir

	if gmsSpec == nil || gmsSpec.ArtifactStorage == nil {
		return resolved, nil
	}

	checkpointID, artifactVersion, err := checkpointIDAndArtifactVersion(snapshotStorage)
	if err != nil {
		return GMSCheckpointStorage{}, err
	}
	artifactConfig := snapshotprotocol.Storage{
		Type:     snapshotprotocol.StorageTypePVC,
		PVCName:  strings.TrimSpace(gmsSpec.ArtifactStorage.PVCName),
		BasePath: strings.TrimSpace(gmsSpec.ArtifactStorage.BasePath),
	}
	artifactStorage, err := snapshotprotocol.ResolveCheckpointStorage(checkpointID, artifactVersion, artifactConfig)
	if err != nil {
		return GMSCheckpointStorage{}, fmt.Errorf("resolve GMS artifact storage: %w", err)
	}
	if artifactStorage.PVCName == "" {
		return GMSCheckpointStorage{}, fmt.Errorf("gms artifact storage pvcName is required")
	}

	resolved.Artifacts = artifactStorage
	resolved.ArtifactDir = artifactStorage.Location
	return resolved, nil
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

func gmsCheckpointPassThroughEnvVars(mainContainer *corev1.Container) []corev1.EnvVar {
	var result []corev1.EnvVar
	for _, env := range mainContainer.Env {
		switch env.Name {
		case EnvTransferBackend, EnvLoadWorkers, EnvSaveWorkers, EnvLocalSSDRoots, EnvShardSizeBytes:
			result = append(result, env)
		}
	}
	return result
}

func addGMSLocalSSDVolumeMounts(container *corev1.Container, mainContainer *corev1.Container) {
	roots := gmsLocalSSDRoots(mainContainer)
	if len(roots) == 0 {
		return
	}
	for _, mount := range mainContainer.VolumeMounts {
		for _, root := range roots {
			if rootUsesMount(root, mount.MountPath) {
				addVolumeMount(container, mount.Name, mount.MountPath)
				break
			}
		}
	}
}

func gmsLocalSSDRoots(mainContainer *corev1.Container) []string {
	for _, env := range mainContainer.Env {
		if env.Name != EnvLocalSSDRoots || strings.TrimSpace(env.Value) == "" {
			continue
		}
		var roots []string
		for _, part := range strings.Split(env.Value, ",") {
			root := strings.TrimSpace(part)
			if root != "" {
				roots = append(roots, filepath.Clean(root))
			}
		}
		return roots
	}
	return nil
}

func rootUsesMount(root string, mountPath string) bool {
	cleanRoot := filepath.Clean(root)
	cleanMount := filepath.Clean(mountPath)
	return cleanRoot == cleanMount || strings.HasPrefix(cleanRoot, cleanMount+string(filepath.Separator))
}

func ResolveGMSArtifactDir(storage snapshotprotocol.Storage) string {
	// GMS data lives under /checkpoints/gms/<hash>/versions/<version>
	// separate from the CRIU tree (/checkpoints/<hash>/versions/<version>)
	// so the non-root saver can create directories at the PVC root.
	artifactVersion := filepath.Base(storage.Location)
	checkpointID := filepath.Base(filepath.Dir(filepath.Dir(storage.Location)))
	return filepath.Join(storage.BasePath, "gms", checkpointID, "versions", artifactVersion)
}

func checkpointIDAndArtifactVersion(storage snapshotprotocol.Storage) (string, string, error) {
	if strings.TrimSpace(storage.Location) == "" {
		return "", "", fmt.Errorf("resolved snapshot checkpoint location is required")
	}
	artifactVersion := filepath.Base(storage.Location)
	checkpointID := filepath.Base(filepath.Dir(filepath.Dir(storage.Location)))
	if checkpointID == "." || checkpointID == string(filepath.Separator) || checkpointID == "" {
		return "", "", fmt.Errorf("could not parse checkpoint ID from %q", storage.Location)
	}
	if artifactVersion == "." || artifactVersion == string(filepath.Separator) || artifactVersion == "" {
		return "", "", fmt.Errorf("could not parse artifact version from %q", storage.Location)
	}
	return checkpointID, artifactVersion, nil
}

func ensureGMSCheckpointVolumes(podSpec *corev1.PodSpec, storage GMSCheckpointStorage) {
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.Control.PVCName)
	if gmsArtifactNeedsSeparateMount(storage) {
		ensurePVCVolume(podSpec, GMSArtifactVolume, storage.Artifacts.PVCName)
	}
}

func gmsArtifactNeedsSeparateMount(storage GMSCheckpointStorage) bool {
	return storage.Artifacts.PVCName != storage.Control.PVCName || storage.Artifacts.BasePath != storage.Control.BasePath
}

func ensurePVCVolume(podSpec *corev1.PodSpec, name string, pvcName string) {
	for _, volume := range podSpec.Volumes {
		if volume.Name == name {
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: name,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
}

func addGMSStorageMounts(container *corev1.Container, storage GMSCheckpointStorage) {
	addVolumeMount(container, snapshotprotocol.CheckpointVolumeName, storage.Control.BasePath)
	if gmsArtifactNeedsSeparateMount(storage) {
		addVolumeMount(container, GMSArtifactVolume, storage.Artifacts.BasePath)
	}
}

func addVolumeMount(container *corev1.Container, volumeName string, mountPath string) {
	for _, mount := range container.VolumeMounts {
		if mount.Name == volumeName && mount.MountPath == mountPath {
			return
		}
	}
	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{Name: volumeName, MountPath: mountPath})
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
