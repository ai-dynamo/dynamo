// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	SnapshotAgentLabelKey      = "app.kubernetes.io/component"
	SnapshotAgentLabelValue    = "snapshot-agent"
	SnapshotAgentContainerName = "agent"
	SnapshotAgentVolumeName    = "checkpoints"
	SnapshotAgentLabelSelector = SnapshotAgentLabelKey + "=" + SnapshotAgentLabelValue
)

type PodOptions struct {
	Namespace       string
	CheckpointID    string
	ArtifactVersion string
	Storage         Storage
	SeccompProfile  string
}

const (
	// RestoreStandbyModeEnv asks Dynamo backend entrypoints to capture restore
	// context and sleep instead of cold-starting the workload. Generic
	// images that do not honor this env must still provide their own inert
	// restore command.
	RestoreStandbyModeEnv          = "DYN_SNAPSHOT_RESTORE_STANDBY"
	restoreStartupFailureThreshold = 1800 // 30 minutes at 1s cadence.
)

// NewRestorePod shapes every annotated target container for restore.
func NewRestorePod(pod *corev1.Pod, opts PodOptions) (*corev1.Pod, error) {
	pod = pod.DeepCopy()
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	ApplyRestoreTargetMetadata(pod.Labels, pod.Annotations, true, opts.CheckpointID, opts.ArtifactVersion)
	if err := PrepareRestorePodSpec(&pod.Spec, pod.Annotations, opts.Storage, opts.SeccompProfile, true); err != nil {
		return nil, err
	}
	pod.Namespace = opts.Namespace
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	return pod, nil
}

// PrepareRestorePodSpec applies restore shaping to annotated target containers.
// It does not change container command/args. Once the checkpoint is ready, it
// sets DYN_SNAPSHOT_RESTORE_STANDBY=1 so Dynamo standby entrypoints
// sleep before CRIU restore; generic images that do not honor the env must
// still provide their own inert restore command.
func PrepareRestorePodSpec(
	podSpec *corev1.PodSpec,
	annotations map[string]string,
	storage Storage,
	seccompProfile string,
	isCheckpointReady bool,
) error {
	if podSpec == nil {
		return fmt.Errorf("pod spec is nil")
	}
	targets, err := TargetContainersFromAnnotations(annotations, 1, 0)
	if err != nil {
		return fmt.Errorf("restore pod spec: %w", err)
	}
	EnsureLocalhostSeccompProfile(podSpec, seccompProfile)
	checkpointVolumeName := CheckpointVolumeName
	if storage.PVCName != "" {
		checkpointVolumeName, err = InjectCheckpointVolume(podSpec, storage.PVCName)
		if err != nil {
			return fmt.Errorf("restore pod spec: %w", err)
		}
	}
	for _, name := range targets {
		var container *corev1.Container
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				container = &podSpec.Containers[i]
				break
			}
		}
		if container == nil {
			return fmt.Errorf("restore target container %q not found in pod spec (from %s annotation)", name, TargetContainersAnnotation)
		}
		if storage.BasePath != "" {
			if err := InjectCheckpointVolumeMount(container, checkpointVolumeName, storage.BasePath); err != nil {
				return fmt.Errorf("restore pod spec: %w", err)
			}
		}
		EnsureControlVolume(podSpec, container)
		if isCheckpointReady {
			// Dynamo standby entrypoints honor this env by writing restore
			// context and sleeping. Keep command/args intact so generic images
			// can provide their own inert restore entrypoint when needed.
			foundRestoreStandbyModeEnv := false
			for i := range container.Env {
				if container.Env[i].Name == RestoreStandbyModeEnv {
					container.Env[i].Value = "1"
					container.Env[i].ValueFrom = nil
					foundRestoreStandbyModeEnv = true
					break
				}
			}
			if !foundRestoreStandbyModeEnv {
				container.Env = append(container.Env, corev1.EnvVar{
					Name:  RestoreStandbyModeEnv,
					Value: "1",
				})
			}
			ensureRestoreStartupProbe(container)
		}
	}
	return nil
}

// ensureRestoreStartupProbe installs a StartupProbe that gates Ready until
// CRIU restore completes. It prefers the workload's existing Startup/Liveness/
// Readiness probe (deep-copied with tightened cadence and infinite retries),
// and falls back to a sentinel-file exec probe when none is defined.
func ensureRestoreStartupProbe(container *corev1.Container) {
	startup := container.StartupProbe
	if startup == nil {
		startup = container.LivenessProbe
		if startup == nil {
			startup = container.ReadinessProbe
		}
	}
	if startup == nil {
		container.StartupProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{"cat", filepath.Join(SnapshotControlMountPath, RestoreCompleteFile)},
				},
			},
			TimeoutSeconds:   1,
			PeriodSeconds:    1,
			FailureThreshold: restoreStartupFailureThreshold,
			SuccessThreshold: 1,
		}
		return
	}

	startup = startup.DeepCopy()
	startup.InitialDelaySeconds = 0
	startup.PeriodSeconds = 1
	startup.FailureThreshold = restoreStartupFailureThreshold
	startup.SuccessThreshold = 1
	container.StartupProbe = startup
}

// ValidateRestorePodSpec verifies the target containers are restore-shaped.
func ValidateRestorePodSpec(
	podSpec *corev1.PodSpec,
	annotations map[string]string,
	storage Storage,
	seccompProfile string,
) error {
	if podSpec == nil {
		return fmt.Errorf("pod spec is nil")
	}
	targets, err := TargetContainersFromAnnotations(annotations, 1, 0)
	if err != nil {
		return err
	}
	checkpointVolumeName := CheckpointVolumeName
	if storage.PVCName != "" {
		checkpointVolumeName, err = resolveCheckpointVolumeName(podSpec, storage.PVCName)
		if err != nil {
			return err
		}
		if checkpointVolumeName == "" {
			return fmt.Errorf("missing volume for checkpoint PVC %q", storage.PVCName)
		}
	}
	hasControlVolume := false
	for _, volume := range podSpec.Volumes {
		if volume.Name == SnapshotControlVolumeName && volume.EmptyDir != nil {
			hasControlVolume = true
			break
		}
	}
	if !hasControlVolume {
		return fmt.Errorf("missing %s emptyDir volume; add it via snapshotprotocol.EnsureControlVolume", SnapshotControlVolumeName)
	}
	for _, name := range targets {
		var container *corev1.Container
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				container = &podSpec.Containers[i]
				break
			}
		}
		if container == nil {
			return fmt.Errorf("restore target container %q not found in pod spec (from %s annotation)", name, TargetContainersAnnotation)
		}
		if storage.BasePath != "" {
			hasMount, err := hasCompatibleCheckpointVolumeMount(container, checkpointVolumeName, storage.BasePath)
			if err != nil {
				return err
			}
			if !hasMount {
				return fmt.Errorf(
					"missing mount of checkpoint volume %q at %q on container %q",
					checkpointVolumeName,
					storage.BasePath,
					name,
				)
			}
		}
		hasControlMount := false
		for _, mount := range container.VolumeMounts {
			if mount.Name == SnapshotControlVolumeName && mount.MountPath == SnapshotControlMountPath {
				hasControlMount = true
				if mount.SubPath != name {
					return fmt.Errorf("expected SubPath %q for %s at %s on container %q, got %q", name, SnapshotControlVolumeName, SnapshotControlMountPath, name, mount.SubPath)
				}
				break
			}
		}
		if !hasControlMount {
			return fmt.Errorf("missing %s mount at %s on container %q", SnapshotControlVolumeName, SnapshotControlMountPath, name)
		}
		hasControlEnv := false
		for _, env := range container.Env {
			if env.Name == SnapshotControlDirEnv {
				hasControlEnv = true
				break
			}
		}
		if !hasControlEnv {
			return fmt.Errorf("missing %s env var on container %q", SnapshotControlDirEnv, name)
		}
		if container.StartupProbe == nil {
			return fmt.Errorf("missing restore-complete startup probe on container %q", name)
		}
	}
	if seccompProfile == "" {
		return nil
	}
	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		return fmt.Errorf("missing localhost seccomp profile")
	}
	profile := podSpec.SecurityContext.SeccompProfile
	if profile.Type != corev1.SeccompProfileTypeLocalhost || profile.LocalhostProfile == nil || *profile.LocalhostProfile != seccompProfile {
		return fmt.Errorf("expected localhost seccomp profile %q", seccompProfile)
	}
	return nil
}

func DiscoverStorageFromDaemonSets(namespace string, daemonSets []appsv1.DaemonSet) (Storage, error) {
	if len(daemonSets) == 0 {
		return Storage{}, fmt.Errorf("no snapshot-agent daemonset found in namespace %s", namespace)
	}

	names := make([]string, 0, len(daemonSets))
	for _, daemonSet := range daemonSets {
		names = append(names, daemonSet.Name)

		mountPaths := map[string]string{}
		for _, container := range daemonSet.Spec.Template.Spec.Containers {
			if container.Name != SnapshotAgentContainerName {
				continue
			}
			for _, mount := range container.VolumeMounts {
				if strings.TrimSpace(mount.MountPath) == "" {
					continue
				}
				mountPaths[mount.Name] = strings.TrimRight(mount.MountPath, "/")
			}
		}

		for _, volume := range daemonSet.Spec.Template.Spec.Volumes {
			if volume.Name != SnapshotAgentVolumeName {
				continue
			}
			if volume.PersistentVolumeClaim == nil {
				continue
			}

			basePath, ok := mountPaths[volume.Name]
			if !ok || basePath == "" {
				continue
			}

			pvcName := strings.TrimSpace(volume.PersistentVolumeClaim.ClaimName)
			if pvcName == "" {
				continue
			}

			return Storage{
				Type:     StorageTypePVC,
				PVCName:  pvcName,
				BasePath: basePath,
			}, nil
		}
	}

	return Storage{}, fmt.Errorf(
		"snapshot-agent daemonset in %s does not mount a PVC-backed checkpoint volume (%s)",
		namespace,
		strings.Join(names, ", "),
	)
}

// DiscoverAndResolveStorage lists snapshot-agent DaemonSets in the given
// namespace, discovers the shared storage configuration, and resolves the
// checkpoint-specific path for the given checkpoint ID and artifact version.
func DiscoverAndResolveStorage(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	checkpointID string,
	artifactVersion string,
) (Storage, error) {
	if reader == nil {
		return Storage{}, fmt.Errorf("snapshot client is required")
	}

	daemonSets := &appsv1.DaemonSetList{}
	if err := reader.List(
		ctx,
		daemonSets,
		ctrlclient.InNamespace(namespace),
		ctrlclient.MatchingLabels{SnapshotAgentLabelKey: SnapshotAgentLabelValue},
	); err != nil {
		return Storage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}

	storage, err := DiscoverStorageFromDaemonSets(namespace, daemonSets.Items)
	if err != nil {
		return Storage{}, err
	}

	return ResolveCheckpointStorage(checkpointID, artifactVersion, storage)
}

// PrepareRestorePodSpecForCheckpoint discovers storage, then shapes targets.
func PrepareRestorePodSpecForCheckpoint(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	annotations map[string]string,
	checkpointID string,
	artifactVersion string,
	seccompProfile string,
	isCheckpointReady bool,
) error {
	storage, err := DiscoverAndResolveStorage(ctx, reader, namespace, checkpointID, artifactVersion)
	if err != nil {
		return err
	}

	return PrepareRestorePodSpec(podSpec, annotations, storage, seccompProfile, isCheckpointReady)
}

// InjectCheckpointVolume resolves the unique volume backed by the checkpoint
// PVC, adding the default volume when none exists.
func InjectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) (string, error) {
	volumeName, err := resolveCheckpointVolumeName(podSpec, pvcName)
	if err != nil {
		return "", err
	}
	if volumeName != "" {
		return volumeName, nil
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
	return CheckpointVolumeName, nil
}

func resolveCheckpointVolumeName(podSpec *corev1.PodSpec, pvcName string) (string, error) {
	var match string
	var matchReadOnly bool
	var reserved bool
	for _, volume := range podSpec.Volumes {
		if volume.Name == CheckpointVolumeName {
			reserved = true
		}
		if volume.PersistentVolumeClaim != nil &&
			volume.PersistentVolumeClaim.ClaimName == pvcName {
			if match != "" {
				return "", fmt.Errorf("multiple volumes reference checkpoint PVC %q", pvcName)
			}
			match = volume.Name
			matchReadOnly = volume.PersistentVolumeClaim.ReadOnly
		}
	}
	if matchReadOnly {
		return "", fmt.Errorf("checkpoint volume %q for PVC %q must be writable", match, pvcName)
	}
	if match == "" && reserved {
		return "", fmt.Errorf(
			"volume %q is already in use and cannot reference checkpoint PVC %q",
			CheckpointVolumeName,
			pvcName,
		)
	}
	return match, nil
}

// InjectCheckpointVolumeMount ensures a root mount at the checkpoint base path.
func InjectCheckpointVolumeMount(
	container *corev1.Container,
	volumeName string,
	basePath string,
) error {
	found, err := hasCompatibleCheckpointVolumeMount(container, volumeName, basePath)
	if err != nil || found {
		return err
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      volumeName,
		MountPath: basePath,
	})
	return nil
}

func hasCompatibleCheckpointVolumeMount(
	container *corev1.Container,
	volumeName string,
	basePath string,
) (bool, error) {
	var found bool
	for _, mount := range container.VolumeMounts {
		if mount.MountPath != basePath {
			continue
		}
		if found {
			return false, fmt.Errorf("container %q has multiple volume mounts at checkpoint base path %q", container.Name, basePath)
		}
		found = true
		if mount.Name != volumeName {
			return false, fmt.Errorf("container %q mounts volume %q at checkpoint base path %q, expected volume %q", container.Name, mount.Name, basePath, volumeName)
		}
		if mount.SubPath != "" || mount.SubPathExpr != "" {
			return false, fmt.Errorf("checkpoint volume %q mount at %q on container %q must not use subPath or subPathExpr", volumeName, basePath, container.Name)
		}
		if mount.ReadOnly {
			return false, fmt.Errorf("checkpoint volume %q mount at %q on container %q must be writable", volumeName, basePath, container.Name)
		}
	}

	return found, nil
}
