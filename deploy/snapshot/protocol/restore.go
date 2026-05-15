// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"context"
	"fmt"
	"math"
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
	if storage.PVCName != "" {
		InjectCheckpointVolume(podSpec, storage.PVCName)
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
			InjectCheckpointVolumeMount(container, storage.BasePath)
		}
		EnsureControlVolume(podSpec, container)
		if isCheckpointReady {
			container.Command = []string{"sleep", "infinity"}
			container.Args = nil
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
			FailureThreshold: math.MaxInt32,
			SuccessThreshold: 1,
		}
		return
	}

	startup = startup.DeepCopy()
	startup.InitialDelaySeconds = 0
	startup.PeriodSeconds = 1
	startup.FailureThreshold = math.MaxInt32
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
	if storage.PVCName != "" {
		hasVolume := false
		for _, volume := range podSpec.Volumes {
			if volume.Name == CheckpointVolumeName &&
				volume.PersistentVolumeClaim != nil &&
				volume.PersistentVolumeClaim.ClaimName == storage.PVCName {
				hasVolume = true
				break
			}
		}
		if !hasVolume {
			return fmt.Errorf("missing %s volume for PVC %s", CheckpointVolumeName, storage.PVCName)
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
			hasMount := false
			for _, mount := range container.VolumeMounts {
				if mount.Name == CheckpointVolumeName && mount.MountPath == storage.BasePath {
					hasMount = true
					break
				}
			}
			if !hasMount {
				return fmt.Errorf("missing %s mount at %s on container %q", CheckpointVolumeName, storage.BasePath, name)
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

// InjectCheckpointVolume adds the checkpoint PVC volume to the pod spec if
// not already present. Used by both the snapshot protocol and the operator's
// GMS checkpoint wiring.
func InjectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
	for _, volume := range podSpec.Volumes {
		if volume.Name == CheckpointVolumeName {
			return
		}
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
}

func InjectCheckpointVolumeMount(container *corev1.Container, basePath string) {
	for _, mount := range container.VolumeMounts {
		if mount.Name == CheckpointVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      CheckpointVolumeName,
		MountPath: basePath,
	})
}
