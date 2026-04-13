// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"context"
	"fmt"
	"math"
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

func NewRestorePod(pod *corev1.Pod, opts PodOptions) *corev1.Pod {
	pod = pod.DeepCopy()
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	ApplyRestoreTargetMetadata(pod.Labels, pod.Annotations, true, opts.CheckpointID, opts.ArtifactVersion)
	container := resolveWorkerContainer(&pod.Spec)
	PrepareRestorePodSpec(&pod.Spec, container, opts.Storage, opts.SeccompProfile, true)
	pod.Namespace = opts.Namespace
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	return pod
}

func resolveWorkerContainer(podSpec *corev1.PodSpec) *corev1.Container {
	if podSpec == nil {
		return nil
	}
	if len(podSpec.Containers) == 1 {
		return &podSpec.Containers[0]
	}
	for index := range podSpec.Containers {
		if podSpec.Containers[index].Name == "main" {
			return &podSpec.Containers[index]
		}
	}
	return nil
}

func PrepareRestorePodSpec(
	podSpec *corev1.PodSpec,
	container *corev1.Container,
	storage Storage,
	seccompProfile string,
	isCheckpointReady bool,
) {
	EnsureLocalhostSeccompProfile(podSpec, seccompProfile)
	if storage.PVCName != "" {
		injectCheckpointVolume(podSpec, storage.PVCName)
	}
	if storage.BasePath != "" {
		injectCheckpointVolumeMount(container, storage.BasePath)
	}
	if isCheckpointReady {
		container.Command = []string{"sleep", "infinity"}
		container.Args = nil
		ensureRestoreStartupProbe(container)
	}
}

func ensureRestoreStartupProbe(container *corev1.Container) {
	startup := container.StartupProbe
	if startup == nil {
		startup = container.LivenessProbe
		if startup == nil {
			startup = container.ReadinessProbe
		}
	}
	if startup == nil {
		return
	}

	startup = startup.DeepCopy()
	startup.FailureThreshold = math.MaxInt32
	startup.SuccessThreshold = 1
	container.StartupProbe = startup
}

func ValidateRestorePodSpec(
	podSpec *corev1.PodSpec,
	storage Storage,
	seccompProfile string,
) error {
	if podSpec == nil {
		return fmt.Errorf("pod spec is nil")
	}
	container := resolveWorkerContainer(podSpec)
	if container == nil {
		return fmt.Errorf("restore target must include a worker container named main")
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
	if storage.BasePath != "" {
		hasMount := false
		for _, mount := range container.VolumeMounts {
			if mount.Name == CheckpointVolumeName && mount.MountPath == storage.BasePath {
				hasMount = true
				break
			}
		}
		if !hasMount {
			return fmt.Errorf("missing %s mount at %s", CheckpointVolumeName, storage.BasePath)
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

func PrepareRestorePodSpecForCheckpoint(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	container *corev1.Container,
	checkpointID string,
	artifactVersion string,
	seccompProfile string,
	isCheckpointReady bool,
) error {
	if reader == nil {
		return fmt.Errorf("snapshot client is required")
	}

	daemonSets := &appsv1.DaemonSetList{}
	if err := reader.List(
		ctx,
		daemonSets,
		ctrlclient.InNamespace(namespace),
		ctrlclient.MatchingLabels{SnapshotAgentLabelKey: SnapshotAgentLabelValue},
	); err != nil {
		return fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}

	storage, err := DiscoverStorageFromDaemonSets(namespace, daemonSets.Items)
	if err != nil {
		return err
	}

	resolvedStorage, err := ResolveCheckpointStorage(checkpointID, artifactVersion, storage)
	if err != nil {
		return err
	}

	PrepareRestorePodSpec(podSpec, container, resolvedStorage, seccompProfile, isCheckpointReady)
	return nil
}

func injectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
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

func injectCheckpointVolumeMount(container *corev1.Container, basePath string) {
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
