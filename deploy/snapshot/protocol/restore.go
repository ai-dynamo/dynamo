// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"path/filepath"

	corev1 "k8s.io/api/core/v1"
)

type PodOptions struct {
	Namespace       string
	CheckpointID    string
	ArtifactVersion string
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
	if err := PrepareRestorePodSpec(&pod.Spec, pod.Annotations, opts.SeccompProfile, true); err != nil {
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
	seccompProfile string,
) error {
	if podSpec == nil {
		return fmt.Errorf("pod spec is nil")
	}
	targets, err := TargetContainersFromAnnotations(annotations, 1, 0)
	if err != nil {
		return err
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
