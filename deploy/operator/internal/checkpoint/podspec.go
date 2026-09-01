/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"fmt"
	"path"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	podcontract "github.com/ai-dynamo/snapshot/api/podcontract"
	corev1 "k8s.io/api/core/v1"
)

const restoreStartupFailureThreshold = 1800 // 30 minutes at 1s cadence.

// ApplyRestoreCandidateMetadata writes Dynamo's private admission handoff for
// a resolved PodSnapshot. Pending automatic captures deliberately produce no
// candidate metadata until their PodSnapshot is ready to resolve.
func ApplyRestoreCandidateMetadata(annotations map[string]string, checkpointInfo *CheckpointInfo) error {
	if annotations == nil {
		return fmt.Errorf("checkpoint restore candidate annotations map is required")
	}
	removeRestoreCandidateMetadata(annotations)
	if checkpointInfo == nil || !checkpointInfo.Enabled || !checkpointInfo.Exists || checkpointInfo.CheckpointName == "" {
		return nil
	}
	if checkpointInfo.NativeSnapshot == nil {
		return fmt.Errorf("restore candidate requires a resolved PodSnapshot")
	}

	targets := checkpointInfo.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	annotations[commonconsts.CheckpointRestoreCandidateAnnotation] = commonconsts.KubeLabelValueTrue
	annotations[commonconsts.CheckpointNameAnnotation] = checkpointInfo.CheckpointName
	annotations[commonconsts.SnapshotCandidateUIDAnnotation] = string(checkpointInfo.NativeSnapshot.UID)
	annotations[commonconsts.SnapshotCandidateContentAnnotation] = checkpointInfo.NativeSnapshot.BoundContentName
	annotations[commonconsts.SnapshotCandidateGMSModeAnnotation] = checkpointInfo.NativeSnapshot.GMSMode
	annotations[commonconsts.SnapshotCandidateVersionAnnotation] = checkpointInfo.NativeSnapshot.CompatibilityVersion
	annotations[commonconsts.RestoreCandidateTargetContainersAnnotation] = strings.Join(targets, ",")
	startupPolicy := checkpointInfo.StartupPolicy
	if startupPolicy == "" {
		startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
	}
	annotations[commonconsts.CheckpointStartupPolicyAnnotation] = string(startupPolicy)
	return nil
}

func removeRestoreCandidateMetadata(annotations map[string]string) {
	delete(annotations, commonconsts.CheckpointRestoreCandidateAnnotation)
	delete(annotations, commonconsts.CheckpointNameAnnotation)
	delete(annotations, commonconsts.CheckpointStartupPolicyAnnotation)
	delete(annotations, commonconsts.SnapshotCandidateUIDAnnotation)
	delete(annotations, commonconsts.SnapshotCandidateContentAnnotation)
	delete(annotations, commonconsts.SnapshotCandidateGMSModeAnnotation)
	delete(annotations, commonconsts.SnapshotCandidateVersionAnnotation)
	delete(annotations, commonconsts.RestoreCandidateTargetContainersAnnotation)
}

// RestoreCandidateTargetContainers reads Dynamo's candidate-only restore destinations.
func RestoreCandidateTargetContainers(annotations map[string]string) ([]string, error) {
	raw, ok := annotations[commonconsts.RestoreCandidateTargetContainersAnnotation]
	if !ok || strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("missing required %s annotation", commonconsts.RestoreCandidateTargetContainersAnnotation)
	}

	// Normalize the comma-separated list while rejecting ambiguous destinations.
	parts := strings.Split(raw, ",")
	seen := make(map[string]struct{}, len(parts))
	targets := make([]string, 0, len(parts))
	for _, part := range parts {
		name := strings.TrimSpace(part)
		if name == "" {
			return nil, fmt.Errorf("empty container name in %s=%q", commonconsts.RestoreCandidateTargetContainersAnnotation, raw)
		}
		if _, duplicate := seen[name]; duplicate {
			return nil, fmt.Errorf("duplicate container name %q in %s=%q", name, commonconsts.RestoreCandidateTargetContainersAnnotation, raw)
		}
		seen[name] = struct{}{}
		targets = append(targets, name)
	}
	return targets, nil
}

// EnsureRestoreStartupProbe installs a StartupProbe that gates Ready until
// CRIU restore completes. It prefers the workload's existing Startup/Liveness/
// Readiness probe (deep-copied with tightened cadence and infinite retries),
// and falls back to a sentinel-file exec probe when none is defined.
func EnsureRestoreStartupProbe(container *corev1.Container) {
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
					Command: []string{"cat", path.Join(podcontract.SnapshotControlMountPath, podcontract.RestoreCompleteFile)},
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

// EnsureIntraPodGPUMemoryService wires the in-pod GMS server sidecar and
// socket clients for SnapshotJob capture pod specs.
func EnsureIntraPodGPUMemoryService(
	podSpec *corev1.PodSpec,
	targetContainers []*corev1.Container,
	extraClientContainerNames []string,
) {
	if len(targetContainers) == 0 {
		return
	}
	gms.EnsureServerSidecar(podSpec, targetContainers[0])
	for _, container := range targetContainers {
		gms.EnsureClient(podSpec, container)
	}
	for _, name := range extraClientContainerNames {
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				gms.EnsureClient(podSpec, &podSpec.Containers[i])
				break
			}
		}
	}
}
