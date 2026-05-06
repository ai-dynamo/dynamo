/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package checkpoint

import (
	"context"
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

func ApplyRestorePodMetadata(labels map[string]string, annotations map[string]string, checkpointInfo *CheckpointInfo) {
	enabled := checkpointInfo != nil && checkpointInfo.Enabled && checkpointInfo.Ready
	hash := ""
	artifactVersion := ""
	if enabled {
		hash = checkpointInfo.Hash
		artifactVersion = checkpointInfo.ArtifactVersion
	}
	snapshotprotocol.ApplyRestoreTargetMetadata(labels, annotations, enabled, hash, artifactVersion)
	if !enabled {
		delete(annotations, snapshotprotocol.TargetContainersAnnotation)
		return
	}
	targets := checkpointInfo.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	annotations[snapshotprotocol.TargetContainersAnnotation] = snapshotprotocol.FormatTargetContainers(targets)
}

func RequireMainContainer(podSpec *corev1.PodSpec) (*corev1.Container, error) {
	if podSpec == nil {
		return nil, fmt.Errorf("pod spec is nil")
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			return &podSpec.Containers[i], nil
		}
	}
	return nil, fmt.Errorf("pod spec has no container named %q", commonconsts.MainContainerName)
}

// InjectCheckpointIntoPodSpec mutates a worker pod spec for restore-from-checkpoint
// once the referenced DynamoCheckpoint is Ready. It adds the snapshot-control
// volume + mount, sets DYN_SNAPSHOT_CONTROL_DIR, attaches the checkpoint PVC,
// applies the localhost seccomp profile (when seccompProfile is non-empty), and
// — for GMS-enabled workloads — wires the GMS restore sidecars. No-ops when
// checkpointInfo is nil, disabled, or not Ready (cold-start path).
//
// seccompProfile is the path passed to the localhost seccomp profile injection;
// pass an empty string to skip seccomp injection (e.g. on OpenShift, or when
// running a CRIU build with io_uring support). Callers typically obtain this
// from operatorConfig.Checkpoint.EffectiveSeccompProfile().
func InjectCheckpointIntoPodSpec(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	seccompProfile string,
) error {
	// Only mutate the worker pod spec once the checkpoint is Ready. Before
	// the checkpoint exists, the worker must cold-start normally without
	// the snapshot-control volume, DYN_SNAPSHOT_CONTROL_DIR, checkpoint PVC
	// mount, or localhost seccomp profile — otherwise the Python worker
	// enters checkpoint mode on env-var presence and sits quiesced waiting
	// for a sentinel that only the checkpoint Job and restore-target path
	// produce. The checkpoint Job itself is built separately through
	// buildCheckpointJob + NewCheckpointJob and does get these.
	if checkpointInfo == nil || !checkpointInfo.Enabled || !checkpointInfo.Ready {
		return nil
	}

	info := checkpointInfo
	if info.Hash == "" {
		if info.Identity == nil {
			return fmt.Errorf("checkpoint enabled but identity is nil and hash is not set")
		}

		hash, err := ComputeIdentityHash(*info.Identity)
		if err != nil {
			return fmt.Errorf("failed to compute identity hash: %w", err)
		}
		info.Hash = hash
	}

	if reader == nil {
		return fmt.Errorf("checkpoint client is required")
	}
	targets := checkpointInfo.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	podInfoContainers := make([]*corev1.Container, 0, len(targets))
	for _, name := range targets {
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				podInfoContainers = append(podInfoContainers, &podSpec.Containers[i])
				break
			}
		}
	}
	if len(podInfoContainers) != len(targets) {
		return fmt.Errorf("checkpoint restore targets %v do not all exist in pod spec", targets)
	}
	syntheticAnnotations := map[string]string{
		snapshotprotocol.TargetContainersAnnotation: snapshotprotocol.FormatTargetContainers(targets),
	}
	if err := snapshotprotocol.PrepareRestorePodSpecForCheckpoint(
		ctx,
		reader,
		namespace,
		podSpec,
		syntheticAnnotations,
		info.Hash,
		info.ArtifactVersion,
		seccompProfile,
		info.Ready,
	); err != nil {
		return err
	}

	EnsurePodInfoVolume(podSpec)
	for _, c := range podInfoContainers {
		EnsurePodInfoMount(c)
	}
	if info.Ready && info.GPUMemoryService != nil && info.GPUMemoryService.Enabled {
		// GMS restore is still single-main-container only.
		mainContainer, err := RequireMainContainer(podSpec)
		if err != nil {
			return fmt.Errorf("gpuMemoryService enabled: %w", err)
		}
		storage, err := snapshotprotocol.DiscoverAndResolveStorage(
			ctx,
			reader,
			namespace,
			info.Hash,
			info.ArtifactVersion,
		)
		if err != nil {
			return err
		}
		EnsureGMSRestoreSidecars(podSpec, mainContainer, storage)
	}

	return nil
}
