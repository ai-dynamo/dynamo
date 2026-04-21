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

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

// ApplyRestorePodMetadata writes restore-target labels and annotations for a
// restore pod, including the container-name list the snapshot agent must act
// on. When checkpoint is disabled or not yet ready, per-container status
// annotations and the container list are cleared.
func ApplyRestorePodMetadata(labels map[string]string, annotations map[string]string, checkpointInfo *CheckpointInfo, containerNames []string) {
	enabled := checkpointInfo != nil && checkpointInfo.Enabled && checkpointInfo.Ready
	hash := ""
	artifactVersion := ""
	if enabled {
		hash = checkpointInfo.Hash
		artifactVersion = checkpointInfo.ArtifactVersion
	}
	snapshotprotocol.ApplyRestoreTargetMetadata(labels, annotations, enabled, hash, artifactVersion)
	if enabled {
		annotations[snapshotprotocol.CheckpointContainersAnnotation] = snapshotprotocol.FormatCheckpointContainers(containerNames)
	} else {
		delete(annotations, snapshotprotocol.CheckpointContainersAnnotation)
	}
}

// InjectCheckpointIntoPodSpec shapes podSpec for the listed workload
// containers as a restore target when checkpoint is enabled. In PR 1 callers
// always pass []string{"main"}; PR 2 will vary this for failover pods.
func InjectCheckpointIntoPodSpec(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	containerNames []string,
) error {
	if checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}
	if len(containerNames) == 0 {
		return fmt.Errorf("checkpoint inject requires at least one container name")
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
	if err := snapshotprotocol.PrepareRestorePodSpecForCheckpoint(
		ctx,
		reader,
		namespace,
		podSpec,
		containerNames,
		info.Hash,
		info.ArtifactVersion,
		snapshotprotocol.DefaultSeccompLocalhostProfile,
		info.Ready,
	); err != nil {
		return err
	}

	EnsurePodInfoVolume(podSpec)
	for _, name := range containerNames {
		container := findContainer(podSpec.Containers, name)
		if container == nil {
			return fmt.Errorf("no container named %q found in pod spec", name)
		}
		EnsurePodInfoMount(container)
		if info.Ready && info.GPUMemoryService != nil && info.GPUMemoryService.Enabled {
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
			EnsureGMSRestoreSidecars(podSpec, container, storage)
		}
	}

	return nil
}

func findContainer(containers []corev1.Container, name string) *corev1.Container {
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i]
		}
	}
	return nil
}
