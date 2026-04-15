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
}

func InjectCheckpointIntoPodSpec(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
) error {
	if checkpointInfo == nil || !checkpointInfo.Enabled {
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

	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("no container found to inject checkpoint config")
	}
	var mainContainer *corev1.Container
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			mainContainer = &podSpec.Containers[i]
			break
		}
	}
	if mainContainer == nil {
		return fmt.Errorf("main container not found in pod spec")
	}
	if reader == nil {
		return fmt.Errorf("checkpoint client is required")
	}
	if err := snapshotprotocol.PrepareRestorePodSpecForCheckpoint(
		ctx,
		reader,
		namespace,
		podSpec,
		mainContainer,
		info.Hash,
		info.ArtifactVersion,
		snapshotprotocol.DefaultSeccompLocalhostProfile,
		info.Ready,
	); err != nil {
		return err
	}

	EnsurePodInfoVolume(podSpec)
	EnsurePodInfoMount(mainContainer)
	return nil
}
