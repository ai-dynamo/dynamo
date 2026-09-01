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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpointjob"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// SourceKind identifies the API that owns a checkpoint reference.
type SourceKind string

const (
	// SourceKindLegacy identifies a DynamoCheckpoint reference retained until
	// the legacy checkpoint implementation is removed.
	SourceKindLegacy SourceKind = "DynamoCheckpoint"
	// SourceKindPodSnapshot identifies a standalone Snapshot PodSnapshot.
	SourceKindPodSnapshot SourceKind = "PodSnapshot"
)

type CheckpointInfo struct {
	Enabled          bool
	Exists           bool
	AutomaticCapture bool
	SourceKind       SourceKind
	GPUMemoryService *nvidiacomv1alpha1.GPUMemoryServiceSpec
	Hash             string
	ArtifactVersion  string
	CheckpointName   string
	Ready            bool
	StartupPolicy    nvidiacomv1alpha1.CheckpointStartupPolicy
	// Empty means the restore pod targets the default main container.
	RestoreTargetContainers []string
	// NativeSnapshot is non-nil when CheckpointName identifies a standalone
	// Snapshot PodSnapshot rather than a legacy DynamoCheckpoint.
	NativeSnapshot *ResolvedPodSnapshot
}

// UsesPodSnapshot reports whether standalone Snapshot owns the capture or
// restore path, including an automatic capture whose PodSnapshot is pending.
// CheckpointInfo must be non-nil.
func (info *CheckpointInfo) UsesPodSnapshot() bool {
	return info.SourceKind == SourceKindPodSnapshot
}

func checkpointInfoFromObject(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (*CheckpointInfo, error) {
	hash, err := CheckpointID(ckpt)
	if err != nil {
		return nil, err
	}

	return &CheckpointInfo{
		Enabled:          true,
		Exists:           true,
		SourceKind:       SourceKindLegacy,
		GPUMemoryService: ckpt.Spec.GPUMemoryService,
		Hash:             hash,
		ArtifactVersion:  checkpointArtifactVersion(ckpt),
		CheckpointName:   ckpt.Name,
		Ready:            ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
	}, nil
}

func checkpointArtifactVersion(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) string {
	if ckpt == nil {
		return snapshotprotocol.DefaultCheckpointArtifactVersion
	}
	return snapshotprotocol.ArtifactVersion(ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation])
}

func ResolveLegacyCheckpointForService(
	ctx context.Context,
	c client.Reader,
	namespace string,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
) (*CheckpointInfo, error) {
	startupPolicy := nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
	if config != nil && config.StartupPolicy != "" {
		startupPolicy = config.StartupPolicy
	}
	switch {
	case config == nil || !config.Enabled:
		return &CheckpointInfo{Enabled: false}, nil
	case config.CheckpointRef != nil && *config.CheckpointRef != "":
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
		if err := c.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      *config.CheckpointRef,
		}, ckpt); err != nil {
			return nil, fmt.Errorf("failed to get referenced checkpoint %s: %w", *config.CheckpointRef, err)
		}

		info, err := checkpointInfoFromObject(ckpt)
		if err != nil {
			return nil, err
		}
		if config.TargetContainerName != "" {
			info.RestoreTargetContainers = []string{config.TargetContainerName}
		}
		info.StartupPolicy = startupPolicy
		return info, nil
	case config.Identity == nil:
		return &CheckpointInfo{
			Enabled:       true,
			SourceKind:    SourceKindLegacy,
			StartupPolicy: startupPolicy,
		}, nil
	}

	hash, err := ComputeIdentityHash(*config.Identity)
	if err != nil {
		return nil, fmt.Errorf("failed to compute identity hash: %w", err)
	}

	existing, err := FindCheckpointByIdentityHash(ctx, c, namespace, hash, "")
	if err != nil {
		return nil, err
	}
	if existing == nil {
		return &CheckpointInfo{
			Enabled:       true,
			SourceKind:    SourceKindLegacy,
			Hash:          hash,
			StartupPolicy: startupPolicy,
		}, nil
	}

	info, err := checkpointInfoFromObject(existing)
	if err != nil {
		return nil, err
	}
	if config.TargetContainerName != "" {
		info.RestoreTargetContainers = []string{config.TargetContainerName}
	}
	info.StartupPolicy = startupPolicy
	return info, nil
}
