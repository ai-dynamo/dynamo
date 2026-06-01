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
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type CheckpointInfo struct {
	Enabled          bool
	Exists           bool
	Identity         *nvidiacomv1alpha1.DynamoCheckpointIdentity
	GPUMemoryService *nvidiacomv1alpha1.GPUMemoryServiceSpec
	Hash             string
	ArtifactVersion  string
	CheckpointName   string
	Ready            bool
	// Empty means the restore pod targets the default main container.
	RestoreTargetContainers []string
}

func checkpointInfoFromObject(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (*CheckpointInfo, error) {
	hash, err := CheckpointID(ckpt)
	if err != nil {
		return nil, err
	}

	return &CheckpointInfo{
		Enabled:          true,
		Exists:           true,
		Identity:         &ckpt.Spec.Identity,
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

func ResolveCheckpointForService(
	ctx context.Context,
	c client.Client,
	namespace string,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
) (*CheckpointInfo, error) {
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
		if err := validateResolvedGMSSnapshotGate(info); err != nil {
			return nil, err
		}
		if config.TargetContainerName != "" {
			info.RestoreTargetContainers = []string{config.TargetContainerName}
		}
		return info, nil
	case config.Identity == nil:
		// Manual mode with neither checkpointRef nor identity cannot resolve or
		// create anything: the DGD controller only creates DynamoCheckpoint CRs
		// in Auto mode, so without a ref or identity a Manual checkpoint would
		// silently never become Ready. Fail fast instead. (Auto mode legitimately
		// reaches here with a nil identity; the controller owns CR creation.)
		if config.Mode == nvidiacomv1alpha1.CheckpointModeManual {
			return nil, fmt.Errorf(
				"checkpoint Manual mode requires checkpointRef or identity to be set")
		}
		return &CheckpointInfo{
			Enabled: true,
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
			Enabled:  true,
			Identity: config.Identity,
			Hash:     hash,
		}, nil
	}

	info, err := checkpointInfoFromObject(existing)
	if err != nil {
		return nil, err
	}
	if err := validateResolvedGMSSnapshotGate(info); err != nil {
		return nil, err
	}
	info.Identity = config.Identity
	if config.TargetContainerName != "" {
		info.RestoreTargetContainers = []string{config.TargetContainerName}
	}
	return info, nil
}

func validateResolvedGMSSnapshotGate(info *CheckpointInfo) error {
	if info == nil {
		return nil
	}
	return ValidateGMSSnapshotGate("checkpoint.gpuMemoryService", info.Enabled, info.GPUMemoryService)
}
