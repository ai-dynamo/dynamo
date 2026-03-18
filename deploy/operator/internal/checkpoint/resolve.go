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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type CheckpointInfo struct {
	Enabled        bool
	Exists         bool
	Identity       *nvidiacomv1alpha1.DynamoCheckpointIdentity
	Hash           string
	Location       string
	StorageType    nvidiacomv1alpha1.DynamoCheckpointStorageType
	CheckpointName string
	Ready          bool
}

func checkpointInfoFromObject(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (*CheckpointInfo, error) {
	hash, err := checkpointIdentityHash(ckpt)
	if err != nil {
		return nil, err
	}

	return &CheckpointInfo{
		Enabled:        true,
		Exists:         true,
		Identity:       &ckpt.Spec.Identity,
		Hash:           hash,
		Location:       ckpt.Status.Location,
		StorageType:    ckpt.Status.StorageType,
		CheckpointName: ckpt.Name,
		Ready:          ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
	}, nil
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

		return checkpointInfoFromObject(ckpt)
	case config.Identity == nil:
		return nil, fmt.Errorf("checkpoint enabled but no checkpointRef or identity provided")
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
	info.Identity = config.Identity
	return info, nil
}

func ResolveCheckpointStorage(
	hash string,
	config *configv1alpha1.CheckpointConfiguration,
) (string, nvidiacomv1alpha1.DynamoCheckpointStorageType, error) {
	storageType := configv1alpha1.CheckpointStorageTypePVC
	if config != nil && config.Storage.Type != "" {
		storageType = config.Storage.Type
	}

	switch storageType {
	case configv1alpha1.CheckpointStorageTypeS3:
		if config == nil || config.Storage.S3.URI == "" {
			return "", "", fmt.Errorf("S3 storage type selected but no S3 URI configured (set checkpoint.storage.s3.uri)")
		}
		return fmt.Sprintf("%s/%s.tar", config.Storage.S3.URI, hash), nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType), nil
	case configv1alpha1.CheckpointStorageTypeOCI:
		if config == nil || config.Storage.OCI.URI == "" {
			return "", "", fmt.Errorf("OCI storage type selected but no OCI URI configured (set checkpoint.storage.oci.uri)")
		}
		return fmt.Sprintf("%s:%s", config.Storage.OCI.URI, hash), nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType), nil
	default:
		if config == nil || config.Storage.PVC.BasePath == "" {
			return "", "", fmt.Errorf("PVC storage type selected but no PVC base path configured (set checkpoint.storage.pvc.basePath)")
		}
		return fmt.Sprintf("%s/%s", config.Storage.PVC.BasePath, hash), nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType), nil
	}
}
