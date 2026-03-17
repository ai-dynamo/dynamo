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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func getCheckpointInfoFromCheckpoint(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (*CheckpointInfo, error) {
	hash, err := getCheckpointIdentityHash(ckpt)
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

func getCheckpointIdentityHash(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (string, error) {
	if ckpt.Status.IdentityHash != "" {
		return ckpt.Status.IdentityHash, nil
	}

	computedHash, err := ComputeIdentityHash(ckpt.Spec.Identity)
	if err != nil {
		return "", fmt.Errorf("failed to compute checkpoint hash for %s: %w", ckpt.Name, err)
	}

	return computedHash, nil
}

func FindCheckpointByIdentityHash(
	ctx context.Context,
	c client.Client,
	namespace string,
	hash string,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	checkpoints := &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(
		ctx,
		checkpoints,
		client.InNamespace(namespace),
		client.MatchingLabels{consts.KubeLabelCheckpointHash: hash},
	); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints by hash label: %w", err)
	}

	var existing *nvidiacomv1alpha1.DynamoCheckpoint
	seen := make(map[string]struct{}, len(checkpoints.Items))
	for i := range checkpoints.Items {
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for identity hash %s", hash)
		}
		existing = checkpoints.Items[i].DeepCopy()
		seen[checkpoints.Items[i].Name] = struct{}{}
	}
	if existing != nil {
		return existing, nil
	}

	checkpoints = &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(ctx, checkpoints, client.InNamespace(namespace)); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints: %w", err)
	}
	for i := range checkpoints.Items {
		if _, ok := seen[checkpoints.Items[i].Name]; ok {
			continue
		}

		existingHash, err := getCheckpointIdentityHash(&checkpoints.Items[i])
		if err != nil {
			return nil, err
		}
		if existingHash != hash {
			continue
		}
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for identity hash %s", hash)
		}
		existing = checkpoints.Items[i].DeepCopy()
	}

	return existing, nil
}

func CreateOrGetAutoCheckpoint(
	ctx context.Context,
	c client.Client,
	namespace string,
	identity nvidiacomv1alpha1.DynamoCheckpointIdentity,
	podTemplate corev1.PodTemplateSpec,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	hash, err := ComputeIdentityHash(identity)
	if err != nil {
		return nil, fmt.Errorf("failed to compute identity hash: %w", err)
	}

	existing, err := FindCheckpointByIdentityHash(ctx, c, namespace, hash)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		return existing, nil
	}

	ckptName := fmt.Sprintf("checkpoint-%s", hash)
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ckptName,
			Namespace: namespace,
			Labels: map[string]string{
				consts.KubeLabelCheckpointHash: hash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: podTemplate,
			},
		},
	}

	if err := c.Create(ctx, ckpt); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return nil, fmt.Errorf("failed to create checkpoint %s: %w", ckptName, err)
		}

		existing = &nvidiacomv1alpha1.DynamoCheckpoint{}
		key := types.NamespacedName{Name: ckptName, Namespace: namespace}
		if getErr := c.Get(ctx, key, existing); getErr != nil {
			return nil, fmt.Errorf("failed to get checkpoint %s after already exists: %w", ckptName, getErr)
		}

		existingHash, err := getCheckpointIdentityHash(existing)
		if err != nil {
			return nil, err
		}
		if existingHash != hash {
			return nil, fmt.Errorf("checkpoint %s already exists with identity hash %s", ckptName, existingHash)
		}
		return existing, nil
	}

	return ckpt, nil
}

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

func ResolveCheckpointForService(
	ctx context.Context,
	c client.Client,
	namespace string,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
) (*CheckpointInfo, error) {
	if config == nil || !config.Enabled {
		return &CheckpointInfo{Enabled: false}, nil
	}

	if config.CheckpointRef != nil && *config.CheckpointRef != "" {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
		if err := c.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      *config.CheckpointRef,
		}, ckpt); err != nil {
			return nil, fmt.Errorf("failed to get referenced checkpoint %s: %w", *config.CheckpointRef, err)
		}

		return getCheckpointInfoFromCheckpoint(ckpt)
	}

	if config.Identity == nil {
		return nil, fmt.Errorf("checkpoint enabled but no checkpointRef or identity provided")
	}

	hash, err := ComputeIdentityHash(*config.Identity)
	if err != nil {
		return nil, fmt.Errorf("failed to compute identity hash: %w", err)
	}

	info := &CheckpointInfo{
		Enabled:  true,
		Identity: config.Identity,
		Hash:     hash,
	}

	existing, err := FindCheckpointByIdentityHash(ctx, c, namespace, hash)
	if err != nil {
		return nil, err
	}
	if existing == nil {
		return info, nil
	}

	foundInfo, err := getCheckpointInfoFromCheckpoint(existing)
	if err != nil {
		return nil, err
	}
	foundInfo.Identity = config.Identity
	return foundInfo, nil
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
