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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func checkpointIdentityHash(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (string, error) {
	if ckpt.Status.IdentityHash != "" {
		return ckpt.Status.IdentityHash, nil
	}

	hash, err := ComputeIdentityHash(ckpt.Spec.Identity)
	if err != nil {
		return "", fmt.Errorf("failed to compute checkpoint hash for %s: %w", ckpt.Name, err)
	}

	return hash, nil
}

func FindCheckpointByIdentityHash(
	ctx context.Context,
	c client.Client,
	namespace string,
	hash string,
	excludeName string,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	checkpoints := &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(
		ctx,
		checkpoints,
		client.InNamespace(namespace),
		client.MatchingLabels{consts.KubeLabelCheckpointID: hash},
	); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints by hash label: %w", err)
	}

	var existing *nvidiacomv1alpha1.DynamoCheckpoint
	for i := range checkpoints.Items {
		if checkpoints.Items[i].Name == excludeName {
			continue
		}
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for identity hash %s", hash)
		}
		existing = checkpoints.Items[i].DeepCopy()
	}
	if existing != nil {
		return existing, nil
	}

	// Fall back to a full scan so legacy checkpoints without the hash label still resolve.
	checkpoints = &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(ctx, checkpoints, client.InNamespace(namespace)); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints: %w", err)
	}

	for i := range checkpoints.Items {
		ckpt := &checkpoints.Items[i]
		if ckpt.Name == excludeName {
			continue
		}
		existingHash, err := checkpointIdentityHash(ckpt)
		if err != nil {
			return nil, err
		}
		if existingHash != hash {
			continue
		}
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for identity hash %s", hash)
		}
		existing = ckpt.DeepCopy()
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

	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("checkpoint-%s", hash),
			Namespace: namespace,
			Labels: map[string]string{
				consts.KubeLabelCheckpointID: hash,
			},
			Annotations: map[string]string{
				snapshotprotocol.CheckpointArtifactVersionAnnotation: snapshotprotocol.DefaultCheckpointArtifactVersion,
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
			return nil, fmt.Errorf("failed to create checkpoint %s: %w", ckpt.Name, err)
		}
		existing := &nvidiacomv1alpha1.DynamoCheckpoint{}
		key := types.NamespacedName{Name: ckpt.Name, Namespace: namespace}
		if err := c.Get(ctx, key, existing); err != nil {
			return nil, fmt.Errorf("failed to get checkpoint %s after already exists: %w", ckpt.Name, err)
		}

		existingHash, err := checkpointIdentityHash(existing)
		if err != nil {
			return nil, err
		}
		if existingHash != hash {
			return nil, fmt.Errorf("checkpoint %s already exists with identity hash %s", ckpt.Name, existingHash)
		}

		return existing, nil
	}

	existing, err := FindCheckpointByIdentityHash(ctx, c, namespace, hash, ckpt.Name)
	if err != nil {
		if deleteErr := c.Delete(ctx, ckpt); deleteErr != nil && !apierrors.IsNotFound(deleteErr) {
			return nil, fmt.Errorf("failed to clean up checkpoint %s after dedupe error: %v (lookup error: %w)", ckpt.Name, deleteErr, err)
		}
		return nil, err
	}
	if existing != nil {
		if err := c.Delete(ctx, ckpt); err != nil && !apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("failed to delete duplicate checkpoint %s: %w", ckpt.Name, err)
		}
		return existing, nil
	}

	return ckpt, nil
}
