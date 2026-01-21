/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package secret

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// SecretReplicator handles replication of secrets across namespaces
type SecretReplicator struct {
	client.Client
	sourceNamespace string
	secretName      string
}

// NewSecretReplicator creates a new SecretReplicator for replicating a specific secret
func NewSecretReplicator(client client.Client, sourceNamespace, secretName string) *SecretReplicator {
	return &SecretReplicator{
		Client:          client,
		sourceNamespace: sourceNamespace,
		secretName:      secretName,
	}
}

// Replicate ensures the secret exists in the target namespace by copying from source namespace
func (r *SecretReplicator) Replicate(ctx context.Context, targetNamespace string) error {
	// Check if secret already exists in target namespace
	targetSecret := &corev1.Secret{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      r.secretName,
		Namespace: targetNamespace,
	}, targetSecret)

	if err == nil {
		// Secret already exists - do nothing
		return nil
	}

	if !k8serrors.IsNotFound(err) {
		return fmt.Errorf("failed to check target secret: %w", err)
	}

	// Get source secret
	sourceSecret := &corev1.Secret{}
	err = r.Get(ctx, types.NamespacedName{
		Name:      r.secretName,
		Namespace: r.sourceNamespace,
	}, sourceSecret)

	if err != nil {
		return fmt.Errorf("error getting source secret: %w", err)
	}

	// Create replica secret
	replicaSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      r.secretName,
			Namespace: targetNamespace,
		},
		Type: sourceSecret.Type,
		Data: sourceSecret.Data,
	}

	// Create the replica
	err = r.Create(ctx, replicaSecret)
	if err != nil && !k8serrors.IsAlreadyExists(err) {
		return fmt.Errorf("failed to create replica: %w", err)
	}

	return nil
}
