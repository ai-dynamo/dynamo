/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"context"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdSSHKeysReconciler owns the namespace-scoped MPI SSH key material needed
// by multinode component workloads.
type dgdSSHKeysReconciler struct {
	manager *secret.SSHKeyManager
}

func newDGDSSHKeysReconciler(manager *secret.SSHKeyManager) *dgdSSHKeysReconciler {
	return &dgdSSHKeysReconciler{manager: manager}
}

func (r *dgdSSHKeysReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if r.manager == nil || !dgd.HasAnyMultinodeComponent() {
		return nil
	}
	if err := r.manager.EnsureAndReplicate(ctx, dgd.Namespace); err != nil {
		log.FromContext(ctx).Error(err, "Failed to ensure MPI SSH key secret", "namespace", dgd.Namespace)
		return fmt.Errorf("failed to ensure MPI SSH key secret: %w", err)
	}
	return nil
}
