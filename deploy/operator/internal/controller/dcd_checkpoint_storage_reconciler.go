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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// dcdCheckpointStorageReconciler owns the checkpoint storage resources needed
// by a DynamoComponentDeployment. Keeping this convergence step outside the
// workload renderer makes rendering read-only and prevents multinode rendering
// from repeating the same write for both leader and worker templates.
type dcdCheckpointStorageReconciler struct {
	client  client.Client
	storage configv1alpha1.CheckpointStorageConfiguration
	gate    features.Gate
}

func newDCDCheckpointStorageReconciler(
	kubeClient client.Client,
	storage configv1alpha1.CheckpointStorageConfiguration,
	gate features.Gate,
) *dcdCheckpointStorageReconciler {
	return &dcdCheckpointStorageReconciler{
		client:  kubeClient,
		storage: storage,
		gate:    gate,
	}
}

func (r *dcdCheckpointStorageReconciler) Reconcile(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	if dcd == nil {
		return fmt.Errorf("dynamo component deployment is required")
	}
	if r.gate == nil {
		return fmt.Errorf("feature gate is required")
	}
	if !r.gate.Enabled(features.Checkpoint) {
		return nil
	}
	if dynamo.GetCheckpoint(&dcd.Spec.DynamoComponentDeploymentSharedSpec) == nil {
		return nil
	}

	return checkpoint.EnsureStoragePVC(ctx, r.client, dcd.Namespace, r.storage)
}
