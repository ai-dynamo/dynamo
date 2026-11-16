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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
)

// dgdWaitForLeaderReconciler owns the wait-for-leader ConfigMap used by
// multinode vLLM mp workers.
type dgdWaitForLeaderReconciler struct {
	dgdResourceSyncer
}

func newDGDWaitForLeaderReconciler(syncer dgdResourceSyncer) *dgdWaitForLeaderReconciler {
	return &dgdWaitForLeaderReconciler{dgdResourceSyncer: syncer}
}

func (r *dgdWaitForLeaderReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if !dgd.HasAnyMultinodeComponent() {
		return nil
	}
	configMap := dynamo.GenerateWaitLeaderConfigMap(dgd.Name, dgd.Namespace)
	_, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*corev1.ConfigMap, bool, error) {
		return configMap, false, nil
	})
	return err
}
