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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdRBACReconciler owns the cluster-wide planner and EPP service-account
// bindings required before workload resources are created.
type dgdRBACReconciler struct {
	config  *configv1alpha1.OperatorConfiguration
	manager rbacManager
}

func newDGDRBACReconciler(
	config *configv1alpha1.OperatorConfiguration,
	manager rbacManager,
) *dgdRBACReconciler {
	return &dgdRBACReconciler{config: config, manager: manager}
}

func (r *dgdRBACReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if r.config.Namespace.Restricted != "" {
		return nil
	}
	if r.manager == nil {
		return fmt.Errorf("RBAC manager not initialized in cluster-wide mode")
	}
	if r.config.RBAC.PlannerClusterRoleName == "" {
		return fmt.Errorf("planner ClusterRole name is required in cluster-wide mode")
	}

	logger := log.FromContext(ctx)
	if err := r.manager.EnsureServiceAccountWithRBAC(
		ctx,
		dgd.Namespace,
		consts.PlannerServiceAccountName,
		r.config.RBAC.PlannerClusterRoleName,
	); err != nil {
		logger.Error(err, "Failed to ensure planner RBAC")
		return fmt.Errorf("failed to ensure planner RBAC: %w", err)
	}

	if !dgd.HasEPPComponent() {
		return nil
	}
	if r.config.RBAC.EPPClusterRoleName == "" {
		return fmt.Errorf("EPP ClusterRole name is required in cluster-wide mode when EPP service is present")
	}
	if err := r.manager.EnsureServiceAccountWithRBAC(
		ctx,
		dgd.Namespace,
		consts.EPPServiceAccountName,
		r.config.RBAC.EPPClusterRoleName,
	); err != nil {
		logger.Error(err, "Failed to ensure EPP RBAC")
		return fmt.Errorf("failed to ensure EPP RBAC: %w", err)
	}

	return nil
}
