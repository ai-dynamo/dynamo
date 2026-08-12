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
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdDiscoveryReconciler owns the service account, role, and role binding used
// by Kubernetes discovery and checkpoint jobs.
type dgdDiscoveryReconciler struct {
	dgdResourceSyncer
	config *configv1alpha1.OperatorConfiguration
}

func newDGDDiscoveryReconciler(
	syncer dgdResourceSyncer,
	config *configv1alpha1.OperatorConfiguration,
) *dgdDiscoveryReconciler {
	return &dgdDiscoveryReconciler{dgdResourceSyncer: syncer, config: config}
}

func (r *dgdDiscoveryReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)
	if !commoncontroller.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, dgd.Annotations) {
		logger.Info("K8s discovery is not enabled")
		return nil
	}
	logger.Info("K8s discovery is enabled")

	serviceAccount := discovery.GetK8sDiscoveryServiceAccount(dgd.Name, dgd.Namespace)
	if _, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*corev1.ServiceAccount, bool, error) {
		return serviceAccount, false, nil
	}); err != nil {
		logger.Error(err, "failed to sync the k8s discovery service account")
		return fmt.Errorf("failed to sync the k8s discovery service account: %w", err)
	}

	role := discovery.GetK8sDiscoveryRole(dgd.Name, dgd.Namespace)
	if _, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*rbacv1.Role, bool, error) {
		return role, false, nil
	}); err != nil {
		logger.Error(err, "failed to sync the k8s discovery role")
		return fmt.Errorf("failed to sync the k8s discovery role: %w", err)
	}

	roleBinding := discovery.GetK8sDiscoveryRoleBinding(dgd.Name, dgd.Namespace)
	if _, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*rbacv1.RoleBinding, bool, error) {
		return roleBinding, false, nil
	}); err != nil {
		logger.Error(err, "failed to sync the k8s discovery role binding")
		return fmt.Errorf("failed to sync the k8s discovery role binding: %w", err)
	}

	return nil
}
