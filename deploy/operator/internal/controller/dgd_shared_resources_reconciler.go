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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdSharedResourcesReconciler preserves the dependency-ordered sequence of
// DGD-owned resources that both workload programs require before realizing
// their workload graphs.
type dgdSharedResourcesReconciler struct {
	rbac          *dgdRBACReconciler
	pvcs          *dgdPVCReconciler
	discovery     *dgdDiscoveryReconciler
	gmsClaims     *dgdGMSResourceClaimsReconciler
	checkpoints   *dgdCheckpointsReconciler
	epp           *dgdEPPReconciler
	waitForLeader *dgdWaitForLeaderReconciler
	sshKeys       *dgdSSHKeysReconciler
}

func newDGDSharedResourcesReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	restConfig *rest.Config,
	dockerSecretRetriever DockerSecretRetriever,
	sshKeyManager *secret.SSHKeyManager,
	rbacManager rbacManager,
) *dgdSharedResourcesReconciler {
	syncer := newDGDResourceSyncer(kubeClient, recorder)
	return &dgdSharedResourcesReconciler{
		rbac:          newDGDRBACReconciler(config, rbacManager),
		pvcs:          newDGDPVCReconciler(syncer),
		discovery:     newDGDDiscoveryReconciler(syncer, config),
		gmsClaims:     newDGDGMSResourceClaimsReconciler(syncer, runtimeConfig.Gate),
		checkpoints:   newDGDCheckpointsReconciler(syncer, config, runtimeConfig, dockerSecretRetriever),
		epp:           newDGDEPPReconciler(syncer, config, runtimeConfig, restConfig),
		waitForLeader: newDGDWaitForLeaderReconciler(syncer),
		sshKeys:       newDGDSSHKeysReconciler(sshKeyManager),
	}
}

func (r *dgdSharedResourcesReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (dgdCheckpointsResult, error) {
	logger := log.FromContext(ctx)
	if err := r.rbac.Reconcile(ctx, dgd); err != nil {
		return dgdCheckpointsResult{}, err
	}
	if err := r.pvcs.Reconcile(ctx, dgd); err != nil {
		logger.Error(err, "Failed to reconcile top-level PVCs")
		return dgdCheckpointsResult{}, fmt.Errorf("failed to reconcile top-level PVCs: %w", err)
	}
	if err := r.discovery.Reconcile(ctx, dgd); err != nil {
		logger.Error(err, "Failed to reconcile K8s discovery resources")
		return dgdCheckpointsResult{}, fmt.Errorf("failed to reconcile K8s discovery resources: %w", err)
	}
	if err := r.gmsClaims.Reconcile(ctx, dgd); err != nil {
		return dgdCheckpointsResult{}, err
	}

	checkpoints, err := r.checkpoints.Reconcile(ctx, dgd)
	if err != nil {
		logger.Error(err, "Failed to reconcile checkpoints")
		return dgdCheckpointsResult{}, fmt.Errorf("failed to reconcile checkpoints: %w", err)
	}

	// Preserve checkpoint observations if a later shared-resource operation
	// fails; programs publish this meaningful partial status before returning.
	if err := r.epp.Reconcile(ctx, dgd); err != nil {
		logger.Error(err, "Failed to reconcile EPP resources")
		return checkpoints, fmt.Errorf("failed to reconcile EPP resources: %w", err)
	}
	if err := r.waitForLeader.Reconcile(ctx, dgd); err != nil {
		logger.Error(err, "Failed to reconcile wait-leader ConfigMap")
		return checkpoints, fmt.Errorf("failed to reconcile wait-leader ConfigMap: %w", err)
	}
	if err := r.sshKeys.Reconcile(ctx, dgd); err != nil {
		return checkpoints, err
	}

	return checkpoints, nil
}
