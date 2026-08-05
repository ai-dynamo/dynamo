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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/log"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// dgdEPPReconciler owns the EPP ConfigMap, InferencePool, and optional service
// mesh resources associated with the DGD's EPP component.
type dgdEPPReconciler struct {
	dgdResourceSyncer
	config        *configv1alpha1.OperatorConfiguration
	runtimeConfig *commoncontroller.RuntimeConfig
	restConfig    *rest.Config
}

func newDGDEPPReconciler(
	syncer dgdResourceSyncer,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	restConfig *rest.Config,
) *dgdEPPReconciler {
	return &dgdEPPReconciler{
		dgdResourceSyncer: syncer,
		config:            config,
		runtimeConfig:     runtimeConfig,
		restConfig:        restConfig,
	}
}

func (r *dgdEPPReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)
	componentName, eppService, hasEPP := dgd.GetEPPComponent()
	if !hasEPP {
		logger.V(1).Info("No EPP service defined, skipping EPP resource reconciliation")
		return nil
	}

	logger.Info("Reconciling EPP resources", "componentName", componentName)

	// Legacy Go EPP: reconcile the ConfigMap when eppConfig is set and not a
	// user-managed ConfigMapRef. Native Rust EPP needs no ConfigMap.
	if epp.IsLegacyGoEPP(eppService.EPPConfig) && eppService.EPPConfig.ConfigMapRef == nil {
		configMap, err := epp.GenerateConfigMap(ctx, dgd, componentName, eppService.EPPConfig)
		if err != nil {
			logger.Error(err, "Failed to generate EPP ConfigMap")
			return fmt.Errorf("failed to generate EPP ConfigMap: %w", err)
		}
		if configMap != nil {
			if _, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*corev1.ConfigMap, bool, error) {
				return configMap, false, nil
			}); err != nil {
				logger.Error(err, "Failed to sync EPP ConfigMap")
				return fmt.Errorf("failed to sync EPP ConfigMap: %w", err)
			}
		}
	}

	eppServiceName := dynamo.GetDCDResourceName(dgd, componentName, "")
	inferencePool, err := epp.GenerateInferencePool(dgd, componentName, eppServiceName)
	if err != nil {
		logger.Error(err, "Failed to generate EPP InferencePool")
		return fmt.Errorf("failed to generate EPP InferencePool: %w", err)
	}
	if _, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*gaiev1.InferencePool, bool, error) {
		return inferencePool, false, nil
	}); err != nil {
		logger.Error(err, "Failed to sync EPP InferencePool")
		return fmt.Errorf("failed to sync EPP InferencePool: %w", err)
	}

	meshEnabled := r.runtimeConfig.Gate.Enabled(features.Istio)
	istioAvailable := meshEnabled
	if !meshEnabled {
		istioAvailable = features.DetectIstioDestinationRuleAvailability(ctx, r.restConfig)
	}
	if istioAvailable {
		destinationRule := dynamo.GenerateEPPDestinationRule(eppServiceName, dgd.Namespace, r.config.ServiceMesh)
		if _, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*networkingv1beta1.DestinationRule, bool, error) {
			return destinationRule, !meshEnabled, nil
		}); err != nil {
			logger.Error(err, "Failed to sync EPP DestinationRule")
			return fmt.Errorf("failed to sync EPP DestinationRule: %w", err)
		}
		if meshEnabled {
			logger.Info("Synced EPP DestinationRule", "name", eppServiceName)
		} else {
			logger.Info("Cleaned up EPP DestinationRule", "name", eppServiceName)
		}
	}

	logger.Info("Successfully reconciled EPP resources", "poolName", inferencePool.GetName())
	return nil
}
