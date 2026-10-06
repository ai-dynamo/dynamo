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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
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
	} else if !epp.IsLegacyGoEPP(eppService.EPPConfig) {
		// Native Rust EPP: reconcile away any ConfigMap the operator
		// generated for a prior Go EPP configuration (in-place migration)
		// rather than leaving stale Go EPP config/labels behind until the DGD
		// itself is deleted. A ConfigMapRef-backed ConfigMap is never touched
		// here: it stays in the IsLegacyGoEPP branch above regardless of
		// whether this else-if runs, since IsLegacyGoEPP only depends on
		// eppConfig being non-nil.
		//
		// A legacy DGD's eppConfig.configMapRef can name a user-managed
		// ConfigMap that happens to collide with this deterministic name, so
		// deleteOwnedLegacyEPPConfigMap only deletes a ConfigMap this DGD
		// actually owns (see its doc comment) instead of deleting by name
		// alone.
		if err := r.deleteOwnedLegacyEPPConfigMap(ctx, dgd); err != nil {
			logger.Error(err, "Failed to delete legacy EPP ConfigMap")
			return fmt.Errorf("failed to delete legacy EPP ConfigMap: %w", err)
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
		istioAvailable, err = features.DetectIstioDestinationRuleAvailability(ctx, r.restConfig)
		if err != nil {
			return fmt.Errorf("detect Istio DestinationRule API availability: %w", err)
		}
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

// deleteOwnedLegacyEPPConfigMap deletes the ConfigMap at the operator's
// deterministic legacy-EPP name (<dgd>-epp-config) only if this DGD is its
// controller owner.
//
// eppConfig.configMapRef lets a user point a legacy DGD at a ConfigMap they
// manage themselves, and that name is an arbitrary user choice — nothing
// stops it from being the same <dgd>-epp-config name the operator would have
// generated. Deleting by name alone (as the generic commoncontroller.SyncResource
// helper does) would delete that foreign, user-owned ConfigMap the moment
// such a DGD migrates off Go EPP by clearing eppConfig. Checking
// metav1.IsControlledBy first ensures only a ConfigMap this operator actually
// generated for this DGD is ever removed; a same-named but foreign ConfigMap
// is left untouched.
//
// An already-absent ConfigMap is success, not an error. If the ConfigMap is
// replaced by an unowned object of the same name in between, the delete
// fails with a conflict and the reconcile retries against the replacement
// instead of removing a ConfigMap this DGD does not own.
func (r *dgdEPPReconciler) deleteOwnedLegacyEPPConfigMap(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	name := epp.GetConfigMapName(dgd.Name)
	existing := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{Name: name, Namespace: dgd.Namespace}, existing)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get ConfigMap %s: %w", name, err)
	}
	if !metav1.IsControlledBy(existing, dgd) {
		return nil
	}

	err = r.Delete(ctx, existing, client.Preconditions{UID: &existing.UID})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to delete ConfigMap %s: %w", name, err)
	}
	return nil
}
