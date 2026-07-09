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

package webhooks

import (
	"fmt"
	"os"

	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"

	configapi "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	webhookmutation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/mutation"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/validation"
)

var log = ctrl.Log.WithName("webhooks")

const (
	dynamoComponentDeploymentWebhook        = "DynamoComponentDeployment"
	dynamoGraphDeploymentWebhook            = "DynamoGraphDeployment"
	dynamoGraphDeploymentRequestWebhook     = "DynamoGraphDeploymentRequest"
	dynamoGraphDeploymentScalingAdapterHook = "DynamoGraphDeploymentScalingAdapter"
	dynamoCheckpointWebhook                 = "DynamoCheckpoint"
	dynamoModelWebhook                      = "DynamoModel"
	podCheckpointRestoreWebhook             = "PodCheckpointRestore"
)

// Setup sets up the webhooks for core controllers. It returns the name of the
// webhook that failed to create and an error, if any.
func Setup(
	mgr ctrl.Manager,
	cfg *configapi.OperatorConfiguration,
	runtimeConfig *commonController.RuntimeConfig,
	operatorVersion string,
) (string, error) {
	isClusterWide := cfg.Namespace.Restricted == ""
	if isClusterWide {
		log.Info("Configuring webhooks with lease-based namespace exclusion for cluster-wide mode")
		internalwebhook.SetExcludedNamespaces(runtimeConfig.ExcludedNamespaces)
	} else {
		log.Info("Configuring webhooks for namespace-restricted mode (no lease checking)",
			"restrictedNamespace", cfg.Namespace.Restricted)
		internalwebhook.SetExcludedNamespaces(nil)
	}

	var operatorPrincipal string
	if sa, ns := os.Getenv("POD_SERVICE_ACCOUNT"), os.Getenv("POD_NAMESPACE"); sa != "" && ns != "" {
		operatorPrincipal = fmt.Sprintf("system:serviceaccount:%s:%s", ns, sa)
		log.Info("Detected operator principal from downward API", "principal", operatorPrincipal)
	} else {
		log.Info("POD_SERVICE_ACCOUNT/POD_NAMESPACE not set; operator SA self-identification disabled")
	}

	// Temporary internal gate for GMS + Snapshot.
	if os.Getenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar) == "1" {
		log.Info(
			"INTERNAL OVERRIDE: GMS + Snapshot admission rule disabled via env var; do NOT enable in production",
			"envVar", consts.DynamoOperatorAllowGMSSnapshotEnvVar,
		)
	}

	log.Info("Registering validation webhooks")

	dcdHandler := webhookvalidation.NewDynamoComponentDeploymentHandler()
	if err := dcdHandler.RegisterWithManager(mgr); err != nil {
		return dynamoComponentDeploymentWebhook, err
	}

	dgdHandler := webhookvalidation.NewDynamoGraphDeploymentHandler(mgr, operatorPrincipal, runtimeConfig.GroveEnabled)
	if err := dgdHandler.RegisterWithManager(mgr); err != nil {
		return dynamoGraphDeploymentWebhook, err
	}

	dckptHandler := webhookvalidation.NewDynamoCheckpointHandler()
	if err := dckptHandler.RegisterWithManager(mgr); err != nil {
		return dynamoCheckpointWebhook, err
	}

	dmHandler := webhookvalidation.NewDynamoModelHandler()
	if err := dmHandler.RegisterWithManager(mgr); err != nil {
		return dynamoModelWebhook, err
	}

	dgdrHandler := webhookvalidation.NewDynamoGraphDeploymentRequestHandler(
		isClusterWide, ptr.Deref(cfg.GPU.DiscoveryEnabled, true),
	)
	if err := dgdrHandler.RegisterWithManager(mgr); err != nil {
		return dynamoGraphDeploymentRequestWebhook, err
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}).
		Complete(); err != nil {
		return dynamoGraphDeploymentRequestWebhook, err
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeployment{}).
		Complete(); err != nil {
		return dynamoGraphDeploymentWebhook, err
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoComponentDeployment{}).
		Complete(); err != nil {
		return dynamoComponentDeploymentWebhook, err
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}).
		Complete(); err != nil {
		return dynamoGraphDeploymentScalingAdapterHook, err
	}

	log.Info("Registering defaulting webhooks")

	dcdDefaulter := webhookdefaulting.NewDCDDefaulter()
	if err := dcdDefaulter.RegisterWithManager(mgr); err != nil {
		return dynamoComponentDeploymentWebhook, err
	}

	dgdDefaulter := webhookdefaulting.NewDGDDefaulter(operatorVersion, runtimeConfig.GroveEnabled)
	if err := dgdDefaulter.RegisterWithManager(mgr); err != nil {
		return dynamoGraphDeploymentWebhook, err
	}

	dgdrDefaulter := webhookdefaulting.NewDGDRDefaulter(operatorVersion)
	if err := dgdrDefaulter.RegisterWithManager(mgr); err != nil {
		return dynamoGraphDeploymentRequestWebhook, err
	}

	log.Info("Registering mutation webhooks")

	podCheckpointRestoreMutator := webhookmutation.NewPodCheckpointRestoreMutator(mgr.GetClient(), cfg)
	if err := podCheckpointRestoreMutator.RegisterWithManager(mgr); err != nil {
		return podCheckpointRestoreWebhook, err
	}

	return "", nil
}
