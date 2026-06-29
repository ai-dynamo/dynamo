/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package registration

import (
	"fmt"
	"os"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	webhookmutation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/mutation"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/validation"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// RegisterHandlers registers the same validating, mutating, defaulting, and
// conversion webhook handlers that the operator serves in production.
func RegisterHandlers(
	mgr manager.Manager,
	operatorCfg *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	operatorVersion string,
) error {
	logger := ctrl.Log.WithName("setup").WithName("webhooks")

	isClusterWide := operatorCfg.Namespace.Restricted == ""
	if isClusterWide {
		logger.Info("Configuring webhooks with lease-based namespace exclusion for cluster-wide mode")
		internalwebhook.SetExcludedNamespaces(runtimeConfig.ExcludedNamespaces)
	} else {
		logger.Info("Configuring webhooks for namespace-restricted mode (no lease checking)",
			"restrictedNamespace", operatorCfg.Namespace.Restricted)
		internalwebhook.SetExcludedNamespaces(nil)
	}

	var operatorPrincipal string
	if sa, ns := os.Getenv("POD_SERVICE_ACCOUNT"), os.Getenv("POD_NAMESPACE"); sa != "" && ns != "" {
		operatorPrincipal = fmt.Sprintf("system:serviceaccount:%s:%s", ns, sa)
		logger.Info("Detected operator principal from downward API", "principal", operatorPrincipal)
	} else {
		logger.Info("POD_SERVICE_ACCOUNT/POD_NAMESPACE not set; operator SA self-identification disabled")
	}

	// Temporary internal gate for GMS + Snapshot.
	if os.Getenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar) == "1" {
		logger.Info(
			"INTERNAL OVERRIDE: GMS + Snapshot admission rule disabled via env var; do NOT enable in production",
			"envVar", consts.DynamoOperatorAllowGMSSnapshotEnvVar,
		)
	}

	logger.Info("Registering validation webhooks")

	dcdHandler := webhookvalidation.NewDynamoComponentDeploymentHandler()
	if err := dcdHandler.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoComponentDeployment webhook: %w", err)
	}

	dgdHandler := webhookvalidation.NewDynamoGraphDeploymentHandler(mgr, operatorPrincipal, runtimeConfig.GroveEnabled)
	if err := dgdHandler.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeployment webhook: %w", err)
	}

	dckptHandler := webhookvalidation.NewDynamoCheckpointHandler()
	if err := dckptHandler.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoCheckpoint webhook: %w", err)
	}

	dmHandler := webhookvalidation.NewDynamoModelHandler()
	if err := dmHandler.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoModel webhook: %w", err)
	}

	dgdrHandler := webhookvalidation.NewDynamoGraphDeploymentRequestHandler(
		isClusterWide, ptr.Deref(operatorCfg.GPU.DiscoveryEnabled, true),
	)
	if err := dgdrHandler.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentRequest webhook: %w", err)
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}).
		Complete(); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentRequest conversion webhook: %w", err)
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeployment{}).
		Complete(); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeployment conversion webhook: %w", err)
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoComponentDeployment{}).
		Complete(); err != nil {
		return fmt.Errorf("unable to register DynamoComponentDeployment conversion webhook: %w", err)
	}

	if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}).
		Complete(); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentScalingAdapter conversion webhook: %w", err)
	}

	logger.Info("Registering defaulting webhooks")

	dcdDefaulter := webhookdefaulting.NewDCDDefaulter()
	if err := dcdDefaulter.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoComponentDeployment defaulting webhook: %w", err)
	}

	dgdDefaulter := webhookdefaulting.NewDGDDefaulter(operatorVersion, runtimeConfig.GroveEnabled)
	if err := dgdDefaulter.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeployment defaulting webhook: %w", err)
	}

	dgdrDefaulter := webhookdefaulting.NewDGDRDefaulter(operatorVersion)
	if err := dgdrDefaulter.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentRequest defaulting webhook: %w", err)
	}

	logger.Info("Registering mutation webhooks")

	podCheckpointRestoreMutator := webhookmutation.NewPodCheckpointRestoreMutator(mgr.GetClient(), operatorCfg)
	if err := podCheckpointRestoreMutator.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register Pod checkpoint restore mutating webhook: %w", err)
	}

	logger.Info("Webhooks registered successfully")
	return nil
}
