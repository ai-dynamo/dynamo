/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package setup

import (
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	webhookmutation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/mutation"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/validation"
	ctrl "sigs.k8s.io/controller-runtime"
)

type Options struct {
	Config            *configv1alpha1.OperatorConfiguration
	RuntimeConfig     *commoncontroller.RuntimeConfig
	OperatorVersion   string
	OperatorPrincipal string
	Gate              features.Gate
}

func SetupAll(mgr ctrl.Manager, opts Options) error {
	if opts.Config == nil {
		return fmt.Errorf("operator configuration is required")
	}
	if opts.RuntimeConfig == nil {
		return fmt.Errorf("runtime configuration is required")
	}
	cfg := opts.Config
	runtimeConfig := opts.RuntimeConfig
	gate := opts.Gate
	if gate == nil {
		gate = runtimeConfig.Gate
	}

	isClusterWide := cfg.Namespace.Restricted == ""
	if isClusterWide {
		internalwebhook.SetExcludedNamespaces(runtimeConfig.ExcludedNamespaces)
	} else {
		internalwebhook.SetExcludedNamespaces(nil)
	}

	dcdHandler := webhookvalidation.NewDynamoComponentDeploymentHandler()
	if err := dcdHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoComponentDeployment webhook: %w", err)
	}

	dgdHandler := webhookvalidation.NewDynamoGraphDeploymentHandler(mgr, opts.OperatorPrincipal)
	if err := dgdHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeployment webhook: %w", err)
	}

	dckptHandler := webhookvalidation.NewDynamoCheckpointHandler()
	if err := dckptHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoCheckpoint webhook: %w", err)
	}

	dmHandler := webhookvalidation.NewDynamoModelHandler()
	if err := dmHandler.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoModel webhook: %w", err)
	}

	dgdrHandler := webhookvalidation.NewDynamoGraphDeploymentRequestHandler()
	if err := dgdrHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentRequest webhook: %w", err)
	}

	if isClusterWide {
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
	}

	dcdDefaulter := webhookdefaulting.NewDCDDefaulter()
	if err := dcdDefaulter.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoComponentDeployment defaulting webhook: %w", err)
	}

	dgdDefaulter := webhookdefaulting.NewDGDDefaulter(opts.OperatorVersion)
	if err := dgdDefaulter.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeployment defaulting webhook: %w", err)
	}

	dgdrDefaulter := webhookdefaulting.NewDGDRDefaulter(opts.OperatorVersion)
	if err := dgdrDefaulter.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentRequest defaulting webhook: %w", err)
	}

	podCheckpointRestoreMutator := webhookmutation.NewPodCheckpointRestoreMutator(mgr.GetClient(), cfg)
	if err := podCheckpointRestoreMutator.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register Pod checkpoint restore mutating webhook: %w", err)
	}
	return nil
}
