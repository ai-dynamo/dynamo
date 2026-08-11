/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func registerValidationWebhook[T client.Object](
	mgr manager.Manager,
	path string,
	validator admission.Validator[T],
	resourceType string,
	gate features.Gate,
) {
	leaseAwareValidator := internalwebhook.NewLeaseAwareValidator(
		validator,
		internalwebhook.GetExcludedNamespaces(),
	)
	observedValidator := observability.NewObservedValidator(leaseAwareValidator, resourceType)
	webhook := admission.WithValidator(mgr.GetScheme(), observedValidator).WithRecoverPanic(true)
	if gate != nil {
		webhook = internalwebhook.WithGate(webhook, gate)
	}
	mgr.GetWebhookServer().Register(path, webhook)
}
