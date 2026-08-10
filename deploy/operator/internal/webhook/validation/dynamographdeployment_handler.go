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

package validation

import (
	"context"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	authenticationv1 "k8s.io/api/authentication/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoGraphDeploymentWebhookName is the name of the validating webhook handler for DynamoGraphDeployment.
	DynamoGraphDeploymentWebhookName = "dynamographdeployment-validating-webhook"
	dynamoGraphDeploymentWebhookPath = "/validate/nvidia.com/v1beta1/dynamographdeployments"
)

// DynamoGraphDeploymentHandler is a handler for validating DynamoGraphDeployment resources.
// It is a thin wrapper around DynamoGraphDeploymentValidator.
type DynamoGraphDeploymentHandler struct {
	mgr               manager.Manager
	operatorPrincipal string
}

// NewDynamoGraphDeploymentHandler creates a new handler for DynamoGraphDeployment Webhook.
// mgr must not be nil.
// operatorPrincipal is the full Kubernetes SA username of the operator, used to authorize
// legacy workload-provider materialization and replica changes on scaling-adapter-enabled components (#7656).
func NewDynamoGraphDeploymentHandler(mgr manager.Manager, operatorPrincipal string) *DynamoGraphDeploymentHandler {
	return &DynamoGraphDeploymentHandler{
		mgr:               mgr,
		operatorPrincipal: operatorPrincipal,
	}
}

// ValidateCreate validates a DynamoGraphDeployment create request.
func (h *DynamoGraphDeploymentHandler) ValidateCreate(ctx context.Context, obj *nvidiacomv1beta1.DynamoGraphDeployment) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentGVK); err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", obj.Name, "namespace", obj.Namespace)

	// Create validator with manager for API group detection and perform validation
	validator := NewDynamoGraphDeploymentValidator(h.mgr)
	return validator.Validate(
		ctx,
		obj,
		runtimeVersionValidationSourceForRequest(ctx, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
	)
}

// ValidateUpdate validates a DynamoGraphDeployment update request.
func (h *DynamoGraphDeploymentHandler) ValidateUpdate(
	ctx context.Context,
	oldObj, newObj *nvidiacomv1beta1.DynamoGraphDeployment,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentGVK); err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newObj.Name, "namespace", newObj.Namespace)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newObj.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newObj.Name)
		return nil, nil
	}

	// Create validator with manager for API group detection and perform validation.
	validator := NewDynamoGraphDeploymentValidator(h.mgr)
	runtimeVersionSource := runtimeVersionValidationSourceForRequest(ctx, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
	warnings, err := validator.validate(
		ctx,
		newObj,
		runtimeVersionSourceDisabled,
		runtimeVersionSource,
	)
	if err != nil {
		return warnings, err
	}

	// Get user info from admission request context for identity-based validation
	var userInfo *authenticationv1.UserInfo
	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, replica changes for DGDSA-enabled services will be rejected")
		// userInfo remains nil, so scaling-adapter replica validation fails closed.
	} else {
		userInfo = &req.UserInfo
	}

	// Validate stateful rules (immutability + replicas protection)
	updateWarnings, err := validator.ValidateUpdate(
		ctx,
		oldObj,
		newObj,
		userInfo,
		h.operatorPrincipal,
		runtimeVersionSource,
	)
	if err != nil {
		username := "<unknown>"
		if userInfo != nil {
			username = userInfo.Username
		}
		logger.Info("validation failed", "error", err.Error(), "user", username)
		return updateWarnings, err
	}

	// Combine warnings
	warnings = append(warnings, updateWarnings...)
	return warnings, nil
}

// ValidateDelete validates a DynamoGraphDeployment delete request.
func (h *DynamoGraphDeploymentHandler) ValidateDelete(ctx context.Context, obj *nvidiacomv1beta1.DynamoGraphDeployment) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentGVK); err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", obj.Name, "namespace", obj.Namespace)

	// No special validation needed for deletion
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic
// and ObservedValidator to add metrics collection.
func (h *DynamoGraphDeploymentHandler) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	registerValidationWebhook(
		mgr,
		dynamoGraphDeploymentWebhookPath,
		h,
		consts.ResourceTypeDynamoGraphDeployment,
		gate,
	)
	return nil
}
