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
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoComponentDeploymentWebhookName is the name of the validating webhook handler for DynamoComponentDeployment.
	DynamoComponentDeploymentWebhookName        = "dynamocomponentdeployment-validating-webhook"
	dynamoComponentDeploymentV1Beta1WebhookPath = "/validate/nvidia.com/v1beta1/dynamocomponentdeployments"
)

// DynamoComponentDeploymentHandler is a handler for validating DynamoComponentDeployment resources.
// It is a thin wrapper around DynamoComponentDeploymentValidator.
type DynamoComponentDeploymentHandler struct{}

// NewDynamoComponentDeploymentHandler creates a new handler for DynamoComponentDeployment Webhook.
func NewDynamoComponentDeploymentHandler() *DynamoComponentDeploymentHandler {
	return &DynamoComponentDeploymentHandler{}
}

// ValidateCreate validates a DynamoComponentDeployment create request.
func (h *DynamoComponentDeploymentHandler) ValidateCreate(ctx context.Context, obj *nvidiacomv1beta1.DynamoComponentDeployment) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoComponentDeploymentGVK); err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", obj.Name, "namespace", obj.Namespace)

	validator := NewDynamoComponentDeploymentValidator()
	return validator.validate(
		ctx,
		obj,
		runtimeVersionValidationSourceForRequest(ctx, nvidiacomv1beta1.DynamoComponentDeploymentGVK),
	)
}

// ValidateUpdate validates a DynamoComponentDeployment update request.
func (h *DynamoComponentDeploymentHandler) ValidateUpdate(
	ctx context.Context,
	oldObj, newObj *nvidiacomv1beta1.DynamoComponentDeployment,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoComponentDeploymentGVK); err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newObj.Name, "namespace", newObj.Namespace)

	// Skip validation if the resource is being deleted to allow finalizer removal.
	if !newObj.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newObj.Name)
		return nil, nil
	}

	validator := NewDynamoComponentDeploymentValidator()
	return validator.ValidateUpdate(
		ctx,
		oldObj,
		newObj,
		runtimeVersionValidationSourceForRequest(ctx, nvidiacomv1beta1.DynamoComponentDeploymentGVK),
	)
}

// ValidateDelete validates a DynamoComponentDeployment delete request.
func (h *DynamoComponentDeploymentHandler) ValidateDelete(ctx context.Context, obj *nvidiacomv1beta1.DynamoComponentDeployment) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoComponentDeploymentGVK); err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", obj.Name, "namespace", obj.Namespace)
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic
// and ObservedValidator to add metrics collection.
func (h *DynamoComponentDeploymentHandler) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	registerValidationWebhook(
		mgr,
		dynamoComponentDeploymentV1Beta1WebhookPath,
		h,
		consts.ResourceTypeDynamoComponentDeployment,
		gate,
	)
	return nil
}
