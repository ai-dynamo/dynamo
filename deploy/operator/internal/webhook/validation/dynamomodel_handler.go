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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoModelWebhookName is the name of the validating webhook handler for DynamoModel.
	DynamoModelWebhookName = "dynamomodel-validating-webhook"
	dynamoModelWebhookPath = "/validate-nvidia-com-v1alpha1-dynamomodel"
)

// DynamoModelHandler is a handler for validating DynamoModel resources.
// It is a thin wrapper around DynamoModelValidator.
type DynamoModelHandler struct{}

// NewDynamoModelHandler creates a new handler for DynamoModel Webhook.
func NewDynamoModelHandler() *DynamoModelHandler {
	return &DynamoModelHandler{}
}

// ValidateCreate validates a DynamoModel create request.
func (h *DynamoModelHandler) ValidateCreate(ctx context.Context, model *nvidiacomv1alpha1.DynamoModel) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoModelWebhookName)

	logger.Info("validate create", "name", model.Name, "namespace", model.Namespace)

	validator := NewDynamoModelValidator()
	return validator.Validate(model)
}

// ValidateUpdate validates a DynamoModel update request.
func (h *DynamoModelHandler) ValidateUpdate(ctx context.Context, oldModel, newModel *nvidiacomv1alpha1.DynamoModel) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoModelWebhookName)

	logger.Info("validate update", "name", newModel.Name, "namespace", newModel.Namespace)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newModel.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newModel.Name)
		return nil, nil
	}

	validator := NewDynamoModelValidator()
	return validator.ValidateUpdate(oldModel, newModel)
}

// ValidateDelete validates a DynamoModel delete request.
func (h *DynamoModelHandler) ValidateDelete(ctx context.Context, model *nvidiacomv1alpha1.DynamoModel) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoModelWebhookName)

	logger.Info("validate delete", "name", model.Name, "namespace", model.Namespace)

	// No special validation needed for deletion
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic.
func (h *DynamoModelHandler) RegisterWithManager(mgr manager.Manager) error {
	registerValidationWebhook(mgr, dynamoModelWebhookPath, h, consts.ResourceTypeDynamoModel, nil)
	return nil
}
