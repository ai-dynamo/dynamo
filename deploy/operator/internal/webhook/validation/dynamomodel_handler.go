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
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"k8s.io/apimachinery/pkg/runtime"
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
func (h *DynamoModelHandler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoModelWebhookName)

	model, err := castToDynamoModel(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", model.Name, "namespace", model.Namespace)

	// Create validator and perform validation
	validator := NewDynamoModelValidator(model)
	return validator.Validate()
}

// ValidateUpdate validates a DynamoModel update request.
func (h *DynamoModelHandler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoModelWebhookName)

	newModel, err := castToDynamoModel(newObj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newModel.Name, "namespace", newModel.Namespace)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newModel.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newModel.Name)
		return nil, nil
	}

	oldModel, err := castToDynamoModel(oldObj)
	if err != nil {
		return nil, err
	}

	// Create validator and perform validation
	validator := NewDynamoModelValidator(newModel)

	// Validate stateless rules
	warnings, err := validator.Validate()
	if err != nil {
		return warnings, err
	}

	// Validate stateful rules (immutability)
	updateWarnings, err := validator.ValidateUpdate(oldModel)
	if err != nil {
		return updateWarnings, err
	}

	// Combine warnings
	warnings = append(warnings, updateWarnings...)
	return warnings, nil
}

// ValidateDelete validates a DynamoModel delete request.
func (h *DynamoModelHandler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoModelWebhookName)

	model, err := castToDynamoModel(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", model.Name, "namespace", model.Namespace)

	// No special validation needed for deletion
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic.
func (h *DynamoModelHandler) RegisterWithManager(mgr manager.Manager) error {
	// Wrap the handler with lease-aware logic for cluster-wide coordination
	validator := internalwebhook.NewLeaseAwareValidator(h, internalwebhook.GetExcludedNamespaces())

	webhook := admission.
		WithCustomValidator(mgr.GetScheme(), &nvidiacomv1alpha1.DynamoModel{}, validator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dynamoModelWebhookPath, webhook)
	return nil
}

// castToDynamoModel attempts to cast a runtime.Object to a DynamoModel.
func castToDynamoModel(obj runtime.Object) (*nvidiacomv1alpha1.DynamoModel, error) {
	model, ok := obj.(*nvidiacomv1alpha1.DynamoModel)
	if !ok {
		return nil, fmt.Errorf("expected DynamoModel but got %T", obj)
	}
	return model, nil
}
