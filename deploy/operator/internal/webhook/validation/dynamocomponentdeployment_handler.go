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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoComponentDeploymentWebhookName is the name of the validating webhook handler for DynamoComponentDeployment.
	DynamoComponentDeploymentWebhookName = "dynamocomponentdeployment-validating-webhook"
	dynamoComponentDeploymentWebhookPath = "/validate-nvidia-com-v1alpha1-dynamocomponentdeployment"
)

// DynamoComponentDeploymentHandler is a handler for validating DynamoComponentDeployment resources.
// It is a thin wrapper around DynamoComponentDeploymentValidator.
type DynamoComponentDeploymentHandler struct{}

// NewDynamoComponentDeploymentHandler creates a new handler for DynamoComponentDeployment Webhook.
func NewDynamoComponentDeploymentHandler() *DynamoComponentDeploymentHandler {
	return &DynamoComponentDeploymentHandler{}
}

// ValidateCreate validates a DynamoComponentDeployment create request.
func (h *DynamoComponentDeploymentHandler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	deployment, err := castToDynamoComponentDeployment(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", deployment.Name, "namespace", deployment.Namespace)

	// Create validator and perform validation
	validator := NewDynamoComponentDeploymentValidator(deployment)
	return validator.Validate(ctx)
}

// ValidateUpdate validates a DynamoComponentDeployment update request.
func (h *DynamoComponentDeploymentHandler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	newDeployment, err := castToDynamoComponentDeployment(newObj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newDeployment.Name, "namespace", newDeployment.Namespace)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newDeployment.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newDeployment.Name)
		return nil, nil
	}

	oldDeployment, err := castToDynamoComponentDeployment(oldObj)
	if err != nil {
		return nil, err
	}

	// Create validator and perform validation
	validator := NewDynamoComponentDeploymentValidator(newDeployment)

	// Validate stateless rules
	warnings, err := validator.Validate(ctx)
	if err != nil {
		return warnings, err
	}

	// Validate stateful rules (immutability)
	updateWarnings, err := validator.ValidateUpdate(oldDeployment)
	if err != nil {
		return updateWarnings, err
	}

	// Combine warnings
	warnings = append(warnings, updateWarnings...)
	return warnings, nil
}

// ValidateDelete validates a DynamoComponentDeployment delete request.
func (h *DynamoComponentDeploymentHandler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	deployment, err := castToDynamoComponentDeployment(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", deployment.Name, "namespace", deployment.Namespace)

	// No special validation needed for deletion
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic
// and ObservedValidator to add metrics collection.
func (h *DynamoComponentDeploymentHandler) RegisterWithManager(mgr manager.Manager) error {
	// Wrap the handler with lease-aware logic for cluster-wide coordination
	leaseAwareValidator := internalwebhook.NewLeaseAwareValidator(h, internalwebhook.GetExcludedNamespaces())

	// Wrap with metrics collection
	observedValidator := observability.NewObservedValidator(leaseAwareValidator, consts.ResourceTypeDynamoComponentDeployment)

	webhook := admission.
		WithCustomValidator(mgr.GetScheme(), &nvidiacomv1alpha1.DynamoComponentDeployment{}, observedValidator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dynamoComponentDeploymentWebhookPath, webhook)
	return nil
}

// castToDynamoComponentDeployment attempts to cast a runtime.Object to a DynamoComponentDeployment.
func castToDynamoComponentDeployment(obj runtime.Object) (*nvidiacomv1alpha1.DynamoComponentDeployment, error) {
	deployment, ok := obj.(*nvidiacomv1alpha1.DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", obj)
	}
	return deployment, nil
}
