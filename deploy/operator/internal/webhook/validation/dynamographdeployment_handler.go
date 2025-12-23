/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoGraphDeploymentWebhookName is the name of the validating webhook handler for DynamoGraphDeployment.
	DynamoGraphDeploymentWebhookName = "dynamographdeployment-validating-webhook"
	dynamoGraphDeploymentWebhookPath = "/validate-nvidia-com-v1alpha1-dynamographdeployment"
)

// DynamoGraphDeploymentHandler is a handler for validating DynamoGraphDeployment resources.
// It is a thin wrapper around DynamoGraphDeploymentValidator.
type DynamoGraphDeploymentHandler struct{}

// NewDynamoGraphDeploymentHandler creates a new handler for DynamoGraphDeployment Webhook.
func NewDynamoGraphDeploymentHandler() *DynamoGraphDeploymentHandler {
	return &DynamoGraphDeploymentHandler{}
}

// ValidateCreate validates a DynamoGraphDeployment create request.
func (h *DynamoGraphDeploymentHandler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	deployment, err := castToDynamoGraphDeployment(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", deployment.Name, "namespace", deployment.Namespace)

	// Create validator and perform validation
	validator := NewDynamoGraphDeploymentValidator(deployment)
	return validator.Validate()
}

// ValidateUpdate validates a DynamoGraphDeployment update request.
func (h *DynamoGraphDeploymentHandler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	newDeployment, err := castToDynamoGraphDeployment(newObj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newDeployment.Name, "namespace", newDeployment.Namespace)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newDeployment.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newDeployment.Name)
		return nil, nil
	}

	oldDeployment, err := castToDynamoGraphDeployment(oldObj)
	if err != nil {
		return nil, err
	}

	// Create validator and perform validation
	validator := NewDynamoGraphDeploymentValidator(newDeployment)

	// Validate stateless rules
	warnings, err := validator.Validate()
	if err != nil {
		return warnings, err
	}

	// Get user info from admission request context for identity-based validation
	var userInfo *authenticationv1.UserInfo
	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, replica changes for DGDSA-enabled services will be rejected")
		// userInfo remains nil - validateReplicasChanges will fail closed
	} else {
		userInfo = &req.UserInfo
	}

	// Validate stateful rules (immutability + replicas protection)
	updateWarnings, err := validator.ValidateUpdate(oldDeployment, userInfo)
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
func (h *DynamoGraphDeploymentHandler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	deployment, err := castToDynamoGraphDeployment(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", deployment.Name, "namespace", deployment.Namespace)

	// No special validation needed for deletion
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic.
func (h *DynamoGraphDeploymentHandler) RegisterWithManager(mgr manager.Manager) error {
	// Wrap the handler with lease-aware logic for cluster-wide coordination
	validator := internalwebhook.NewLeaseAwareValidator(h, internalwebhook.GetExcludedNamespaces())

	webhook := admission.
		WithCustomValidator(mgr.GetScheme(), &nvidiacomv1alpha1.DynamoGraphDeployment{}, validator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dynamoGraphDeploymentWebhookPath, webhook)
	return nil
}

// castToDynamoGraphDeployment attempts to cast a runtime.Object to a DynamoGraphDeployment.
func castToDynamoGraphDeployment(obj runtime.Object) (*nvidiacomv1alpha1.DynamoGraphDeployment, error) {
	deployment, ok := obj.(*nvidiacomv1alpha1.DynamoGraphDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}
	return deployment, nil
}
