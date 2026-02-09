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
	// DynamoGraphDeploymentRequestWebhookName is the name of the validating webhook handler for DynamoGraphDeploymentRequest.
	DynamoGraphDeploymentRequestWebhookName = "dynamographdeploymentrequest-validating-webhook"
	dynamoGraphDeploymentRequestWebhookPath = "/validate-nvidia-com-v1alpha1-dynamographdeploymentrequest"
)

// DynamoGraphDeploymentRequestHandler is a handler for validating DynamoGraphDeploymentRequest resources.
// It is a thin wrapper around DynamoGraphDeploymentRequestValidator.
type DynamoGraphDeploymentRequestHandler struct {
	isClusterWideOperator bool
}

// NewDynamoGraphDeploymentRequestHandler creates a new handler for DynamoGraphDeploymentRequest Webhook.
// The isClusterWide parameter indicates whether the operator is running in cluster-wide or namespace-restricted mode.
func NewDynamoGraphDeploymentRequestHandler(isClusterWide bool) *DynamoGraphDeploymentRequestHandler {
	return &DynamoGraphDeploymentRequestHandler{
		isClusterWideOperator: isClusterWide,
	}
}

// ValidateCreate validates a DynamoGraphDeploymentRequest create request.
func (h *DynamoGraphDeploymentRequestHandler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestWebhookName)

	request, err := castToDynamoGraphDeploymentRequest(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", request.Name, "namespace", request.Namespace)

	// Create validator and perform validation
	validator := NewDynamoGraphDeploymentRequestValidator(request, h.isClusterWideOperator)
	return validator.Validate()
}

// ValidateUpdate validates a DynamoGraphDeploymentRequest update request.
func (h *DynamoGraphDeploymentRequestHandler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestWebhookName)

	newRequest, err := castToDynamoGraphDeploymentRequest(newObj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newRequest.Name, "namespace", newRequest.Namespace)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newRequest.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newRequest.Name)
		return nil, nil
	}

	oldRequest, err := castToDynamoGraphDeploymentRequest(oldObj)
	if err != nil {
		return nil, err
	}

	// Create validator and perform validation
	validator := NewDynamoGraphDeploymentRequestValidator(newRequest, h.isClusterWideOperator)

	// Validate stateless rules
	warnings, err := validator.Validate()
	if err != nil {
		return warnings, err
	}

	// Validate stateful rules (immutability)
	updateWarnings, err := validator.ValidateUpdate(oldRequest)
	if err != nil {
		return updateWarnings, err
	}

	// Combine warnings
	warnings = append(warnings, updateWarnings...)
	return warnings, nil
}

// ValidateDelete validates a DynamoGraphDeploymentRequest delete request.
func (h *DynamoGraphDeploymentRequestHandler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestWebhookName)

	request, err := castToDynamoGraphDeploymentRequest(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", request.Name, "namespace", request.Namespace)

	// No special validation needed for deletion
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic.
func (h *DynamoGraphDeploymentRequestHandler) RegisterWithManager(mgr manager.Manager) error {
	// Wrap the handler with lease-aware logic for cluster-wide coordination
	leaseAwareValidator := internalwebhook.NewLeaseAwareValidator(h, internalwebhook.GetExcludedNamespaces())

	// Wrap with metrics collection
	observedValidator := observability.NewObservedValidator(leaseAwareValidator, consts.ResourceTypeDynamoGraphDeploymentRequest)

	webhook := admission.
		WithCustomValidator(mgr.GetScheme(), &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{}, observedValidator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dynamoGraphDeploymentRequestWebhookPath, webhook)
	return nil
}

// castToDynamoGraphDeploymentRequest attempts to cast a runtime.Object to a DynamoGraphDeploymentRequest.
func castToDynamoGraphDeploymentRequest(obj runtime.Object) (*nvidiacomv1alpha1.DynamoGraphDeploymentRequest, error) {
	request, ok := obj.(*nvidiacomv1alpha1.DynamoGraphDeploymentRequest)
	if !ok {
		return nil, fmt.Errorf("expected DynamoGraphDeploymentRequest but got %T", obj)
	}
	return request, nil
}
