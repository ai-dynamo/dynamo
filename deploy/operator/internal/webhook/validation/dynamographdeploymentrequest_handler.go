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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoGraphDeploymentRequestWebhookName is the name of the validating webhook handler for DynamoGraphDeploymentRequest.
	DynamoGraphDeploymentRequestWebhookName = "dynamographdeploymentrequest-validating-webhook"
	dynamoGraphDeploymentRequestWebhookPath = "/validate-nvidia-com-v1beta1-dynamographdeploymentrequest"
)

// DynamoGraphDeploymentRequestHandler is a handler for validating DynamoGraphDeploymentRequest resources.
// It is a thin wrapper around DynamoGraphDeploymentRequestValidator.
type DynamoGraphDeploymentRequestHandler struct {
	isClusterWideOperator bool
	gpuDiscoveryEnabled   bool
}

// NewDynamoGraphDeploymentRequestHandler creates a new handler for DynamoGraphDeploymentRequest Webhook.
// isClusterWide indicates whether the operator has cluster-wide permissions.
// gpuDiscoveryEnabled indicates whether a ClusterRole for node read access was provisioned by Helm.
func NewDynamoGraphDeploymentRequestHandler(isClusterWide bool, gpuDiscoveryEnabled bool) *DynamoGraphDeploymentRequestHandler {
	return &DynamoGraphDeploymentRequestHandler{
		isClusterWideOperator: isClusterWide,
		gpuDiscoveryEnabled:   gpuDiscoveryEnabled,
	}
}

// ValidateCreate validates a DynamoGraphDeploymentRequest create request.
func (h *DynamoGraphDeploymentRequestHandler) ValidateCreate(ctx context.Context, request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK); err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", request.Name, "namespace", request.Namespace)

	// Create validator and perform validation
	validator := NewDynamoGraphDeploymentRequestValidator(request, h.isClusterWideOperator, h.gpuDiscoveryEnabled)
	return validator.Validate()
}

// ValidateUpdate validates a DynamoGraphDeploymentRequest update request.
func (h *DynamoGraphDeploymentRequestHandler) ValidateUpdate(ctx context.Context, oldRequest, newRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK); err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newRequest.Name, "namespace", newRequest.Namespace)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newRequest.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newRequest.Name)
		return nil, nil
	}

	// Create validator and perform validation
	validator := NewDynamoGraphDeploymentRequestValidator(newRequest, h.isClusterWideOperator, h.gpuDiscoveryEnabled)
	return validator.ValidateUpdate(oldRequest)
}

// ValidateDelete validates a DynamoGraphDeploymentRequest delete request.
func (h *DynamoGraphDeploymentRequestHandler) ValidateDelete(ctx context.Context, request *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK); err != nil {
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
		WithValidator(mgr.GetScheme(), observedValidator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dynamoGraphDeploymentRequestWebhookPath, webhook)
	return nil
}
