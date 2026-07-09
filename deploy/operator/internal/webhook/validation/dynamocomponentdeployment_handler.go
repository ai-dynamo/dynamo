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
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoComponentDeploymentWebhookName is the name of the validating webhook handler for DynamoComponentDeployment.
	DynamoComponentDeploymentWebhookName         = "dynamocomponentdeployment-validating-webhook"
	dynamoComponentDeploymentV1Alpha1WebhookPath = "/validate-nvidia-com-v1alpha1-dynamocomponentdeployment"
	dynamoComponentDeploymentV1Beta1WebhookPath  = "/validate/nvidia.com/v1beta1/dynamocomponentdeployments"
)

// DynamoComponentDeploymentHandler is a handler for validating DynamoComponentDeployment resources.
// It is a thin wrapper around DynamoComponentDeploymentValidator.
type DynamoComponentDeploymentHandler struct{}

// dynamoComponentDeploymentV1Alpha1Handler keeps the previous endpoint available
// during the v1alpha1-to-v1beta1 admission migration. It converts the spoke
// request to the v1beta1 hub before invoking the shared validation logic.
type dynamoComponentDeploymentV1Alpha1Handler struct {
	handler *DynamoComponentDeploymentHandler
}

// NewDynamoComponentDeploymentHandler creates a new handler for DynamoComponentDeployment Webhook.
func NewDynamoComponentDeploymentHandler() *DynamoComponentDeploymentHandler {
	return &DynamoComponentDeploymentHandler{}
}

// ValidateCreate validates a DynamoComponentDeployment create request.
func (h *DynamoComponentDeploymentHandler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return h.validateCreate(ctx, obj, nvidiacomv1beta1.DynamoComponentDeploymentGVK)
}

func (h *DynamoComponentDeploymentHandler) validateCreate(
	ctx context.Context,
	obj runtime.Object,
	expectedGVK schema.GroupVersionKind,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, expectedGVK); err != nil {
		return nil, err
	}

	deployment, err := castToDynamoComponentDeployment(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", deployment.Name, "namespace", deployment.Namespace)

	validator := NewDynamoComponentDeploymentValidator()
	return validator.Validate(ctx, deployment)
}

// ValidateUpdate validates a DynamoComponentDeployment update request.
func (h *DynamoComponentDeploymentHandler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	return h.validateUpdate(ctx, oldObj, newObj, nvidiacomv1beta1.DynamoComponentDeploymentGVK)
}

func (h *DynamoComponentDeploymentHandler) validateUpdate(
	ctx context.Context,
	oldObj, newObj runtime.Object,
	expectedGVK schema.GroupVersionKind,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, expectedGVK); err != nil {
		return nil, err
	}

	newDeployment, err := castToDynamoComponentDeployment(newObj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate update", "name", newDeployment.Name, "namespace", newDeployment.Namespace)

	// Skip validation if the resource is being deleted to allow finalizer removal.
	if !newDeployment.DeletionTimestamp.IsZero() {
		logger.Info("skipping validation for resource being deleted", "name", newDeployment.Name)
		return nil, nil
	}

	oldDeployment, err := castToDynamoComponentDeployment(oldObj)
	if err != nil {
		return nil, err
	}

	validator := NewDynamoComponentDeploymentValidator()
	warnings, err := validator.Validate(ctx, newDeployment)
	if err != nil {
		return warnings, err
	}

	updateWarnings, err := validator.ValidateUpdate(ctx, oldDeployment, newDeployment)
	if err != nil {
		return updateWarnings, err
	}

	warnings = append(warnings, updateWarnings...)
	return warnings, nil
}

// ValidateDelete validates a DynamoComponentDeployment delete request.
func (h *DynamoComponentDeploymentHandler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return h.validateDelete(ctx, obj, nvidiacomv1beta1.DynamoComponentDeploymentGVK)
}

func (h *DynamoComponentDeploymentHandler) validateDelete(
	ctx context.Context,
	obj runtime.Object,
	expectedGVK schema.GroupVersionKind,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoComponentDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, expectedGVK); err != nil {
		return nil, err
	}

	deployment, err := dynamoComponentDeploymentMetadata(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", deployment.GetName(), "namespace", deployment.GetNamespace())
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic
// and ObservedValidator to add metrics collection.
func (h *DynamoComponentDeploymentHandler) RegisterWithManager(mgr manager.Manager) error {
	h.registerWithManager(
		mgr,
		&nvidiacomv1beta1.DynamoComponentDeployment{},
		dynamoComponentDeploymentV1Beta1WebhookPath,
		h,
	)

	// TODO(1.5): Remove the v1alpha1 endpoint and handler after 1.3 is no longer
	// a supported upgrade or rollback target.
	alphaHandler := &dynamoComponentDeploymentV1Alpha1Handler{handler: h}
	h.registerWithManager(
		mgr,
		&nvidiacomv1alpha1.DynamoComponentDeployment{},
		dynamoComponentDeploymentV1Alpha1WebhookPath,
		alphaHandler,
	)
	return nil
}

func (h *DynamoComponentDeploymentHandler) registerWithManager(
	mgr manager.Manager,
	object runtime.Object,
	path string,
	validator admission.CustomValidator,
) {
	leaseAwareValidator := internalwebhook.NewLeaseAwareValidator(validator, internalwebhook.GetExcludedNamespaces())
	observedValidator := observability.NewObservedValidator(leaseAwareValidator, consts.ResourceTypeDynamoComponentDeployment)

	webhook := admission.
		WithCustomValidator(mgr.GetScheme(), object, observedValidator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(path, webhook)
}

func (h *dynamoComponentDeploymentV1Alpha1Handler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return h.handler.validateCreate(ctx, obj, nvidiacomv1alpha1.DynamoComponentDeploymentGVK)
}

func (h *dynamoComponentDeploymentV1Alpha1Handler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	return h.handler.validateUpdate(ctx, oldObj, newObj, nvidiacomv1alpha1.DynamoComponentDeploymentGVK)
}

func (h *dynamoComponentDeploymentV1Alpha1Handler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return h.handler.validateDelete(ctx, obj, nvidiacomv1alpha1.DynamoComponentDeploymentGVK)
}
