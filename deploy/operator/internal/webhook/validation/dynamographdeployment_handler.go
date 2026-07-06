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
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoGraphDeploymentWebhookName is the name of the validating webhook handler for DynamoGraphDeployment.
	DynamoGraphDeploymentWebhookName         = "dynamographdeployment-validating-webhook"
	dynamoGraphDeploymentV1Alpha1WebhookPath = "/validate-nvidia-com-v1alpha1-dynamographdeployment"
	dynamoGraphDeploymentV1Beta1WebhookPath  = "/validate/nvidia.com/v1beta1/dynamographdeployments"
)

// DynamoGraphDeploymentHandler is a handler for validating DynamoGraphDeployment resources.
// It is a thin wrapper around DynamoGraphDeploymentValidator.
type DynamoGraphDeploymentHandler struct {
	mgr               manager.Manager
	operatorPrincipal string
	groveEnabled      bool
}

// dynamoGraphDeploymentV1Alpha1Handler keeps the previous endpoint available
// during the v1alpha1-to-v1beta1 admission migration. It converts the spoke
// request to the v1beta1 hub before invoking the shared validation logic.
type dynamoGraphDeploymentV1Alpha1Handler struct {
	handler *DynamoGraphDeploymentHandler
}

// NewDynamoGraphDeploymentHandler creates a new handler for DynamoGraphDeployment Webhook.
// mgr must not be nil.
// operatorPrincipal is the full Kubernetes SA username of the operator, used to authorize
// replica changes on scaling-adapter-enabled components (#7656).
// groveEnabled reflects the operator's runtime Grove configuration.
func NewDynamoGraphDeploymentHandler(mgr manager.Manager, operatorPrincipal string, groveEnabled bool) *DynamoGraphDeploymentHandler {
	return &DynamoGraphDeploymentHandler{
		mgr:               mgr,
		operatorPrincipal: operatorPrincipal,
		groveEnabled:      groveEnabled,
	}
}

// ValidateCreate validates a DynamoGraphDeployment create request.
func (h *DynamoGraphDeploymentHandler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return h.validateCreate(ctx, obj, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
}

func (h *DynamoGraphDeploymentHandler) validateCreate(
	ctx context.Context,
	obj runtime.Object,
	expectedGVK schema.GroupVersionKind,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, expectedGVK); err != nil {
		return nil, err
	}

	deployment, err := castToDynamoGraphDeployment(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate create", "name", deployment.Name, "namespace", deployment.Namespace)

	// Create validator with manager for API group detection and perform validation
	validator := NewDynamoGraphDeploymentValidator(h.mgr, h.groveEnabled)
	return validator.Validate(ctx, deployment)
}

// ValidateUpdate validates a DynamoGraphDeployment update request.
func (h *DynamoGraphDeploymentHandler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	return h.validateUpdate(ctx, oldObj, newObj, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
}

func (h *DynamoGraphDeploymentHandler) validateUpdate(
	ctx context.Context,
	oldObj, newObj runtime.Object,
	expectedGVK schema.GroupVersionKind,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, expectedGVK); err != nil {
		return nil, err
	}

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

	// Create validator with manager for API group detection and perform validation.
	validator := NewDynamoGraphDeploymentValidator(h.mgr, h.groveEnabled)
	warnings, err := validator.Validate(ctx, newDeployment)
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
	updateWarnings, err := validator.ValidateUpdate(ctx, oldDeployment, newDeployment, userInfo, h.operatorPrincipal)
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
	return h.validateDelete(ctx, obj, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
}

func (h *DynamoGraphDeploymentHandler) validateDelete(
	ctx context.Context,
	obj runtime.Object,
	expectedGVK schema.GroupVersionKind,
) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, expectedGVK); err != nil {
		return nil, err
	}

	deployment, err := dynamoGraphDeploymentMetadata(obj)
	if err != nil {
		return nil, err
	}

	logger.Info("validate delete", "name", deployment.GetName(), "namespace", deployment.GetNamespace())

	// No special validation needed for deletion
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager.
// The handler is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic
// and ObservedValidator to add metrics collection.
func (h *DynamoGraphDeploymentHandler) RegisterWithManager(mgr manager.Manager) error {
	h.registerWithManager(
		mgr,
		&nvidiacomv1beta1.DynamoGraphDeployment{},
		dynamoGraphDeploymentV1Beta1WebhookPath,
		h,
	)

	// TODO(1.5): Remove the v1alpha1 endpoint and handler after 1.3 is no longer
	// a supported upgrade or rollback target.
	alphaHandler := &dynamoGraphDeploymentV1Alpha1Handler{handler: h}
	h.registerWithManager(
		mgr,
		&nvidiacomv1alpha1.DynamoGraphDeployment{},
		dynamoGraphDeploymentV1Alpha1WebhookPath,
		alphaHandler,
	)
	return nil
}

func (h *DynamoGraphDeploymentHandler) registerWithManager(
	mgr manager.Manager,
	object runtime.Object,
	path string,
	validator admission.CustomValidator,
) {
	// Wrap the handler with lease-aware logic for cluster-wide coordination
	leaseAwareValidator := internalwebhook.NewLeaseAwareValidator(validator, internalwebhook.GetExcludedNamespaces())

	// Wrap with metrics collection
	observedValidator := observability.NewObservedValidator(leaseAwareValidator, consts.ResourceTypeDynamoGraphDeployment)

	webhook := admission.
		WithCustomValidator(mgr.GetScheme(), object, observedValidator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(path, webhook)
}

func (h *dynamoGraphDeploymentV1Alpha1Handler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return h.handler.validateCreate(ctx, obj, nvidiacomv1alpha1.DynamoGraphDeploymentGVK)
}

func (h *dynamoGraphDeploymentV1Alpha1Handler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	return h.handler.validateUpdate(ctx, oldObj, newObj, nvidiacomv1alpha1.DynamoGraphDeploymentGVK)
}

func (h *dynamoGraphDeploymentV1Alpha1Handler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return h.handler.validateDelete(ctx, obj, nvidiacomv1alpha1.DynamoGraphDeploymentGVK)
}

// castToDynamoGraphDeployment converts the v1alpha1 spoke to the v1beta1 hub
// used by the DGD validator, or returns a v1beta1 object unchanged.
func castToDynamoGraphDeployment(obj runtime.Object) (*nvidiacomv1beta1.DynamoGraphDeployment, error) {
	switch deployment := obj.(type) {
	case *nvidiacomv1beta1.DynamoGraphDeployment:
		return deployment, nil
	case *nvidiacomv1alpha1.DynamoGraphDeployment:
		return internalwebhook.ConvertDynamoGraphDeploymentToV1Beta1(deployment)
	default:
		return nil, fmt.Errorf("expected v1alpha1 or v1beta1 DynamoGraphDeployment but got %T", obj)
	}
}

func dynamoGraphDeploymentMetadata(obj runtime.Object) (metav1.Object, error) {
	switch deployment := obj.(type) {
	case *nvidiacomv1beta1.DynamoGraphDeployment:
		return deployment, nil
	case *nvidiacomv1alpha1.DynamoGraphDeployment:
		return deployment, nil
	default:
		return nil, fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}
}
