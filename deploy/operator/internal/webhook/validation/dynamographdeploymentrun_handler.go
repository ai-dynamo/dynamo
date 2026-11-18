/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const dynamoGraphDeploymentRunWebhookPath = "/validate-nvidia-com-v1beta2-dynamographdeploymentrun"

// DynamoGraphDeploymentRunHandler validates native v1beta2 Run snapshots.
type DynamoGraphDeploymentRunHandler struct{}

// NewDynamoGraphDeploymentRunHandler creates the native Run admission boundary.
func NewDynamoGraphDeploymentRunHandler() *DynamoGraphDeploymentRunHandler {
	return &DynamoGraphDeploymentRunHandler{}
}

// ValidateCreate validates a Run's immutable, already defaulted spec. run must not be nil.
func (h *DynamoGraphDeploymentRunHandler) ValidateCreate(ctx context.Context, run *nvidiacomv1beta2.DynamoGraphDeploymentRun) (admission.Warnings, error) {
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta2.DynamoGraphDeploymentRunGVK); err != nil {
		return nil, err
	}

	// Use the same structural input validation as Request admission, without mutation.
	validation := &dynamoGraphDeploymentRunValidation{}
	validation.ctx = ctx
	allErrs := validation.validateDynamoGraphDeploymentRunSpec(&run.Spec, field.NewPath("spec"))
	if len(allErrs) > 0 {
		return validation.warnings, apierrors.NewInvalid(nvidiacomv1beta2.DynamoGraphDeploymentRunGVK.GroupKind(), run.Name, allErrs)
	}
	return validation.warnings, nil
}

// ValidateUpdate validates the new snapshot while permitting unchanged-spec metadata/status updates.
// oldRun and newRun must not be nil.
func (h *DynamoGraphDeploymentRunHandler) ValidateUpdate(ctx context.Context, oldRun, newRun *nvidiacomv1beta2.DynamoGraphDeploymentRun) (admission.Warnings, error) {
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta2.DynamoGraphDeploymentRunGVK); err != nil {
		return nil, err
	}

	// CEL owns immutability; the shared update validator owns the schema ratchet.
	validation := &dynamoGraphDeploymentRunValidation{}
	validation.ctx = ctx
	allErrs := validation.validateDynamoGraphDeploymentRunSpecUpdate(&newRun.Spec, &oldRun.Spec, field.NewPath("spec"))
	if len(allErrs) > 0 {
		return validation.warnings, apierrors.NewInvalid(nvidiacomv1beta2.DynamoGraphDeploymentRunGVK.GroupKind(), newRun.Name, allErrs)
	}
	return validation.warnings, nil
}

// ValidateDelete has no additional deletion rules. run must not be nil.
func (h *DynamoGraphDeploymentRunHandler) ValidateDelete(ctx context.Context, run *nvidiacomv1beta2.DynamoGraphDeploymentRun) (admission.Warnings, error) {
	return nil, internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta2.DynamoGraphDeploymentRunGVK)
}

// RegisterWithManager installs the standard lease-aware, observed validation handler.
// mgr must not be nil. gate may be nil when no feature context is required.
func (h *DynamoGraphDeploymentRunHandler) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	registerValidationWebhook(mgr, dynamoGraphDeploymentRunWebhookPath, h, consts.ResourceTypeDynamoGraphDeploymentRun, gate)
	return nil
}
