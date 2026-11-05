/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"
	"encoding/json"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	jsonpatch "github.com/evanphx/json-patch/v5"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const candidateWebhookPath = "/validate/nvidia.com/v1beta2/dynamographdeploymentcandidates"

// DynamoGraphDeploymentCandidateHandler preserves immutability of opaque DGD specs.
type DynamoGraphDeploymentCandidateHandler struct{}

func (h *DynamoGraphDeploymentCandidateHandler) ValidateCreate(ctx context.Context, obj *v1beta2.DynamoGraphDeploymentCandidate) (admission.Warnings, error) {
	return nil, internalwebhook.ValidateAdmissionGVK(ctx, v1beta2.DynamoGraphDeploymentCandidateGVK)
}

func (h *DynamoGraphDeploymentCandidateHandler) ValidateUpdate(ctx context.Context, oldObj, newObj *v1beta2.DynamoGraphDeploymentCandidate) (admission.Warnings, error) {
	if err := internalwebhook.ValidateAdmissionGVK(ctx, v1beta2.DynamoGraphDeploymentCandidateGVK); err != nil {
		return nil, err
	}
	allErrs := validateDynamoGraphDeploymentCandidateUpdate(newObj, oldObj)
	if len(allErrs) == 0 {
		return nil, nil
	}
	return nil, apierrors.NewInvalid(v1beta2.DynamoGraphDeploymentCandidateGVK.GroupKind(), newObj.Name, allErrs)
}

func (h *DynamoGraphDeploymentCandidateHandler) ValidateDelete(ctx context.Context, obj *v1beta2.DynamoGraphDeploymentCandidate) (admission.Warnings, error) {
	return nil, internalwebhook.ValidateAdmissionGVK(ctx, v1beta2.DynamoGraphDeploymentCandidateGVK)
}

// validateDynamoGraphDeploymentCandidateUpdate compares complete JSON values, including opaque fields.
// newObj and oldObj must not be nil. JSON formatting and object member order do not change the spec.
func validateDynamoGraphDeploymentCandidateUpdate(newObj, oldObj *v1beta2.DynamoGraphDeploymentCandidate) field.ErrorList {
	newSpec, newErr := json.Marshal(newObj.Spec)
	oldSpec, oldErr := json.Marshal(oldObj.Spec)
	if newErr != nil || oldErr != nil || !jsonpatch.Equal(newSpec, oldSpec) {
		return field.ErrorList{field.Forbidden(field.NewPath("spec"), "spec is immutable")}
	}
	return nil
}

func (h *DynamoGraphDeploymentCandidateHandler) RegisterWithManager(mgr manager.Manager, gate features.Gate) {
	registerValidationWebhook(mgr, candidateWebhookPath, h, "DynamoGraphDeploymentCandidate", gate)
}
