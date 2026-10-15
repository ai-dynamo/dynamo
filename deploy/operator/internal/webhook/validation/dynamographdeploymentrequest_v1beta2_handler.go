/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrinput"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// DynamoGraphDeploymentRequestV1Beta2WebhookName is the name of the validating webhook
	// handler for v1beta2 DynamoGraphDeploymentRequest.
	DynamoGraphDeploymentRequestV1Beta2WebhookName = "dynamographdeploymentrequest-v1beta2-validating-webhook"
	dgdrV1Beta2WebhookPath                         = "/validate-nvidia-com-v1beta2-dynamographdeploymentrequest"
)

// DynamoGraphDeploymentRequestV1Beta2Validator validates v1beta2 DynamoGraphDeploymentRequest
// resources. It checks what the CRD schema cannot: the typed objective against the native
// parameters, and the complete effective AISimulate input.
type DynamoGraphDeploymentRequestV1Beta2Validator struct {
	options dgdrinput.Options
}

// NewDynamoGraphDeploymentRequestV1Beta2Validator creates a v1beta2 validator.
func NewDynamoGraphDeploymentRequestV1Beta2Validator() *DynamoGraphDeploymentRequestV1Beta2Validator {
	return &DynamoGraphDeploymentRequestV1Beta2Validator{options: dgdrinput.DefaultOptions()}
}

// Validate validates a new request. ctx and request must not be nil.
func (v *DynamoGraphDeploymentRequestV1Beta2Validator) Validate(
	_ context.Context,
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) (admission.Warnings, error) {
	return nil, invalidV1Beta2Request(request, dgdrinput.ValidateSpec(&request.Spec, v.options))
}

// ValidateUpdate validates newRequest against oldRequest. Only violations the update
// introduces are rejected, and an update that leaves the spec unchanged (a status write,
// a metadata change) is always admitted.
func (v *DynamoGraphDeploymentRequestV1Beta2Validator) ValidateUpdate(
	_ context.Context,
	oldRequest *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	newRequest *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) (admission.Warnings, error) {
	if apiequality.Semantic.DeepEqual(&oldRequest.Spec, &newRequest.Spec) {
		return nil, nil
	}
	introduced := dgdrinput.NewViolations(
		dgdrinput.ValidateSpec(&newRequest.Spec, v.options),
		dgdrinput.ValidateSpec(&oldRequest.Spec, v.options),
	)
	return nil, invalidV1Beta2Request(newRequest, introduced)
}

func invalidV1Beta2Request(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest, errs field.ErrorList) error {
	if len(errs) == 0 {
		return nil
	}
	return apierrors.NewInvalid(nvidiacomv1beta2.DynamoGraphDeploymentRequestGVK.GroupKind(), request.Name, errs)
}

// DynamoGraphDeploymentRequestV1Beta2Handler is the admission handler for validating
// v1beta2 DynamoGraphDeploymentRequest resources. It is a thin wrapper around the validator.
type DynamoGraphDeploymentRequestV1Beta2Handler struct{}

// NewDynamoGraphDeploymentRequestV1Beta2Handler creates a handler.
func NewDynamoGraphDeploymentRequestV1Beta2Handler() *DynamoGraphDeploymentRequestV1Beta2Handler {
	return &DynamoGraphDeploymentRequestV1Beta2Handler{}
}

// ValidateCreate validates a create request.
func (h *DynamoGraphDeploymentRequestV1Beta2Handler) ValidateCreate(ctx context.Context, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestV1Beta2WebhookName)
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta2.DynamoGraphDeploymentRequestGVK); err != nil {
		return nil, err
	}
	logger.Info("validate create", "name", request.Name, "namespace", request.Namespace)
	return NewDynamoGraphDeploymentRequestV1Beta2Validator().Validate(ctx, request)
}

// ValidateUpdate validates an update request.
func (h *DynamoGraphDeploymentRequestV1Beta2Handler) ValidateUpdate(ctx context.Context, oldRequest, newRequest *nvidiacomv1beta2.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoGraphDeploymentRequestV1Beta2WebhookName)
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta2.DynamoGraphDeploymentRequestGVK); err != nil {
		return nil, err
	}
	logger.Info("validate update", "name", newRequest.Name, "namespace", newRequest.Namespace)

	// Skip validation while the resource is being deleted, so finalizers can be removed.
	if !newRequest.DeletionTimestamp.IsZero() {
		return nil, nil
	}
	return NewDynamoGraphDeploymentRequestV1Beta2Validator().ValidateUpdate(ctx, oldRequest, newRequest)
}

// ValidateDelete admits every delete.
func (h *DynamoGraphDeploymentRequestV1Beta2Handler) ValidateDelete(ctx context.Context, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta2.DynamoGraphDeploymentRequestGVK); err != nil {
		return nil, err
	}
	return nil, nil
}

// RegisterWithManager registers the webhook with the manager. registerValidationWebhook
// wraps the handler with the lease-aware namespace exclusion.
func (h *DynamoGraphDeploymentRequestV1Beta2Handler) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	registerValidationWebhook(mgr, dgdrV1Beta2WebhookPath, h, consts.ResourceTypeDynamoGraphDeploymentRequest, gate)
	return nil
}
