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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	authenticationv1 "k8s.io/api/authentication/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	DynamoCheckpointWebhookName = "dynamocheckpoint-validating-webhook"
	dynamoCheckpointWebhookPath = "/validate-nvidia-com-v1alpha1-dynamocheckpoint"
)

type DynamoCheckpointHandler struct {
	operatorPrincipal string
}

func NewDynamoCheckpointHandler(operatorPrincipal string) *DynamoCheckpointHandler {
	return &DynamoCheckpointHandler{
		operatorPrincipal: operatorPrincipal,
	}
}

func (h *DynamoCheckpointHandler) requestUserInfo(ctx context.Context) *authenticationv1.UserInfo {
	request, err := admission.RequestFromContext(ctx)
	if err != nil {
		return nil
	}
	return &request.UserInfo
}

func (h *DynamoCheckpointHandler) ValidateCreate(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoCheckpointWebhookName)
	logger.Info("validate create", "name", ckpt.Name, "namespace", ckpt.Namespace)
	validator := NewDynamoCheckpointValidator()
	return validator.Validate(ctx, ckpt, h.requestUserInfo(ctx), h.operatorPrincipal)
}

func (h *DynamoCheckpointHandler) ValidateUpdate(ctx context.Context, oldCheckpoint, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoCheckpointWebhookName)
	logger.Info("validate update", "name", ckpt.Name, "namespace", ckpt.Namespace)
	if !ckpt.DeletionTimestamp.IsZero() {
		return nil, nil
	}
	validator := NewDynamoCheckpointValidator()
	return validator.ValidateUpdate(ctx, oldCheckpoint, ckpt, h.requestUserInfo(ctx), h.operatorPrincipal)
}

func (h *DynamoCheckpointHandler) ValidateDelete(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (admission.Warnings, error) {
	log.FromContext(ctx).WithName(DynamoCheckpointWebhookName).Info("validate delete", "name", ckpt.Name, "namespace", ckpt.Namespace)
	return nil, nil
}

func (h *DynamoCheckpointHandler) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	registerValidationWebhook(mgr, dynamoCheckpointWebhookPath, h, consts.ResourceTypeDynamoCheckpoint, gate)
	return nil
}
