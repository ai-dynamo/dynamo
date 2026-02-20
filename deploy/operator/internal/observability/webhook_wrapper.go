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

package observability

import (
	"context"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// ObservedValidator wraps any CustomValidator and automatically records metrics
// for webhook validation duration, results, and denials.
type ObservedValidator struct {
	admission.CustomValidator
	resourceType string
}

// NewObservedValidator creates a new ObservedValidator wrapper
func NewObservedValidator(v admission.CustomValidator, resourceType string) *ObservedValidator {
	return &ObservedValidator{
		CustomValidator: v,
		resourceType:    resourceType,
	}
}

// ValidateCreate wraps the underlying validator's ValidateCreate method with metrics collection
func (m *ObservedValidator) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	startTime := time.Now()
	warnings, err := m.CustomValidator.ValidateCreate(ctx, obj)
	duration := time.Since(startTime)

	allowed := err == nil
	RecordWebhookAdmission(m.resourceType, "CREATE", allowed, duration)

	if !allowed {
		RecordWebhookDenial(m.resourceType, "CREATE", err)
	}

	return warnings, err
}

// ValidateUpdate wraps the underlying validator's ValidateUpdate method with metrics collection
func (m *ObservedValidator) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	startTime := time.Now()
	warnings, err := m.CustomValidator.ValidateUpdate(ctx, oldObj, newObj)
	duration := time.Since(startTime)

	allowed := err == nil
	RecordWebhookAdmission(m.resourceType, "UPDATE", allowed, duration)

	if !allowed {
		RecordWebhookDenial(m.resourceType, "UPDATE", err)
	}

	return warnings, err
}

// ValidateDelete wraps the underlying validator's ValidateDelete method with metrics collection
func (m *ObservedValidator) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	startTime := time.Now()
	warnings, err := m.CustomValidator.ValidateDelete(ctx, obj)
	duration := time.Since(startTime)

	allowed := err == nil
	RecordWebhookAdmission(m.resourceType, "DELETE", allowed, duration)

	if !allowed {
		RecordWebhookDenial(m.resourceType, "DELETE", err)
	}

	return warnings, err
}
