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

package webhook

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

var webhookCommonLog = logf.Log.WithName("webhook-common")

// ExcludedNamespacesChecker defines the interface for checking namespace exclusions
// This matches controller_common.ExcludedNamespacesInterface to allow reuse of the
// lease-based coordination mechanism.
type ExcludedNamespacesChecker interface {
	Contains(namespace string) bool
}

// webhookExcludedNamespaces holds the excluded namespaces checker (usually leaseWatcher)
// This is set by main.go and shared across all webhook validators
var webhookExcludedNamespaces ExcludedNamespacesChecker

// SetExcludedNamespaces sets the excluded namespaces checker for all webhooks.
// This should be called from main.go before starting the webhook server.
func SetExcludedNamespaces(checker ExcludedNamespacesChecker) {
	webhookExcludedNamespaces = checker
}

// GetExcludedNamespaces returns the current excluded namespaces checker.
func GetExcludedNamespaces() ExcludedNamespacesChecker {
	return webhookExcludedNamespaces
}

// LeaseAwareValidator wraps a CustomValidator and adds lease-based namespace exclusion logic.
// It checks if a namespace-restricted operator is managing the namespace (via active lease)
// before delegating validation to the underlying validator.
//
// This implements the Decorator pattern to transparently add coordination logic without
// modifying the actual validation implementations.
type LeaseAwareValidator struct {
	validator          admission.CustomValidator
	excludedNamespaces ExcludedNamespacesChecker
}

// NewLeaseAwareValidator creates a new LeaseAwareValidator that wraps the given validator.
// If excludedNamespaces is nil, the wrapper acts as a pass-through (no filtering).
func NewLeaseAwareValidator(validator admission.CustomValidator, excludedNamespaces ExcludedNamespacesChecker) admission.CustomValidator {
	if excludedNamespaces == nil {
		// No exclusion logic needed, return validator as-is
		return validator
	}
	return &LeaseAwareValidator{
		validator:          validator,
		excludedNamespaces: excludedNamespaces,
	}
}

// ValidateCreate implements admission.CustomValidator
func (v *LeaseAwareValidator) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	if v.shouldSkipValidation(obj) {
		return nil, nil
	}
	return v.validator.ValidateCreate(ctx, obj)
}

// ValidateUpdate implements admission.CustomValidator
func (v *LeaseAwareValidator) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	if v.shouldSkipValidation(newObj) {
		return nil, nil
	}
	return v.validator.ValidateUpdate(ctx, oldObj, newObj)
}

// ValidateDelete implements admission.CustomValidator
func (v *LeaseAwareValidator) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	if v.shouldSkipValidation(obj) {
		return nil, nil
	}
	return v.validator.ValidateDelete(ctx, obj)
}

// shouldSkipValidation checks if validation should be skipped for the given object
func (v *LeaseAwareValidator) shouldSkipValidation(obj runtime.Object) bool {
	// Try to extract namespace from object using client.Object interface
	clientObj, ok := obj.(client.Object)
	if !ok {
		// If we can't determine the namespace, don't skip (fail-safe)
		return false
	}

	namespace := clientObj.GetNamespace()
	if v.excludedNamespaces.Contains(namespace) {
		webhookCommonLog.Info("skipping validation - namespace has namespace-restricted operator",
			"name", clientObj.GetName(),
			"namespace", namespace,
			"kind", obj.GetObjectKind().GroupVersionKind().Kind)
		return true
	}

	return false
}
