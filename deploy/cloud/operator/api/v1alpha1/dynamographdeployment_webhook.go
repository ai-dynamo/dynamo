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

package v1alpha1

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// log is for logging in this package.
var dynamographdeploymentlog = logf.Log.WithName("dynamographdeployment-resource")

// Ensure DynamoGraphDeployment implements admission.CustomValidator interface
var _ admission.CustomValidator = &DynamoGraphDeployment{}

// SetupWebhookWithManager will setup the manager to manage the webhooks.
// The validator is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic.
func (r *DynamoGraphDeployment) SetupWebhookWithManager(mgr ctrl.Manager) error {
	// Wrap the validator with lease-aware logic (reuses the same excludedNamespaces as DynamoComponentDeployment)
	// This transparently adds namespace exclusion without modifying validation methods
	validator := NewLeaseAwareValidator(r, webhookExcludedNamespaces)

	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		WithValidator(validator).
		Complete()
}

// Validate performs stateless validation on the DynamoGraphDeployment.
// This can be called from webhooks, controllers, or tests without needing the old object.
// It validates:
// - Services (at least one, valid names, replicas, autoscaling)
// - Environment variables (no duplicates)
func (r *DynamoGraphDeployment) Validate() error {
	// Validate services if provided
	if len(r.Spec.Services) > 0 {
		for serviceName, service := range r.Spec.Services {
			if serviceName == "" {
				return fmt.Errorf("service name in spec.services cannot be empty")
			}

			// Validate service replicas
			if service.Replicas != nil && *service.Replicas < 0 {
				return fmt.Errorf("spec.services[%s].replicas must be non-negative", serviceName)
			}

			// Validate service autoscaling
			if service.Autoscaling != nil {
				if err := r.validateServiceAutoscaling(serviceName, service); err != nil {
					return err
				}
			}
		}
	} else {
		return fmt.Errorf("spec.services is required")
	}

	// Validate environment variables for duplicates
	if err := r.validateEnvVars(); err != nil {
		return err
	}

	return nil
}

// ValidateGraphUpdate performs stateful validation comparing old and new DynamoGraphDeployment.
// This checks immutability constraints that require comparing with the previous state.
// It validates:
// - backendFramework immutability
func (r *DynamoGraphDeployment) ValidateGraphUpdate(old *DynamoGraphDeployment) error {
	// Validate that BackendFramework is not changed (immutable)
	if r.Spec.BackendFramework != old.Spec.BackendFramework {
		return fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation")
	}

	return nil
}

// ValidateCreate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoGraphDeployment) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	deployment, ok := obj.(*DynamoGraphDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}

	dynamographdeploymentlog.Info("validate create", "name", deployment.Name)

	// Use reusable stateless validation
	return nil, deployment.Validate()
}

// ValidateUpdate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoGraphDeployment) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	newDeployment, ok := newObj.(*DynamoGraphDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoGraphDeployment but got %T", newObj)
	}

	dynamographdeploymentlog.Info("validate update", "name", newDeployment.Name)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newDeployment.DeletionTimestamp.IsZero() {
		dynamographdeploymentlog.Info("skipping validation for resource being deleted", "name", newDeployment.Name)
		return nil, nil
	}

	// Validate new object using reusable stateless validation
	if err := newDeployment.Validate(); err != nil {
		return nil, err
	}

	// Check for immutable fields using stateful validation
	oldDeployment, ok := oldObj.(*DynamoGraphDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoGraphDeployment but got %T", oldObj)
	}

	if err := newDeployment.ValidateGraphUpdate(oldDeployment); err != nil {
		return admission.Warnings{"Changing spec.backendFramework may cause service disruption"}, err
	}

	return nil, nil
}

// ValidateDelete implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoGraphDeployment) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	deployment, ok := obj.(*DynamoGraphDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}

	dynamographdeploymentlog.Info("validate delete", "name", deployment.Name)

	// No validation needed for delete
	return nil, nil
}

// validateServiceAutoscaling validates autoscaling for a service
func (r *DynamoGraphDeployment) validateServiceAutoscaling(serviceName string, service *DynamoComponentDeploymentSharedSpec) error {
	if service.Autoscaling.MinReplicas < 1 {
		return fmt.Errorf("spec.services[%s].autoscaling.minReplicas must be at least 1", serviceName)
	}

	if service.Autoscaling.MaxReplicas < 1 {
		return fmt.Errorf("spec.services[%s].autoscaling.maxReplicas must be at least 1", serviceName)
	}

	if service.Autoscaling.MinReplicas > service.Autoscaling.MaxReplicas {
		return fmt.Errorf("spec.services[%s].autoscaling.minReplicas cannot be greater than maxReplicas", serviceName)
	}

	return nil
}

// validateEnvVars checks for duplicate environment variable names
func (r *DynamoGraphDeployment) validateEnvVars() error {
	envNames := make(map[string]bool)
	for _, env := range r.Spec.Envs {
		if envNames[env.Name] {
			return fmt.Errorf("duplicate environment variable name: %s", env.Name)
		}
		envNames[env.Name] = true
	}
	return nil
}
