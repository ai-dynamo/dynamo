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
var dynamocomponentdeploymentlog = logf.Log.WithName("dynamocomponentdeployment-resource")

// Ensure DynamoComponentDeployment implements admission.CustomValidator interface
var _ admission.CustomValidator = &DynamoComponentDeployment{}

// SetupWebhookWithManager will setup the manager to manage the webhooks.
// The validator is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic.
func (r *DynamoComponentDeployment) SetupWebhookWithManager(mgr ctrl.Manager) error {
	// Wrap the validator with lease-aware logic
	// This transparently adds namespace exclusion without modifying validation methods
	validator := NewLeaseAwareValidator(r, webhookExcludedNamespaces)

	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		WithValidator(validator).
		Complete()
}

//+kubebuilder:webhook:path=/validate-nvidia-com-v1alpha1-dynamocomponentdeployment,mutating=false,failurePolicy=fail,sideEffects=None,groups=nvidia.com,resources=dynamocomponentdeployments,verbs=create;update,versions=v1alpha1,name=vdynamocomponentdeployment.kb.io,admissionReviewVersions=v1

// Validate performs stateless validation on the DynamoComponentDeployment.
// This can be called from webhooks, controllers, or tests without needing the old object.
// It validates:
// - Replicas (non-negative)
// - Autoscaling configuration
// - Ingress configuration
func (r *DynamoComponentDeployment) Validate() error {
	// Validate replicas if specified
	if r.Spec.Replicas != nil && *r.Spec.Replicas < 0 {
		return fmt.Errorf("spec.replicas must be non-negative")
	}

	// Validate autoscaling configuration if specified
	if r.Spec.Autoscaling != nil {
		if err := r.validateAutoscaling(); err != nil {
			return err
		}
	}

	// Validate ingress configuration if enabled
	if r.Spec.Ingress != nil && r.Spec.Ingress.Enabled {
		if err := r.validateIngress(); err != nil {
			return err
		}
	}

	return nil
}

// ValidateComponentUpdate performs stateful validation comparing old and new DynamoComponentDeployment.
// This checks immutability constraints that require comparing with the previous state.
// It validates:
// - backendFramework immutability
func (r *DynamoComponentDeployment) ValidateComponentUpdate(old *DynamoComponentDeployment) error {
	// Validate that BackendFramework is not changed (immutable)
	if r.Spec.BackendFramework != old.Spec.BackendFramework {
		return fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation")
	}

	return nil
}

// ValidateCreate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoComponentDeployment) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	deployment, ok := obj.(*DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", obj)
	}

	dynamocomponentdeploymentlog.Info("validate create", "name", deployment.Name)

	// Use reusable stateless validation
	return nil, deployment.Validate()
}

// ValidateUpdate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoComponentDeployment) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	newDeployment, ok := newObj.(*DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", newObj)
	}

	dynamocomponentdeploymentlog.Info("validate update", "name", newDeployment.Name)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newDeployment.DeletionTimestamp.IsZero() {
		dynamocomponentdeploymentlog.Info("skipping validation for resource being deleted", "name", newDeployment.Name)
		return nil, nil
	}

	// Validate new object using reusable stateless validation
	if err := newDeployment.Validate(); err != nil {
		return nil, err
	}

	// Check for immutable fields using stateful validation
	oldDeployment, ok := oldObj.(*DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", oldObj)
	}

	if err := newDeployment.ValidateComponentUpdate(oldDeployment); err != nil {
		return admission.Warnings{"Changing spec.backendFramework may cause service disruption"}, err
	}

	return nil, nil
}

// ValidateDelete implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoComponentDeployment) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	deployment, ok := obj.(*DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", obj)
	}

	dynamocomponentdeploymentlog.Info("validate delete", "name", deployment.Name)

	// No validation needed for delete
	return nil, nil
}

// validateAutoscaling validates autoscaling configuration
func (r *DynamoComponentDeployment) validateAutoscaling() error {
	if r.Spec.Autoscaling.MinReplicas < 1 {
		return fmt.Errorf("spec.autoscaling.minReplicas must be at least 1")
	}

	if r.Spec.Autoscaling.MaxReplicas < 1 {
		return fmt.Errorf("spec.autoscaling.maxReplicas must be at least 1")
	}

	if r.Spec.Autoscaling.MinReplicas > r.Spec.Autoscaling.MaxReplicas {
		return fmt.Errorf("spec.autoscaling.minReplicas cannot be greater than maxReplicas")
	}

	return nil
}

// validateIngress validates ingress configuration
func (r *DynamoComponentDeployment) validateIngress() error {
	if r.Spec.Ingress.UseVirtualService {
		// If using VirtualService, validate gateway is specified
		if r.Spec.Ingress.VirtualServiceGateway == nil || *r.Spec.Ingress.VirtualServiceGateway == "" {
			return fmt.Errorf("spec.ingress.virtualServiceGateway is required when useVirtualService is true")
		}
	} else {
		// If using standard Ingress, validate required fields
		if r.Spec.Ingress.Host == "" && (r.Spec.Ingress.HostPrefix == nil || *r.Spec.Ingress.HostPrefix == "") {
			return fmt.Errorf("either spec.ingress.host or spec.ingress.hostPrefix must be specified")
		}
	}

	return nil
}
