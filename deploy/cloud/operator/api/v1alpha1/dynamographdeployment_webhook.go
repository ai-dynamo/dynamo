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
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// log is for logging in this package.
var dynamographdeploymentlog = logf.Log.WithName("dynamographdeployment-resource")

// SetupWebhookWithManager will setup the manager to manage the webhooks
func (r *DynamoGraphDeployment) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		Complete()
}

//+kubebuilder:webhook:path=/validate-nvidia-com-v1alpha1-dynamographdeployment,mutating=false,failurePolicy=fail,sideEffects=None,groups=nvidia.com,resources=dynamographdeployments,verbs=create;update,versions=v1alpha1,name=vdynamographdeployment.kb.io,admissionReviewVersions=v1

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *DynamoGraphDeployment) ValidateCreate() (admission.Warnings, error) {
	dynamographdeploymentlog.Info("validate create", "name", r.Name)

	// Validate services if provided
	if len(r.Spec.Services) > 0 {
		for serviceName, service := range r.Spec.Services {
			if serviceName == "" {
				return nil, fmt.Errorf("service name in spec.services cannot be empty")
			}

			// Validate service replicas
			if service.Replicas != nil && *service.Replicas < 0 {
				return nil, fmt.Errorf("spec.services[%s].replicas must be non-negative", serviceName)
			}

			// Validate service autoscaling
			if service.Autoscaling != nil {
				if err := r.validateServiceAutoscaling(serviceName, service); err != nil {
					return nil, err
				}
			}
		}
	}

	// Validate environment variables for duplicates
	if err := r.validateEnvVars(); err != nil {
		return nil, err
	}

	return nil, nil
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *DynamoGraphDeployment) ValidateUpdate(old runtime.Object) (admission.Warnings, error) {
	dynamographdeploymentlog.Info("validate update", "name", r.Name)

	// Run the same validations as create
	warnings, err := r.ValidateCreate()
	if err != nil {
		return warnings, err
	}

	oldDeployment, ok := old.(*DynamoGraphDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoGraphDeployment but got %T", old)
	}

	// Validate that BackendFramework is not changed (immutable)
	if r.Spec.BackendFramework != oldDeployment.Spec.BackendFramework {
		return admission.Warnings{"Changing spec.backendFramework may cause service disruption"},
			fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation")
	}

	return warnings, nil
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *DynamoGraphDeployment) ValidateDelete() (admission.Warnings, error) {
	dynamographdeploymentlog.Info("validate delete", "name", r.Name)

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
