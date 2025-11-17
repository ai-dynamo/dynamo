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

// SetupWebhookWithManager will setup the manager to manage the webhooks
func (r *DynamoComponentDeployment) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		WithValidator(r).
		Complete()
}

//+kubebuilder:webhook:path=/validate-nvidia-com-v1alpha1-dynamocomponentdeployment,mutating=false,failurePolicy=fail,sideEffects=None,groups=nvidia.com,resources=dynamocomponentdeployments,verbs=create;update,versions=v1alpha1,name=vdynamocomponentdeployment.kb.io,admissionReviewVersions=v1

// ValidateCreate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoComponentDeployment) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	deployment, ok := obj.(*DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", obj)
	}

	dynamocomponentdeploymentlog.Info("validate create", "name", deployment.Name)

	// Validate ServiceName if provided
	if deployment.Spec.ServiceName != "" && len(deployment.Spec.ServiceName) > 63 {
		return nil, fmt.Errorf("spec.serviceName must be 63 characters or less")
	}

	// Validate replicas if specified
	if deployment.Spec.Replicas != nil && *deployment.Spec.Replicas < 0 {
		return nil, fmt.Errorf("spec.replicas must be non-negative")
	}

	// Validate autoscaling configuration if specified
	if deployment.Spec.Autoscaling != nil {
		if err := deployment.validateAutoscaling(); err != nil {
			return nil, err
		}
	}

	// Validate ingress configuration if enabled
	if deployment.Spec.Ingress != nil && deployment.Spec.Ingress.Enabled {
		if err := deployment.validateIngress(); err != nil {
			return nil, err
		}
	}

	return nil, nil
}

// ValidateUpdate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoComponentDeployment) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	newDeployment, ok := newObj.(*DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", newObj)
	}

	dynamocomponentdeploymentlog.Info("validate update", "name", newDeployment.Name)

	// Run the same validations as create
	warnings, err := newDeployment.ValidateCreate(ctx, newObj)
	if err != nil {
		return warnings, err
	}

	oldDeployment, ok := oldObj.(*DynamoComponentDeployment)
	if !ok {
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", oldObj)
	}

	// Validate that BackendFramework is not changed (immutable)
	if newDeployment.Spec.BackendFramework != oldDeployment.Spec.BackendFramework {
		return admission.Warnings{"Changing spec.backendFramework may cause service disruption"},
			fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation")
	}

	return warnings, nil
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
