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
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// log is for logging in this package.
var dynamomodellog = logf.Log.WithName("dynamomodel-resource")

// Ensure DynamoModel implements admission.CustomValidator interface
var _ admission.CustomValidator = &DynamoModel{}

// SetupWebhookWithManager will setup the manager to manage the webhooks.
// The validator is automatically wrapped with LeaseAwareValidator to add namespace exclusion logic.
func (r *DynamoModel) SetupWebhookWithManager(mgr ctrl.Manager) error {
	// Wrap the validator with lease-aware logic
	// This transparently adds namespace exclusion without modifying validation methods
	validator := NewLeaseAwareValidator(r, webhookExcludedNamespaces)

	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		WithValidator(validator).
		Complete()
}

//+kubebuilder:webhook:path=/validate-nvidia-com-v1alpha1-dynamomodel,mutating=false,failurePolicy=fail,sideEffects=None,groups=nvidia.com,resources=dynamomodels,verbs=create;update,versions=v1alpha1,name=vdynamomodel.kb.io,admissionReviewVersions=v1

// Validate performs stateless validation on the DynamoModel.
// This can be called from webhooks, controllers, or tests without needing the old object.
// It validates:
// - Required fields (modelName, baseModelName)
// - LoRA-specific requirements (source and URI)
// - URI format validation
func (r *DynamoModel) Validate() error {
	// Validate modelName is not empty
	if r.Spec.ModelName == "" {
		return fmt.Errorf("spec.modelName is required")
	}

	// Validate baseModelName is not empty
	if r.Spec.BaseModelName == "" {
		return fmt.Errorf("spec.baseModelName is required")
	}

	// Validate LoRA model requirements
	if r.IsLoRA() {
		if r.Spec.Source == nil {
			return fmt.Errorf("spec.source is required when modelType is 'lora'")
		}

		if r.Spec.Source.URI == "" {
			return fmt.Errorf("spec.source.uri must be specified when modelType is 'lora'")
		}

		// Validate URI format
		if err := validateSourceURI(r.Spec.Source.URI); err != nil {
			return err
		}
	}

	return nil
}

// ValidateModelUpdate performs stateful validation comparing old and new DynamoModel.
// This checks immutability constraints that require comparing with the previous state.
// It validates:
// - modelType immutability
// - baseModelName immutability
func (r *DynamoModel) ValidateModelUpdate(old *DynamoModel) error {
	// modelType is immutable
	if r.Spec.ModelType != old.Spec.ModelType {
		return fmt.Errorf("spec.modelType is immutable and cannot be changed after creation")
	}

	// baseModelName is immutable
	if r.Spec.BaseModelName != old.Spec.BaseModelName {
		return fmt.Errorf("spec.baseModelName is immutable and cannot be changed after creation")
	}

	return nil
}

// ValidateCreate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoModel) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	model, ok := obj.(*DynamoModel)
	if !ok {
		return nil, fmt.Errorf("expected DynamoModel but got %T", obj)
	}

	dynamomodellog.Info("validate create", "name", model.Name)

	// Use reusable stateless validation
	return nil, model.Validate()
}

// ValidateUpdate implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoModel) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	newModel, ok := newObj.(*DynamoModel)
	if !ok {
		return nil, fmt.Errorf("expected DynamoModel but got %T", newObj)
	}

	dynamomodellog.Info("validate update", "name", newModel.Name)

	// Skip validation if the resource is being deleted (to allow finalizer removal)
	if !newModel.DeletionTimestamp.IsZero() {
		dynamomodellog.Info("skipping validation for resource being deleted", "name", newModel.Name)
		return nil, nil
	}

	// Validate new object using reusable stateless validation
	if err := newModel.Validate(); err != nil {
		return nil, err
	}

	// Check for immutable fields using stateful validation
	oldModel, ok := oldObj.(*DynamoModel)
	if !ok {
		return nil, fmt.Errorf("expected DynamoModel but got %T", oldObj)
	}

	if err := newModel.ValidateModelUpdate(oldModel); err != nil {
		// Determine which field changed for appropriate warning
		var warning string
		if newModel.Spec.ModelType != oldModel.Spec.ModelType {
			warning = "Changing spec.modelType may cause unexpected behavior"
		} else if newModel.Spec.BaseModelName != oldModel.Spec.BaseModelName {
			warning = "Changing spec.baseModelName will break endpoint discovery"
		}

		if warning != "" {
			return admission.Warnings{warning}, err
		}
		return nil, err
	}

	return nil, nil
}

// ValidateDelete implements webhook.CustomValidator so a webhook will be registered for the type
func (r *DynamoModel) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	model, ok := obj.(*DynamoModel)
	if !ok {
		return nil, fmt.Errorf("expected DynamoModel but got %T", obj)
	}

	dynamomodellog.Info("validate delete", "name", model.Name)

	// No special validation needed for deletion
	return nil, nil
}

// validateSourceURI validates the model source URI format
func validateSourceURI(uri string) error {
	if uri == "" {
		return fmt.Errorf("source URI cannot be empty")
	}

	// Check for supported schemes
	if !strings.HasPrefix(uri, "s3://") && !strings.HasPrefix(uri, "hf://") {
		return fmt.Errorf("source URI must start with 's3://' or 'hf://', got: %s", uri)
	}

	return nil
}
