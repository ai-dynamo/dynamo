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

package dynamo

import (
	"context"
	"fmt"
	"maps"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/modelservice"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ReconcileModelServicesForComponents creates services for components with modelRef
// This is common logic used by both DynamoGraphDeployment and DynamoComponentDeployment controllers
// reconciler must implement controller_common.Reconciler interface
func ReconcileModelServicesForComponents(
	ctx context.Context,
	reconciler commonController.Reconciler,
	owner client.Object,
	components map[string]*v1beta1.DynamoComponentDeploymentSharedSpec,
	namespace string,
) error {
	logger := log.FromContext(ctx)

	// Track unique base models to avoid creating duplicate services
	seenBaseModels := make(map[string]bool)

	for componentName, component := range components {
		// Skip if no modelRef
		if component.ModelRef == nil || component.ModelRef.Name == "" {
			continue
		}

		baseModelName := component.ModelRef.Name

		// Skip if we've already created service for this base model
		if seenBaseModels[baseModelName] {
			logger.V(1).Info("Skipping duplicate headless service for base model",
				"componentName", componentName,
				"baseModelName", baseModelName)
			continue
		}
		seenBaseModels[baseModelName] = true

		annotations := make(map[string]string)
		if dgd, ok := owner.(*v1beta1.DynamoGraphDeployment); ok {
			if dgd.Spec.Annotations != nil {
				maps.Copy(annotations, dgd.Spec.Annotations)
			}
		}

		// Generate headless service with deterministic name based on model name
		headlessService := modelservice.Render(modelservice.Params{Namespace: namespace, ModelName: baseModelName, Annotations: annotations})

		// Sync the service (create or update)
		_, syncedService, err := commonController.SyncResource(
			ctx,
			reconciler,
			owner,
			func(ctx context.Context) (*corev1.Service, bool, error) {
				return headlessService, false, nil
			},
			// Components that serve the same base model share this Service.
			commonController.WithSharedOwnership(),
		)
		if err != nil {
			logger.Error(err, "Failed to sync headless service for model",
				"baseModelName", baseModelName,
				"componentName", componentName)
			return fmt.Errorf("failed to sync headless service for model %s: %w", baseModelName, err)
		}

		if syncedService.Annotations == nil {
			syncedService.Annotations = make(map[string]string)
		}
		var updateAnnotations bool
		for key, value := range annotations {
			if val, ok := syncedService.Annotations[key]; !ok || val != value {
				syncedService.Annotations[key] = value
				updateAnnotations = true
			}
		}
		if updateAnnotations {
			err = reconciler.Update(ctx, syncedService)
			if err != nil {
				logger.Error(err, fmt.Sprintf("Failed to update model service %s.", componentName))
				reconciler.GetRecorder().Eventf(owner, syncedService, corev1.EventTypeWarning, "UpdateService", "Update", "Failed to update model Service %s: %s", componentName, err)
				return fmt.Errorf("failed to update main component service %s: %w", componentName, err)
			}
		}

		logger.Info("Synced headless service for model",
			"serviceName", syncedService.GetName(),
			"baseModelName", baseModelName,
			"namespace", namespace)
	}

	return nil
}

// AddBaseModelLabel adds the base model hash label to a label map if modelRef is present
// Uses a hash of the model name to avoid label length/character restrictions
func AddBaseModelLabel(labels map[string]string, modelRef *v1beta1.ModelReference) {
	if labels == nil || modelRef == nil || modelRef.Name == "" {
		return
	}
	labels[commonconsts.KubeLabelDynamoBaseModelHash] = modelservice.HashModelName(modelRef.Name)
}

// AddBaseModelAnnotation adds the base model annotation to preserve the original model name
func AddBaseModelAnnotation(annotations map[string]string, modelRef *v1beta1.ModelReference) {
	if annotations == nil || modelRef == nil || modelRef.Name == "" {
		return
	}
	annotations[commonconsts.KubeAnnotationDynamoBaseModel] = modelRef.Name
}
