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

package dynamo

import (
	"context"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ReconcileModelServicesForComponents creates headless services for components with modelRef
// This is common logic used by both DynamoGraphDeployment and DynamoComponentDeployment controllers
// reconciler must implement controller_common.Reconciler interface
func ReconcileModelServicesForComponents(
	ctx context.Context,
	reconciler commonController.Reconciler,
	owner client.Object,
	services map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec,
	namespace string,
) error {
	logger := log.FromContext(ctx)

	// Track unique base models to avoid creating duplicate services
	seenBaseModels := make(map[string]bool)

	for componentName, component := range services {
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

		// Generate headless service
		headlessService := GenerateHeadlessServiceForModel(
			ctx,
			baseModelName, // Use base model name as service name
			namespace,
			baseModelName,
		)

		// Sync the service (create or update)
		_, syncedService, err := commonController.SyncResource(
			ctx,
			reconciler,
			owner,
			func(ctx context.Context) (*corev1.Service, bool, error) {
				return headlessService, false, nil
			},
		)
		if err != nil {
			logger.Error(err, "Failed to sync headless service for model",
				"baseModelName", baseModelName,
				"componentName", componentName)
			return fmt.Errorf("failed to sync headless service for model %s: %w", baseModelName, err)
		}

		logger.Info("Synced headless service for model",
			"serviceName", syncedService.GetName(),
			"baseModelName", baseModelName,
			"namespace", namespace)
	}

	return nil
}

// GenerateHeadlessServiceForModel creates a headless service for model endpoint discovery
func GenerateHeadlessServiceForModel(
	ctx context.Context,
	serviceName string,
	namespace string,
	baseModelName string,
) *corev1.Service {
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: namespace,
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoBaseModel: baseModelName,
				"nvidia.com/managed-by":               "dynamo-operator",
			},
		},
		Spec: corev1.ServiceSpec{
			// Headless service - no ClusterIP, no load balancing
			ClusterIP: corev1.ClusterIPNone,

			// Selector to match pods with the base model label
			Selector: map[string]string{
				commonconsts.KubeLabelDynamoBaseModel: baseModelName,
			},

			// Don't publish not-ready addresses - only ready pods in EndpointSlices
			PublishNotReadyAddresses: false,

			// System port for model HTTP APIs
			Ports: []corev1.ServicePort{
				{
					Name:       commonconsts.DynamoSystemPortName,
					Port:       commonconsts.DynamoSystemPort,
					TargetPort: intstr.FromInt32(commonconsts.DynamoSystemPort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	return service
}

// AddBaseModelLabel adds the base model label to a label map if modelRef is present
func AddBaseModelLabel(labels map[string]string, modelRef *v1alpha1.ModelReference) {
	if modelRef != nil && modelRef.Name != "" {
		labels[commonconsts.KubeLabelDynamoBaseModel] = modelRef.Name
	}
}
