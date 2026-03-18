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
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
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
	components map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec,
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

		// Generate headless service with deterministic name based on model name
		headlessService := generateHeadlessServiceForModel(
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
// Service name is generated deterministically from the base model name using a hash
// The base model name hash is stored as a label for efficient discovery
// The original base model name is stored in an annotation for human readability
func generateHeadlessServiceForModel(
	namespace string,
	baseModelName string,
) *corev1.Service {
	// Generate deterministic service name from model name
	serviceName := GenerateServiceName(baseModelName)

	// Hash the base model name for use in labels (no length or character restrictions)
	modelHash := HashModelName(baseModelName)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: namespace,
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoBaseModelHash: modelHash,
				"nvidia.com/managed-by":                   "dynamo-operator",
			},
			Annotations: map[string]string{
				commonconsts.KubeAnnotationDynamoBaseModel: baseModelName, // Original name for humans
			},
		},
		Spec: corev1.ServiceSpec{
			// Headless service - no ClusterIP, no load balancing
			ClusterIP: corev1.ClusterIPNone,

			// Selector to match pods with the base model hash label
			Selector: map[string]string{
				commonconsts.KubeLabelDynamoBaseModelHash: modelHash,
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

// HashModelName creates a deterministic hash from a base model name for use in labels
// Returns an 8-character hex string (always valid as a Kubernetes label value)
func HashModelName(baseModelName string) string {
	hash := sha256.Sum256([]byte(baseModelName))
	// Use 8 characters for brevity and consistency
	return hex.EncodeToString(hash[:])[:8]
}

// GenerateServiceName creates a deterministic, DNS-safe service name from a base model name
// Format: dynamo-model-{8-char-hash}
func GenerateServiceName(baseModelName string) string {
	return fmt.Sprintf("dynamo-model-%s", HashModelName(baseModelName))
}

// AddBaseModelLabel adds the base model hash label to a label map if modelRef is present
// Uses a hash of the model name to avoid label length/character restrictions
func AddBaseModelLabel(labels map[string]string, modelRef *v1alpha1.ModelReference) {
	if labels == nil || modelRef == nil || modelRef.Name == "" {
		return
	}
	labels[commonconsts.KubeLabelDynamoBaseModelHash] = HashModelName(modelRef.Name)
}

// AddBaseModelAnnotation adds the base model annotation to preserve the original model name
func AddBaseModelAnnotation(annotations map[string]string, modelRef *v1alpha1.ModelReference) {
	if annotations == nil || modelRef == nil || modelRef.Name == "" {
		return
	}
	annotations[commonconsts.KubeAnnotationDynamoBaseModel] = modelRef.Name
}
