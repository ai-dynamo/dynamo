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

package controller_common

import (
	"context"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// ExcludedNamespacesInterface defines the interface for checking namespace exclusions
type ExcludedNamespacesInterface interface {
	Contains(namespace string) bool
}

// GetDiscoveryBackend returns the discovery backend for the given annotations,
// falling back to the configured default.
// For DGD, pass in the meta annotations; for DCD, pass in the spec annotations.
func GetDiscoveryBackend(discoveryBackend configv1alpha1.DiscoveryBackend, annotations map[string]string) configv1alpha1.DiscoveryBackend {
	if dgdDiscoveryBackend, exists := annotations[commonconsts.KubeAnnotationDynamoDiscoveryBackend]; exists {
		return configv1alpha1.DiscoveryBackend(dgdDiscoveryBackend)
	}
	return discoveryBackend
}

// IsK8sDiscoveryEnabled returns whether Kubernetes discovery is enabled for the given annotations.
func IsK8sDiscoveryEnabled(discoveryBackend configv1alpha1.DiscoveryBackend, annotations map[string]string) bool {
	return GetDiscoveryBackend(discoveryBackend, annotations) == configv1alpha1.DiscoveryBackendKubernetes
}

// GetKubeDiscoveryMode returns the kube discovery mode from annotations, defaulting to pod mode.
func GetKubeDiscoveryMode(annotations map[string]string) configv1alpha1.KubeDiscoveryMode {
	if mode, exists := annotations[commonconsts.KubeAnnotationDynamoKubeDiscoveryMode]; exists {
		return configv1alpha1.KubeDiscoveryMode(mode)
	}
	return configv1alpha1.KubeDiscoveryModePod
}

// EphemeralDeploymentEventFilter returns a predicate that filters events based on namespace configuration.
func EphemeralDeploymentEventFilter(config *configv1alpha1.OperatorConfiguration, runtimeConfig *RuntimeConfig) predicate.Predicate {
	return predicate.NewPredicateFuncs(func(o client.Object) bool {
		return NamespaceAllowed(config, runtimeConfig, o, o.GetNamespace())
	})
}

// NamespaceAllowed reports whether the operator should process an event whose logical namespace is
// namespace, applying restricted-namespace, excluded-namespace, and ephemeral filtering. Callers
// filtering cluster-scoped resources pass the namespace of the namespaced object the event acts for
// (e.g. a PodSnapshotContent's bound PodSnapshot); o is used only for diagnostic logging.
func NamespaceAllowed(config *configv1alpha1.OperatorConfiguration, runtimeConfig *RuntimeConfig, o client.Object, namespace string) bool {
	if config.Namespace.Restricted != "" {
		// in case of a restricted namespace, we only want to process the events that are in the restricted namespace
		return namespace == config.Namespace.Restricted
	}

	// Namespace exclusion filters new events, not requests already in the reconcile queue.
	// This best-effort isolation is acceptable for the development-and-testing-only mode.
	if runtimeConfig.ExcludedNamespaces != nil && runtimeConfig.ExcludedNamespaces.Contains(namespace) {
		log.FromContext(context.Background()).V(1).Info("Skipping resource - namespace is excluded",
			"namespace", namespace,
			"resource", o.GetName(),
			"kind", o.GetObjectKind().GroupVersionKind().Kind)
		return false
	}

	// in all other cases, discard the event if it is destined to an ephemeral deployment
	return !strings.Contains(namespace, "ephemeral")
}
