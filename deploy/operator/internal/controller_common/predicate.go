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
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/discovery"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// ExcludedNamespacesInterface defines the interface for checking namespace exclusions
type ExcludedNamespacesInterface interface {
	Contains(namespace string) bool
}

// DetectGroveAvailability checks if Grove is available by checking if the Grove API group is registered
func DetectGroveAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	return detectAPIGroupAvailability(ctx, mgr, "grove.io")
}

// DetectLWSAvailability checks if LWS is available by checking if the LWS API group is registered
func DetectLWSAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	return detectAPIGroupAvailability(ctx, mgr, "leaderworkerset.x-k8s.io")
}

// DetectVolcanoAvailability checks if Volcano is available by checking if the Volcano API group is registered
func DetectVolcanoAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	return detectAPIGroupAvailability(ctx, mgr, "scheduling.volcano.sh")
}

// DetectKaiSchedulerAvailability checks if Kai-scheduler is available by checking if the scheduling.run.ai API group is registered
func DetectKaiSchedulerAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	return detectAPIGroupAvailability(ctx, mgr, "scheduling.run.ai")
}

// DetectInferencePoolAvailability checks if the Gateway API Inference Extension is available
// by checking if the inference.networking.k8s.io API group is registered
func DetectInferencePoolAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	return detectAPIGroupAvailability(ctx, mgr, "inference.networking.k8s.io")
}

// detectAPIGroupAvailability checks if a specific API group is registered in the cluster
func detectAPIGroupAvailability(ctx context.Context, mgr ctrl.Manager, groupName string) bool {
	logger := log.FromContext(ctx)

	cfg := mgr.GetConfig()
	if cfg == nil {
		logger.Info("detection failed, no discovery client available", "group", groupName)
		return false
	}

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		logger.Error(err, "detection failed, could not create discovery client", "group", groupName)
		return false
	}

	apiGroups, err := discoveryClient.ServerGroups()
	if err != nil {
		logger.Error(err, "detection failed, could not list server groups", "group", groupName)
		return false
	}

	for _, group := range apiGroups.Groups {
		if group.Name == groupName {
			logger.Info("API group is available", "group", groupName)
			return true
		}
	}

	logger.Info("API group not available", "group", groupName)
	return false
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

// EphemeralDeploymentEventFilter returns a predicate that filters events based on namespace configuration.
func EphemeralDeploymentEventFilter(config *configv1alpha1.OperatorConfiguration, runtimeConfig *RuntimeConfig) predicate.Predicate {
	return predicate.NewPredicateFuncs(func(o client.Object) bool {
		l := log.FromContext(context.Background())
		objMeta, err := meta.Accessor(o)
		if err != nil {
			l.Error(err, "Error extracting object metadata")
			return false
		}
		if config.Namespace.Restricted != "" {
			// in case of a restricted namespace, we only want to process the events that are in the restricted namespace
			return objMeta.GetNamespace() == config.Namespace.Restricted
		}

		// Cluster-wide mode: check if namespace is excluded
		if runtimeConfig.ExcludedNamespaces != nil && runtimeConfig.ExcludedNamespaces.Contains(objMeta.GetNamespace()) {
			l.V(1).Info("Skipping resource - namespace is excluded",
				"namespace", objMeta.GetNamespace(),
				"resource", objMeta.GetName(),
				"kind", o.GetObjectKind().GroupVersionKind().Kind)
			return false
		}

		// in all other cases, discard the event if it is destined to an ephemeral deployment
		if strings.Contains(objMeta.GetNamespace(), "ephemeral") {
			return false
		}
		return true
	})
}
