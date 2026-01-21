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

package observability

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

const (
	resourceCountInterval = 30 * time.Second
)

// ExcludedNamespaces defines the interface for checking namespace exclusions
type ExcludedNamespaces interface {
	Contains(namespace string) bool
}

// StartResourceCounter starts a background goroutine that periodically updates resource count metrics.
// It uses the manager's cached client to avoid loading the API server.
// The client's cache scope is automatically determined by the manager's configuration:
// - Namespace-restricted operators: cache is scoped to specific namespace
// - Cluster-wide operators: cache includes all namespaces (except those filtered by excludedNamespaces)
// The excludedNamespaces parameter allows filtering out namespaces managed by namespace-restricted operators.
func StartResourceCounter(ctx context.Context, c client.Client, excludedNamespaces ExcludedNamespaces) {
	logger := log.FromContext(ctx).WithName("resource-counter")
	logger.Info("Starting resource counter", "interval", resourceCountInterval)

	ticker := time.NewTicker(resourceCountInterval)
	defer ticker.Stop()

	// Initial update
	updateResourceMetrics(ctx, c, excludedNamespaces, logger)

	for {
		select {
		case <-ctx.Done():
			logger.Info("Stopping resource counter")
			return
		case <-ticker.C:
			updateResourceMetrics(ctx, c, excludedNamespaces, logger)
		}
	}
}

// updateResourceMetrics queries DynamoGraphDeployments and updates gauges
// The client's cache scope determines which namespaces are queried
func updateResourceMetrics(ctx context.Context, c client.Client, excludedNamespaces ExcludedNamespaces, logger logr.Logger) {
	// Count DynamoGraphDeployments by state
	updateDynamoGraphDeploymentCounts(ctx, c, excludedNamespaces, logger)
}

func updateDynamoGraphDeploymentCounts(ctx context.Context, c client.Client, excludedNamespaces ExcludedNamespaces, logger logr.Logger) {
	dgdList := &v1alpha1.DynamoGraphDeploymentList{}
	if err := c.List(ctx, dgdList); err != nil {
		logger.Error(err, "failed to list DynamoGraphDeployments")
		return
	}

	// Count by state and namespace
	counts := make(map[string]map[string]int)
	for i := range dgdList.Items {
		dgd := &dgdList.Items[i]
		namespace := dgd.Namespace

		// Skip if namespace is managed by a namespace-restricted operator
		if excludedNamespaces != nil && excludedNamespaces.Contains(namespace) {
			continue
		}

		state := dgd.Status.State
		if state == "" {
			state = "unknown"
		}

		if counts[namespace] == nil {
			counts[namespace] = make(map[string]int)
		}
		counts[namespace][state]++
	}

	// Update metrics
	for namespace, stateCounts := range counts {
		for state, count := range stateCounts {
			UpdateResourceCount(consts.ResourceTypeDynamoGraphDeployment, namespace, state, float64(count))
		}
	}
}
