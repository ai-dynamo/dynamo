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

package modelendpoint

import (
	"context"
	"fmt"

	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
)

// ExtractCandidates extracts endpoint candidates from EndpointSlices
// Only returns endpoints that are pod-ready
func ExtractCandidates(endpointSlices *discoveryv1.EndpointSliceList, port int32) ([]Candidate, map[string]bool) {
	var candidates []Candidate
	serviceNames := make(map[string]bool)

	for _, slice := range endpointSlices.Items {
		serviceName := slice.Labels[discoveryv1.LabelServiceName]
		if serviceName != "" {
			serviceNames[serviceName] = true
		}

		for _, ep := range slice.Endpoints {
			if len(ep.Addresses) == 0 {
				continue
			}

			// Only consider endpoints that are ready at the pod level
			podReady := ep.Conditions.Ready != nil && *ep.Conditions.Ready
			if !podReady {
				continue
			}

			// Get pod name from TargetRef
			podName := ""
			if ep.TargetRef != nil && ep.TargetRef.Kind == "Pod" {
				podName = ep.TargetRef.Name
			}

			for _, addr := range ep.Addresses {
				address := fmt.Sprintf("http://%s:%d", addr, port)
				candidates = append(candidates, Candidate{
					Address: address,
					PodName: podName,
				})
			}
		}
	}

	return candidates, serviceNames
}

// FindModelsForBaseModel finds all DynamoModels that reference a specific base model
// Uses field indexer for efficient O(1) lookup
func FindModelsForBaseModel(
	ctx context.Context,
	c client.Client,
	namespace string,
	baseModelName string,
	indexField string,
) ([]reconcile.Request, error) {
	logs := log.FromContext(ctx)

	models := &v1alpha1.DynamoModelList{}
	if err := c.List(ctx, models,
		client.InNamespace(namespace),
		client.MatchingFields{indexField: baseModelName},
	); err != nil {
		logs.Error(err, "Failed to list DynamoModels", "baseModel", baseModelName)
		return nil, err
	}

	requests := make([]reconcile.Request, 0, len(models.Items))
	for _, model := range models.Items {
		requests = append(requests, reconcile.Request{
			NamespacedName: types.NamespacedName{
				Name:      model.Name,
				Namespace: model.Namespace,
			},
		})
	}

	if len(requests) > 0 {
		logs.V(1).Info("Found DynamoModels for base model",
			"baseModel", baseModelName,
			"count", len(requests))
	}

	return requests, nil
}
