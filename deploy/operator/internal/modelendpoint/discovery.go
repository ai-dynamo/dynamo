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
	"net"
	"strconv"

	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

// ExtractCandidates extracts endpoint candidates from EndpointSlices
// Returns all pod-backed endpoints regardless of ready status
// The readiness will be determined by probing (for LoRA models) or set to false (for base models)
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

			// Get pod name from TargetRef - skip if not a Pod
			if ep.TargetRef == nil || ep.TargetRef.Kind != "Pod" {
				continue
			}
			podName := ep.TargetRef.Name

			for _, addr := range ep.Addresses {
				address := "http://" + net.JoinHostPort(addr, strconv.Itoa(int(port)))
				candidates = append(candidates, Candidate{
					Address: address,
					PodName: podName,
				})
			}
		}
	}

	return candidates, serviceNames
}

// FindModelsForBaseModel finds all DynamoModels that match a specific index value
// Uses field indexer for efficient O(1) lookup
// The indexValue can be a base model name or hash, depending on the indexField
func FindModelsForBaseModel(
	ctx context.Context,
	c client.Client,
	namespace string,
	indexValue string,
	indexField string,
) ([]reconcile.Request, error) {
	logs := log.FromContext(ctx)

	models := &v1alpha1.DynamoModelList{}
	if err := c.List(ctx, models,
		client.InNamespace(namespace),
		client.MatchingFields{indexField: indexValue},
	); err != nil {
		logs.Error(err, "Failed to list DynamoModels", "indexField", indexField, "indexValue", indexValue)
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
		logs.V(1).Info("Found DynamoModels for index value",
			"indexField", indexField,
			"indexValue", indexValue,
			"count", len(requests))
	}

	return requests, nil
}
