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
	"net/http"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/workerpool"
)

const (
	// MaxConcurrentOperations is the maximum number of concurrent endpoint operations
	MaxConcurrentOperations = 10
	// RequestTimeout is the timeout for individual HTTP requests
	RequestTimeout = 15 * time.Second
	// TotalTimeout is the timeout for all operations to complete
	TotalTimeout = 30 * time.Second
)

// Client handles HTTP communication with model endpoint control APIs
type Client struct {
	httpClient *http.Client
}

// NewClient creates a new model endpoint client
func NewClient() *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: RequestTimeout,
		},
	}
}

// LoadLoRA loads a LoRA model on all endpoints in parallel with bounded concurrency
// Returns endpoint info with ready status and partial results even if some endpoints fail
func (c *Client) LoadLoRA(
	ctx context.Context,
	candidates []Candidate,
	model *v1alpha1.DynamoModel,
) ([]v1alpha1.EndpointInfo, error) {
	logs := log.FromContext(ctx)

	// Skip loading for non-LoRA models
	if !model.IsLoRA() {
		logs.V(1).Info("Skipping LoRA load for non-LoRA model", "modelType", model.Spec.ModelType)
		endpoints := make([]v1alpha1.EndpointInfo, len(candidates))
		for i, c := range candidates {
			endpoints[i] = v1alpha1.EndpointInfo{
				Address: c.Address,
				PodName: c.PodName,
				Ready:   false,
			}
		}
		return endpoints, nil
	}

	// Get source URI for LoRA loading
	sourceURI := ""
	if model.Spec.Source != nil {
		sourceURI = model.Spec.Source.URI
	}
	if sourceURI == "" {
		logs.Error(nil, "Source URI is required for LoRA models")
		return nil, fmt.Errorf("source URI is required for LoRA models")
	}

	// Build tasks for the worker pool
	tasks := make([]workerpool.Task[v1alpha1.EndpointInfo], len(candidates))
	for i, candidate := range candidates {
		tasks[i] = workerpool.Task[v1alpha1.EndpointInfo]{
			Index: i,
			Work: func(ctx context.Context) (v1alpha1.EndpointInfo, error) {
				// Load the LoRA on this endpoint (idempotent operation)
				err := c.loadLoRA(ctx, candidate.Address, model.Spec.ModelName, sourceURI)
				ready := err == nil

				return v1alpha1.EndpointInfo{
					Address: candidate.Address,
					PodName: candidate.PodName,
					Ready:   ready,
				}, err
			},
		}
	}

	// Execute all load operations in parallel with bounded concurrency
	results, err := workerpool.Execute(ctx, MaxConcurrentOperations, TotalTimeout, tasks)

	// Extract endpoint info from results and collect failures
	endpoints := make([]v1alpha1.EndpointInfo, len(results))
	readyCount := 0
	var notReadyEndpoints []string
	for _, result := range results {
		endpoints[result.Index] = result.Value
		if result.Value.Ready {
			readyCount++
		} else {
			notReadyEndpoints = append(notReadyEndpoints, result.Value.Address)
			if result.Err != nil {
				logs.Info("Endpoint load operation failed",
					"address", result.Value.Address,
					"podName", result.Value.PodName,
					"error", result.Err)
			}
		}
	}

	logs.Info("Completed parallel LoRA load operations",
		"total", len(endpoints),
		"ready", readyCount,
		"notReady", len(notReadyEndpoints),
		"notReadyEndpoints", notReadyEndpoints)

	return endpoints, err
}

// UnloadLoRA unloads a LoRA model from all endpoints in parallel
func (c *Client) UnloadLoRA(ctx context.Context, candidates []Candidate, modelName string) error {
	logs := log.FromContext(ctx)

	if len(candidates) == 0 {
		logs.Info("No candidates to unload LoRA from")
		return nil
	}

	logs.Info("Starting parallel LoRA unload", "endpointCount", len(candidates), "modelName", modelName)

	// Build tasks for the worker pool
	tasks := make([]workerpool.Task[bool], len(candidates))
	for i, candidate := range candidates {
		tasks[i] = workerpool.Task[bool]{
			Index: i,
			Work: func(ctx context.Context) (bool, error) {
				// Unload the LoRA from this endpoint (calls method in lora.go)
				err := c.unloadLoRA(ctx, candidate.Address, modelName)
				if err != nil {
					return false, err
				}
				return true, nil
			},
		}
	}

	// Execute all unload operations in parallel with bounded concurrency
	results, err := workerpool.Execute(ctx, MaxConcurrentOperations, TotalTimeout, tasks)

	// Collect successes and failures with details
	successCount := 0
	var failedEndpoints []string
	for _, result := range results {
		if result.Value {
			successCount++
		} else {
			// Log failed endpoint with error details
			endpoint := candidates[result.Index].Address
			failedEndpoints = append(failedEndpoints, endpoint)
			logs.Info("Failed to unload LoRA from endpoint",
				"address", endpoint,
				"podName", candidates[result.Index].PodName,
				"error", result.Err)
		}
	}

	logs.Info("Completed parallel LoRA unload",
		"total", len(candidates),
		"successful", successCount,
		"failed", len(failedEndpoints),
		"failedEndpoints", failedEndpoints)

	return err
}
