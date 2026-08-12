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

package modelendpoint

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
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
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
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

	sourceURI := ""
	if model.Spec.Source != nil {
		sourceURI = model.Spec.Source.URI
	}
	if sourceURI == "" {
		logs.Error(nil, "Source URI is required for LoRA models")
		return nil, fmt.Errorf("source URI is required for LoRA models")
	}

	endpoints := make([]v1alpha1.EndpointInfo, len(candidates))
	for index, candidate := range candidates {
		endpoints[index] = v1alpha1.EndpointInfo{
			Address: candidate.Address,
			PodName: candidate.PodName,
		}
	}

	loadCtx, cancel := context.WithTimeout(ctx, TotalTimeout)
	defer cancel()

	// Legacy prefill coverage depends on another capable worker in the same topology.
	var fallbackMu sync.Mutex
	capableVLLMPrefillGroups := make(map[string]struct{})
	unavailableFallbackIndices := make(map[int]struct{})

	loadCandidate := func(index int) {
		candidate := candidates[index]
		// Always call the lifecycle API first so capable vLLM prefill workers
		// register and publish the adapter. Only explicitly unsupported legacy
		// prefill workers may use the rolling-upgrade fallback below.
		err := c.loadLoRA(loadCtx, candidate.Address, model.Spec.ModelName, sourceURI)
		if err != nil && candidate.AllowLoRAManagementUnavailable && isLoRAManagementUnavailable(err) {
			fallbackMu.Lock()
			unavailableFallbackIndices[index] = struct{}{}
			fallbackMu.Unlock()
			return
		}
		if err != nil {
			logs.Info("Endpoint load operation failed",
				"address", candidate.Address,
				"podName", candidate.PodName,
				"error", err)
			return
		}

		endpoints[index].Ready = true
		if candidate.AllowLoRAManagementUnavailable && candidate.LoRAFallbackGroup != "" {
			fallbackMu.Lock()
			capableVLLMPrefillGroups[candidate.LoRAFallbackGroup] = struct{}{}
			fallbackMu.Unlock()
		}
	}

	workqueue.ParallelizeUntil(loadCtx, MaxConcurrentOperations, len(candidates), loadCandidate)

	// Resolve fallback coverage only after every scheduled worker has reported capability.
	readyCount := 0
	failureCount := 0
	var notReadyEndpoints []string
	for index, candidate := range candidates {
		endpoint := &endpoints[index]
		if _, usedUnavailableFallback := unavailableFallbackIndices[index]; usedUnavailableFallback {
			// A legacy prefill can be non-serving only when another capable
			// vLLM prefill in the same runtime topology published the adapter card.
			_, covered := capableVLLMPrefillGroups[candidate.LoRAFallbackGroup]
			endpoint.LoRAFallbackCovered = candidate.LoRAFallbackGroup != "" && covered && candidate.KubernetesReady
		}

		if endpoint.Ready || endpoint.LoRAFallbackCovered {
			readyCount++
		} else {
			failureCount++
			notReadyEndpoints = append(notReadyEndpoints, candidate.Address)
		}
	}

	logs.Info("Completed parallel LoRA load operations",
		"total", len(endpoints),
		"ready", readyCount,
		"notReady", len(notReadyEndpoints),
		"loraManagementUnavailableFallbackUsed", len(unavailableFallbackIndices),
		"capableVLLMPrefillGroups", len(capableVLLMPrefillGroups),
		"notReadyEndpoints", notReadyEndpoints)

	if failureCount > 0 {
		return endpoints, fmt.Errorf("%d task(s) failed", failureCount)
	}
	return endpoints, nil
}

// UnloadLoRA unloads a LoRA model from all endpoints in parallel
func (c *Client) UnloadLoRA(ctx context.Context, candidates []Candidate, modelName string) error {
	logs := log.FromContext(ctx)

	if len(candidates) == 0 {
		logs.Info("No candidates to unload LoRA from")
		return nil
	}

	logs.Info("Starting parallel LoRA unload", "endpointCount", len(candidates), "modelName", modelName)

	fallbackEligibleCount := 0
	for _, candidate := range candidates {
		if candidate.AllowLoRAManagementUnavailable {
			fallbackEligibleCount++
		}
	}

	unloadCtx, cancel := context.WithTimeout(ctx, TotalTimeout)
	defer cancel()

	var successCount atomic.Int64

	unloadCandidate := func(index int) {
		candidate := candidates[index]
		err := c.unloadLoRA(unloadCtx, candidate.Address, modelName)
		if err == nil || candidate.AllowLoRAManagementUnavailable && isLoRAManagementUnavailable(err) {
			successCount.Add(1)
			return
		}

		logs.Info("Failed to unload LoRA from endpoint",
			"address", candidate.Address,
			"podName", candidate.PodName,
			"error", err)
	}

	workqueue.ParallelizeUntil(unloadCtx, MaxConcurrentOperations, len(candidates), unloadCandidate)

	// Treat work not started due to cancellation as failed, matching the batch contract.
	completedCount := successCount.Load()
	failureCount := int64(len(candidates)) - completedCount

	logs.Info("Completed parallel LoRA unload",
		"total", len(candidates),
		"successful", completedCount,
		"loraManagementUnavailableFallbackEligible", fallbackEligibleCount,
		"failed", failureCount)

	if failureCount > 0 {
		return fmt.Errorf("%d task(s) failed", failureCount)
	}
	return nil
}
