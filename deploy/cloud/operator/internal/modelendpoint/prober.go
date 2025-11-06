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
	"net/http"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/workerpool"
)

const (
	// MaxConcurrentProbes is the maximum number of concurrent endpoint probes
	MaxConcurrentProbes = 10
	// ProbeTimeout is the timeout for individual HTTP requests
	ProbeTimeout = 15 * time.Second
	// TotalProbeTimeout is the timeout for all probes to complete
	TotalProbeTimeout = 30 * time.Second
)

// Prober handles HTTP-based endpoint probing
type Prober struct {
	httpClient *http.Client
}

// NewProber creates a new endpoint prober
func NewProber() *Prober {
	return &Prober{
		httpClient: &http.Client{
			Timeout: ProbeTimeout,
		},
	}
}

// ProbeEndpoints probes all endpoints in parallel with bounded concurrency
// Returns partial results even if some endpoints fail
func (p *Prober) ProbeEndpoints(
	ctx context.Context,
	candidates []Candidate,
	model *v1alpha1.DynamoModel,
) ([]v1alpha1.EndpointInfo, error) {
	logs := log.FromContext(ctx)

	// Skip probing for non-LoRA models
	if strings.ToLower(model.Spec.ModelType) != "lora" {
		logs.V(1).Info("Skipping probe for non-LoRA model", "modelType", model.Spec.ModelType)
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

	// Build tasks for the worker pool
	tasks := make([]workerpool.Task[v1alpha1.EndpointInfo], len(candidates))
	for i, candidate := range candidates {
		c := candidate // Capture loop variable
		tasks[i] = workerpool.Task[v1alpha1.EndpointInfo]{
			Index: i,
			Work: func(ctx context.Context) (v1alpha1.EndpointInfo, error) {
				// Probe the endpoint
				ready := p.probeLoRAEndpoint(ctx, c.Address, model.Spec.ModelName)

				logs.V(1).Info("Endpoint probe completed", "address", c.Address, "ready", ready)

				return v1alpha1.EndpointInfo{
					Address: c.Address,
					PodName: c.PodName,
					Ready:   ready,
				}, nil
			},
		}
	}

	// Execute all probes in parallel with bounded concurrency
	results, err := workerpool.Execute(ctx, MaxConcurrentProbes, TotalProbeTimeout, tasks)

	// Extract endpoint info from results
	endpoints := make([]v1alpha1.EndpointInfo, len(results))
	readyCount := 0
	for _, result := range results {
		endpoints[result.Index] = result.Value
		if result.Value.Ready {
			readyCount++
		}
	}

	logs.Info("Completed parallel endpoint probing",
		"total", len(endpoints),
		"ready", readyCount)

	return endpoints, err
}

// UnloadLoRA unloads a LoRA model from all endpoints in parallel
func (p *Prober) UnloadLoRA(ctx context.Context, candidates []Candidate, modelName string) error {
	logs := log.FromContext(ctx)

	if len(candidates) == 0 {
		logs.Info("No candidates to unload LoRA from")
		return nil
	}

	logs.Info("Starting parallel LoRA unload", "endpointCount", len(candidates), "modelName", modelName)

	// Build tasks for the worker pool
	tasks := make([]workerpool.Task[bool], len(candidates))
	for i, candidate := range candidates {
		c := candidate // Capture loop variable
		tasks[i] = workerpool.Task[bool]{
			Index: i,
			Work: func(ctx context.Context) (bool, error) {
				// Unload the LoRA from this endpoint (calls method in lora.go)
				err := p.unloadLoRA(ctx, c.Address, modelName)
				if err != nil {
					logs.V(1).Info("Failed to unload LoRA from endpoint",
						"address", c.Address,
						"modelName", modelName,
						"error", err)
					return false, err
				}

				logs.V(1).Info("Successfully unloaded LoRA from endpoint",
					"address", c.Address,
					"modelName", modelName)
				return true, nil
			},
		}
	}

	// Execute all unload operations in parallel with bounded concurrency
	results, err := workerpool.Execute(ctx, MaxConcurrentProbes, TotalProbeTimeout, tasks)

	// Count successes
	successCount := 0
	for _, result := range results {
		if result.Value {
			successCount++
		}
	}

	logs.Info("Completed parallel LoRA unload",
		"total", len(candidates),
		"successful", successCount,
		"failed", len(candidates)-successCount)

	return err
}
