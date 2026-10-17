/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package defaulting

import (
	"context"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrinput"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// defaultMaxHardwareBudget caps a discovered GPU budget. It matches the limit DGDR
	// v1beta1 applies to auto-discovered hardware, which keeps a search from using every GPU
	// in a large cluster. Set a budget explicitly to exceed it.
	defaultMaxHardwareBudget int32 = 32

	// defaultHardwareDiscoveryTimeout bounds GPU discovery inside the admission call. Keep it
	// below the webhook's timeoutSeconds, or the API server gives up first and the request
	// fails under the webhook's failurePolicy instead of reaching validation.
	defaultHardwareDiscoveryTimeout = 5 * time.Second
)

// DiscoveredHardware is what the cluster has, as far as it can be determined.
type DiscoveredHardware struct {
	// SKU is the AISimulate hardware catalog name, or empty when the GPU model is not one
	// the catalog knows.
	SKU string
	// GPUs is the total number of GPUs of that SKU in the cluster.
	GPUs int
}

// HardwareDiscoverer finds the cluster's GPUs. filterSKU, when set, restricts the result to
// GPUs of that SKU.
type HardwareDiscoverer interface {
	Discover(ctx context.Context, filterSKU string) (DiscoveredHardware, error)
}

// DGDRV1Beta2DefaulterOptions configures the v1beta2 DGDR defaulter.
type DGDRV1Beta2DefaulterOptions struct {
	// Hardware discovers the cluster's GPUs. Nil disables hardware defaulting, and
	// validation then requires the hardware to be spelled out.
	Hardware HardwareDiscoverer
	// HardwareDiscoveryTimeout defaults to 5 seconds.
	HardwareDiscoveryTimeout time.Duration
	// MaxBudget caps a discovered GPU budget; zero means 32.
	MaxBudget int32
	// AISimulateImage is put into spec.aisimulate.image when unset; empty leaves it unset.
	AISimulateImage string
}

// DGDRV1Beta2Defaulter fills the v1beta2 Request values the CRD schema cannot: the scalar
// objective metric, the AISimulate runner image, and the hardware discovered from the
// cluster. It works on a v1beta2 spec and knows nothing about how the spec reaches the
// webhook; the caller decides when defaulting applies (creation only).
type DGDRV1Beta2Defaulter struct {
	options DGDRV1Beta2DefaulterOptions
}

// NewDGDRV1Beta2Defaulter creates the v1beta2 DGDR defaulter.
func NewDGDRV1Beta2Defaulter(options DGDRV1Beta2DefaulterOptions) *DGDRV1Beta2Defaulter {
	return &DGDRV1Beta2Defaulter{options: options}
}

// DefaultSpec fills the missing values of spec in place. It never replaces a value the
// caller set, typed or native. A hardware discovery failure is not an error: the hardware
// stays unset and validation rejects the Request with a message that names the fields.
func (d *DGDRV1Beta2Defaulter) DefaultSpec(ctx context.Context, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
	d.defaultAISimulateImage(spec)
	if metric, ok := dgdrinput.MetricDefault(spec); ok {
		spec.Objective.Metric = &metric
	}
	d.defaultHardware(ctx, spec)
}

func (d *DGDRV1Beta2Defaulter) defaultAISimulateImage(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
	if d.options.AISimulateImage == "" {
		return
	}
	if spec.AISimulate == nil {
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{}
	}
	if spec.AISimulate.Image == "" {
		spec.AISimulate.Image = d.options.AISimulateImage
	}
}

func (d *DGDRV1Beta2Defaulter) defaultHardware(ctx context.Context, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
	if d.options.Hardware == nil {
		return
	}
	logger := log.FromContext(ctx)

	timeout := d.options.HardwareDiscoveryTimeout
	if timeout <= 0 {
		timeout = defaultHardwareDiscoveryTimeout
	}
	discoveryCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	sku, budget, err := d.resolveHardware(discoveryCtx, spec)
	if err != nil {
		logger.Info("hardware discovery failed; leaving hardware unresolved", "error", err.Error())
		return
	}
	applyHardware(spec, sku, budget)
	if sku != "" || budget != 0 {
		logger.Info("defaulted spec.hardware.gpu from the cluster", "sku", sku, "budget", budget)
	}
}

// resolveHardware returns the SKU and budget missing from the effective input; a zero value
// is not missing. The typed spec and spec.aisimulate.parameters.search_space may each supply
// either value, so each is resolved on its own: a caller can pin the SKU and let admission
// size the budget. Discovery runs only when something is missing, restricted to the
// effective SKU when there is one. d.options.Hardware must not be nil.
func (d *DGDRV1Beta2Defaulter) resolveHardware(ctx context.Context, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) (string, int32, error) {
	sku, hasBudget, err := dgdrinput.Hardware(spec)
	if err != nil {
		return "", 0, err
	}
	if sku != "" && hasBudget {
		return "", 0, nil
	}

	found, err := d.options.Hardware.Discover(ctx, sku)
	if err != nil {
		return "", 0, err
	}

	var defaultSKU string
	var defaultBudget int32
	if sku == "" {
		defaultSKU = found.SKU
	}
	if !hasBudget && found.GPUs > 0 {
		maxBudget := d.options.MaxBudget
		if maxBudget <= 0 {
			maxBudget = defaultMaxHardwareBudget
		}
		defaultBudget = int32(min(found.GPUs, int(maxBudget)))
	}
	return defaultSKU, defaultBudget, nil
}

// applyHardware writes the resolved values into the typed spec. It never replaces a typed
// value.
func applyHardware(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec, sku string, budget int32) {
	if sku == "" && budget == 0 {
		return
	}
	if spec.Hardware == nil {
		spec.Hardware = &nvidiacomv1beta2.HardwareSpec{}
	}
	if spec.Hardware.GPU == nil {
		spec.Hardware.GPU = &nvidiacomv1beta2.GPUHardwareSpec{}
	}
	gpuSpec := spec.Hardware.GPU
	if sku != "" && (gpuSpec.SKU == nil || *gpuSpec.SKU == "") {
		gpuSpec.SKU = &sku
	}
	if budget > 0 && gpuSpec.Budget == nil {
		gpuSpec.Budget = &budget
	}
}

// gpuHardwareDiscoverer backs HardwareDiscoverer with the operator's GPU discovery.
type gpuHardwareDiscoverer struct {
	discoverer *gpu.Discoverer
}

// NewGPUHardwareDiscoverer adapts the operator's GPU discovery for hardware defaulting.
func NewGPUHardwareDiscoverer(discoverer *gpu.Discoverer) HardwareDiscoverer {
	return gpuHardwareDiscoverer{discoverer: discoverer}
}

func (g gpuHardwareDiscoverer) Discover(ctx context.Context, filterSKU string) (DiscoveredHardware, error) {
	info, err := g.discoverer.Discover(ctx, nvidiacomv1beta1.GPUSKUType(filterSKU))
	if err != nil {
		return DiscoveredHardware{}, err
	}
	return DiscoveredHardware{
		SKU:  string(gpu.KnownSKU(info)),
		GPUs: info.GPUsPerNode * info.NodesWithGPUs,
	}, nil
}
