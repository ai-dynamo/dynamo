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

// Package dgdrhardware resolves the hardware of a v1beta2 DynamoGraphDeploymentRequest
// that the caller left out.
//
// The typed spec and spec.aisimulate.parameters.search_space each may supply the GPU SKU
// and the GPU budget. Only what neither supplies is discovered from the cluster, and each
// value is resolved on its own, so a caller can pin the SKU and let admission size the
// budget. The package knows nothing about how discovery works; it talks to a Discoverer,
// which the webhook backs with the operator's DCGM and node-label discovery.
package dgdrhardware

import (
	"context"
	"errors"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrinput"
)

// DefaultMaxBudget caps a discovered GPU budget. It matches the limit DGDR v1beta1 applies
// to auto-discovered hardware, which exists to keep a search from using every GPU in a
// large cluster. Set a budget explicitly to exceed it.
const DefaultMaxBudget int32 = 32

// Discovered is what the cluster has, as far as it can be determined.
type Discovered struct {
	// SKU is the AISimulate hardware catalog name, or empty when the GPU model is not one
	// the catalog knows.
	SKU string
	// GPUs is the total number of GPUs of that SKU in the cluster.
	GPUs int
}

// Discoverer finds the cluster's GPUs. filterSKU, when set, restricts the result to GPUs
// of that SKU.
type Discoverer interface {
	Discover(ctx context.Context, filterSKU string) (Discovered, error)
}

// Options tunes Resolve. A zero MaxBudget means DefaultMaxBudget.
type Options struct {
	MaxBudget int32
}

// Defaults holds the values to persist. A zero field is not set.
type Defaults struct {
	SKU    string
	Budget int32
}

// Resolve returns the hardware values missing from the effective input. It discovers only
// when something is missing and returns the zero Defaults when nothing is. A discovery
// error means nothing could be resolved; the validating webhook then reports the hardware
// as unresolved.
func Resolve(ctx context.Context, discoverer Discoverer, spec *v1beta2.DynamoGraphDeploymentRequestSpec, opts Options) (Defaults, error) {
	sku, hasBudget, err := dgdrinput.Hardware(spec)
	if err != nil {
		return Defaults{}, err
	}
	if sku != "" && hasBudget {
		return Defaults{}, nil
	}
	if discoverer == nil {
		return Defaults{}, errors.New("hardware is incomplete and no hardware discoverer is configured")
	}

	found, err := discoverer.Discover(ctx, sku)
	if err != nil {
		return Defaults{}, err
	}

	var defaults Defaults
	if sku == "" {
		defaults.SKU = found.SKU
	}
	if !hasBudget && found.GPUs > 0 {
		maxBudget := opts.MaxBudget
		if maxBudget <= 0 {
			maxBudget = DefaultMaxBudget
		}
		defaults.Budget = int32(min(found.GPUs, int(maxBudget)))
	}
	return defaults, nil
}

// Apply writes defaults into the typed spec. It never replaces a typed value.
func Apply(spec *v1beta2.DynamoGraphDeploymentRequestSpec, defaults Defaults) {
	if defaults.SKU == "" && defaults.Budget == 0 {
		return
	}
	if spec.Hardware == nil {
		spec.Hardware = &v1beta2.HardwareSpec{}
	}
	if spec.Hardware.GPU == nil {
		spec.Hardware.GPU = &v1beta2.GPUHardwareSpec{}
	}
	gpu := spec.Hardware.GPU
	if defaults.SKU != "" && (gpu.SKU == nil || *gpu.SKU == "") {
		sku := defaults.SKU
		gpu.SKU = &sku
	}
	if defaults.Budget > 0 && gpu.Budget == nil {
		budget := defaults.Budget
		gpu.Budget = &budget
	}
}
