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

package gpu

import (
	"context"
	"errors"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Discoverer finds the cluster's GPUs the way the DGDR controller does: DCGM first, then
// node labels when node-label discovery is enabled. It is a plain value so callers other
// than the v1beta1 reconciler (the v1beta2 admission webhook) share the same order and the
// same error wording.
type Discoverer struct {
	Reader client.Reader
	// DCGM scrapes the DCGM exporter. Nil skips it.
	DCGM  *GPUDiscovery
	Cache *GPUDiscoveryCache
	// NodeLabels enables the node-label fallback (the GPUDiscovery feature gate).
	NodeLabels bool
}

// Discover returns the discovered GPUs, restricted to filterSKU when it is set. An error
// means no backend produced a result.
func (d *Discoverer) Discover(ctx context.Context, filterSKU nvidiacomv1beta1.GPUSKUType) (*GPUInfo, error) {
	if d == nil || d.Reader == nil {
		return nil, errors.New("auto-discovery failed: APIReader is not configured")
	}

	var dcgmErr error
	if d.DCGM != nil {
		info, err := d.DCGM.DiscoverGPUsFromDCGMFiltered(ctx, d.Reader, d.Cache, filterSKU)
		if err == nil {
			return info, nil
		}
		dcgmErr = err
	}

	if !d.NodeLabels {
		if dcgmErr != nil {
			return nil, fmt.Errorf("auto-discovery failed: %w", dcgmErr)
		}
		return nil, errors.New("auto-discovery failed: node-label discovery is disabled")
	}

	info, err := DiscoverGPUsFiltered(ctx, d.Reader, filterSKU)
	if err != nil {
		return nil, fmt.Errorf("auto-discovery failed: %w", errors.Join(dcgmErr, err))
	}
	return info, nil
}

// KnownSKU returns the hardware catalog name of the discovered GPU, or empty when the model
// is not one the catalog knows. Unlike the v1beta1 reconciler it never falls back to the raw
// model string: a v1beta2 SKU is an AISimulate catalog name, and an unknown name would only
// fail later inside the search.
func KnownSKU(info *GPUInfo) nvidiacomv1beta1.GPUSKUType {
	if info == nil {
		return ""
	}
	if info.System != "" {
		return info.System
	}
	return InferHardwareSystem(info.Model)
}
