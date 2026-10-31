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
	"strings"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Discoverer finds the cluster's GPUs the way the DGDR controller does: DCGM first, then
// node labels when node-label discovery is enabled. It is a plain value so callers other
// than the v1beta1 reconciler (the v1beta2 admission webhook) share the same order and the
// same error wording.
type Discoverer struct {
	// Reader queries the Kubernetes API. Nil makes Discover return a configuration error.
	Reader client.Reader
	// DCGM scrapes the DCGM exporter. Nil skips it.
	DCGM  *GPUDiscovery
	Cache *GPUDiscoveryCache
	// NodeLabels reports whether the node-label fallback is enabled (the GPUDiscovery feature
	// gate). It is a func so the gate is only read when DCGM did not answer, as the v1beta1
	// reconciler always did. Nil means disabled.
	NodeLabels func() bool
}

// Discover returns the discovered GPUs, restricted to filterSKU when it is set. An error
// means no backend produced a result. d and ctx must not be nil; Reader may be nil.
func (d *Discoverer) Discover(ctx context.Context, filterSKU nvidiacomv1beta1.GPUSKUType) (*GPUInfo, error) {
	logger := log.FromContext(ctx)
	if d.Reader == nil {
		return nil, errors.New("auto-discovery failed: APIReader is not configured")
	}

	var dcgmErr error
	if d.DCGM != nil {
		// A slow exporter must not leave the node-label fallback with an expired context.
		dcgmCtx, cancel := dcgmPhaseContext(ctx, d.NodeLabels)
		info, err := d.DCGM.DiscoverGPUsFromDCGMFiltered(dcgmCtx, d.Reader, d.Cache, filterSKU)
		cancel()
		if err == nil {
			return info, nil
		}
		dcgmErr = err
		logger.Info("DCGM discovery failed, falling back to node-label discovery",
			"reason", DiscoveryFailureReason(err), "error", err.Error())
	}

	if d.NodeLabels == nil || !d.NodeLabels() {
		if dcgmErr != nil {
			return nil, fmt.Errorf("auto-discovery failed: %w", dcgmErr)
		}
		return nil, errors.New("auto-discovery failed: node-label discovery is disabled")
	}

	info, err := DiscoverGPUsFiltered(ctx, d.Reader, filterSKU)
	if err != nil {
		logger.Info("Node-label discovery also failed", "error", err.Error())
		return nil, fmt.Errorf("auto-discovery failed: %w", err)
	}
	return info, nil
}

// KnownSKU returns the hardware catalog name of the discovered GPU, or empty when the model
// is not one the catalog knows. Unlike the v1beta1 reconciler it never falls back to the raw
// model string: a v1beta2 SKU is an AISimulate catalog name, and an unknown name would only
// fail later inside the search. info must not be nil.
func KnownSKU(info *GPUInfo) nvidiacomv1beta1.GPUSKUType {
	if info.System != "" {
		return info.System
	}
	return InferHardwareSystem(info.Model)
}

// DiscoveryFailureReason classifies a GPU discovery error and
// returns a stable, actionable reason string suitable for structured logging.
//
// The classification is based on known error message patterns produced during:
//   - DCGM exporter pod discovery
//   - Helm-based GPU operator and DCGM discovery
//   - Metrics scraping
//   - Prometheus parsing
//
// If the error does not match any known category, "unknown" is returned.
func DiscoveryFailureReason(err error) string {
	const unknownReason = "unknown"

	if err == nil {
		return unknownReason
	}
	errMsg := strings.ToLower(err.Error())

	switch {
	case strings.Contains(errMsg, "list pods"):
		return "failed to list DCGM exporter pods (RBAC/cluster connectivity issue)"
	case strings.Contains(errMsg, "gpu operator is not installed"):
		return "GPU Operator not installed in expected namespace"
	case strings.Contains(errMsg, "helm init failed"):
		return "failed to initialize Helm client (RBAC, kubeconfig, or Helm driver issue)"
	case strings.Contains(errMsg, "timeout waiting for dcgm exporter pods"):
		return "timeout while waiting for DCGM exporter pods to become ready"
	case strings.Contains(errMsg, "http get"):
		return "failed to reach DCGM metrics endpoint on pod (network/port issue)"
	case strings.Contains(errMsg, "metrics endpoint") &&
		strings.Contains(errMsg, "status"):
		return "DCGM pod metrics endpoint returned non-200 status"
	case strings.Contains(errMsg, "parse prometheus metrics"):
		return "failed to parse dcgm Prometheus metrics (invalid format)"
	case strings.Contains(errMsg, "no gpus detected"):
		return "no GPUs detected in dcgm metrics (GPU model or metrics missing)"
	case strings.Contains(errMsg, "dcgm is not enabled in the gpu operator"):
		return "DCGM is not enabled in the GPU Operator (check GPU Operator configuration and permissions)"
	case strings.Contains(errMsg, "failed to scrape any dcgm exporter pod"):
		return "failed to scrape any dcgm exporter pod (check DCGM exporter pod status and network connectivity)"
	case strings.Contains(errMsg, "no gpu metrics could be parsed from any dcgm pod"):
		return "no GPU metrics could be parsed from any DCGM pod (check DCGM exporter pod status and network connectivity)"
	case strings.Contains(errMsg, "failed to create helm path"):
		return "failed to initialize Helm client (RBAC, kubeconfig, or Helm driver issue)"
	}
	return unknownReason
}

// dcgmPhaseContext bounds the DCGM phase of discovery. When ctx has a deadline and fallbackEnabled
// reports that the node-label fallback is enabled, the DCGM phase gets half of what remains, so a slow exporter cannot
// consume the whole admission budget and leave the fallback with an expired context. Otherwise
// ctx is returned unchanged. The returned cancel function must be called.
func dcgmPhaseContext(ctx context.Context, fallbackEnabled func() bool) (context.Context, context.CancelFunc) {
	deadline, ok := ctx.Deadline()
	// fallbackEnabled is only asked when there is a deadline to share: without one nothing is reserved,
	// and the callback may depend on state that a caller without a deadline never configured.
	if !ok || fallbackEnabled == nil || !fallbackEnabled() {
		return ctx, func() {}
	}
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, remaining/2)
}
