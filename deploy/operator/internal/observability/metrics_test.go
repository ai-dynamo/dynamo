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
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

func TestInitMetricsOptsInToRESTClientMetrics(t *testing.T) {
	// Stub the opt-in seam to capture which client-go REST metrics InitMetrics
	// requests; the seam is the production call path, so this proves the opt-in
	// regardless of test order and of controller-runtime's internal sync.Once.
	var requested []ctrlmetrics.RESTClientMetric
	originalRegister := registerRESTClientMetrics
	registerRESTClientMetrics = func(metrics ...ctrlmetrics.RESTClientMetric) {
		requested = append(requested, metrics...)
	}
	t.Cleanup(func() { registerRESTClientMetrics = originalRegister })

	// InitMetrics registers custom collectors on the package-global registry;
	// swap in a temporary one so repeated InitMetrics calls across tests are safe.
	originalRegistry := ctrlmetrics.Registry
	ctrlmetrics.Registry = prometheus.NewRegistry()
	t.Cleanup(func() { ctrlmetrics.Registry = originalRegistry })

	InitMetrics("9.9.9")

	t.Log("all six opt-in REST client metrics are requested")
	assert.ElementsMatch(t, []ctrlmetrics.RESTClientMetric{
		ctrlmetrics.MetricRequestLatency,
		ctrlmetrics.MetricDNSResolutionLatency,
		ctrlmetrics.MetricRequestSize,
		ctrlmetrics.MetricResponseSize,
		ctrlmetrics.MetricRateLimiterLatency,
		ctrlmetrics.MetricRequestRetry,
	}, requested)
}
