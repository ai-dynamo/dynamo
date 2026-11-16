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
	"runtime"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

func TestBuildInfoCollector(t *testing.T) {
	// Use a temporary registry — never the global ctrlmetrics.Registry.
	reg := prometheus.NewRegistry()
	reg.MustRegister(buildInfoCollector("9.9.9"))

	families, err := reg.Gather()
	require.NoError(t, err)
	require.Len(t, families, 1)

	family := families[0]
	assert.Equal(t, "dynamo_operator_build_info", family.GetName())
	require.Len(t, family.GetMetric(), 1)

	metric := family.GetMetric()[0]
	assert.Equal(t, 1.0, metric.GetGauge().GetValue())

	labels := map[string]string{}
	for _, lp := range metric.GetLabel() {
		labels[lp.GetName()] = lp.GetValue()
	}

	assert.Equal(t, "9.9.9", labels["version"])
	assert.Equal(t, runtime.GOOS+"/"+runtime.GOARCH, labels["platform"])
	assert.NotEmpty(t, labels["go_version"])
	// Test binaries usually lack VCS settings, so the fallback "unknown" is
	// expected — do not assert exact values.
	assert.NotEmpty(t, labels["git_commit"])
	assert.NotEmpty(t, labels["git_commit_date"])
}

func TestInitMetricsRegistersBuildInfo(t *testing.T) {
	// InitMetrics registers into the package-global controller-runtime
	// registry; swap in a fresh one so repeated runs (-count=2) do not panic
	// on duplicate registration. Not parallel: replaces package-global state.
	originalRegistry := ctrlmetrics.Registry
	ctrlmetrics.Registry = prometheus.NewRegistry()
	t.Cleanup(func() {
		ctrlmetrics.Registry = originalRegistry
	})

	InitMetrics("9.9.9")

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)

	var found int
	for _, f := range families {
		if f.GetName() != "dynamo_operator_build_info" {
			continue
		}
		found++
		require.Len(t, f.GetMetric(), 1)
		metric := f.GetMetric()[0]
		assert.Equal(t, 1.0, metric.GetGauge().GetValue())

		// InitMetrics must forward operatorVersion to the version label.
		labels := map[string]string{}
		for _, lp := range metric.GetLabel() {
			labels[lp.GetName()] = lp.GetValue()
		}
		assert.Equal(t, "9.9.9", labels["version"])
	}
	assert.Equal(t, 1, found, "dynamo_operator_build_info must be registered exactly once")
}
