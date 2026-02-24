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

package gpu

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/cli"
)

const (
	GPUOperatorReleaseName = "gpu-operator"
	GPUOperatorNamespace   = "gpu-operator"
)

// EnsureDCGMEnabled ensures that both DCGM and DCGM Exporter are enabled
// in the NVIDIA GPU Operator Helm release for the specified namespace.
//
// Function Behavior:
//  1. Checks if the GPU Operator Helm release exists in the given namespace.
//     - If not installed, returns an error asking the user to install it first.
//  2. Retrieves the existing Helm release values and merges in the following:
//     - dcgm.enabled = true       (enables GPU telemetry collection)
//     - dcgmExporter.enabled = true (enables Prometheus metrics export)
//  3. Performs a Helm upgrade in-place using the existing chart, preserving
//     other user-provided values.
//
// Notes:
//   - Does NOT install the GPU Operator; it only modifies an existing release.
//   - Both DCGM and DCGM Exporter must be enabled to collect full GPU metrics
//     (utilization, memory, power, ECC errors, MIG profiles, per-process stats).
//   - The function assumes the caller has cluster-wide RBAC and access to
//     the target namespace for Helm operations.
//   - Uses /tmp as a temporary HOME directory for Helm caching and config.
func EnsureDCGMEnabled(ctx context.Context) error {
	// Configure Helm environment
	settings := cli.New()

	// Explicitly configure paths
	settings.RepositoryCache = "/tmp/helm/.cache/repository"
	settings.RepositoryConfig = "/tmp/helm/.config/repositories.yaml"
	settings.RegistryConfig = "/tmp/helm/.config/registry.json"
	settings.PluginsDirectory = "/tmp/helm/plugins"

	paths := []string{
		settings.RepositoryCache,
		filepath.Dir(settings.RepositoryConfig),
		filepath.Dir(settings.RegistryConfig),
		filepath.Dir(settings.PluginsDirectory),
	}
	for _, p := range paths {
		if err := os.MkdirAll(p, 0o755); err != nil {
			return fmt.Errorf("failed to create helm path %s: %w", p, err)
		}
	}

	// Initialize Helm action configuration for the target namespace
	actionConfig := new(action.Configuration)
	if err := actionConfig.Init(
		settings.RESTClientGetter(),
		GPUOperatorNamespace,
		os.Getenv("HELM_DRIVER"),
		func(format string, v ...interface{}) {}, // no-op logger
	); err != nil {
		return fmt.Errorf("helm init failed: %w", err)
	}

	// Verify GPU Operator is installed
	get := action.NewGet(actionConfig)
	release, err := get.Run(GPUOperatorReleaseName)
	if err != nil {
		return fmt.Errorf(
			"gpu operator is not installed in namespace %s. please install gpu operator first",
			GPUOperatorNamespace,
		)
	}

	// Upgrade GPU Operator and enable DCGM + DCGM Exporter
	upgrade := action.NewUpgrade(actionConfig)
	upgrade.Namespace = GPUOperatorNamespace

	// Start with existing release values to avoid overwriting user config
	values := release.Config
	if values == nil {
		values = map[string]interface{}{}
	}

	// Enable DCGM
	dcgmValues, ok := values["dcgm"].(map[string]interface{})
	if !ok {
		dcgmValues = map[string]interface{}{}
	}
	dcgmValues["enabled"] = true
	values["dcgm"] = dcgmValues

	// Enable DCGM Exporter
	dcgmExporterValues, ok := values["dcgmExporter"].(map[string]interface{})
	if !ok {
		dcgmExporterValues = map[string]interface{}{}
	}
	dcgmExporterValues["enabled"] = true
	values["dcgmExporter"] = dcgmExporterValues

	// Reuse existing chart to upgrade in-place
	chart := release.Chart

	if _, err := upgrade.Run(GPUOperatorReleaseName, chart, values); err != nil {
		return fmt.Errorf("failed to enable dcgm and dcgmExporter in gpu operator: %w", err)
	}

	return nil
}
