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
	"time"

	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/cli"
	discoveryv1 "k8s.io/api/discovery/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
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
	// Set temporary HOME for Helm to write cache/config in controller environments.
	os.Setenv("HOME", "/tmp")

	settings := cli.New()
	settings.RepositoryCache = "/tmp/.cache/helm/repository"
	settings.RepositoryConfig = "/tmp/.config/helm/repositories.yaml"

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

// WaitForDCGMService waits until the DCGM exporter Service has at least one ready endpoint.
//
// It polls the Kubernetes API, checking the Endpoints associated with the
// "nvidia-dcgm-exporter" Service in the specified namespace. Returns as soon as
// any endpoint is ready, or an error if the timeout elapses.
//
// Parameters:
//   - ctx: context for cancellation and deadlines.
//   - c: Kubernetes client used to list Endpoints.
//   - timeout: maximum duration to wait for an endpoint to become ready.
//
// Returns an error if the timeout is exceeded, or if there is a failure accessing the API.
func WaitForDCGMService(ctx context.Context, c client.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	serviceName := "nvidia-dcgm-exporter"

	for time.Now().Before(deadline) {
		sliceList := &discoveryv1.EndpointSliceList{}

		err := c.List(ctx, sliceList,
			client.InNamespace(GPUOperatorNamespace),
			client.MatchingLabels{
				"kubernetes.io/service-name": serviceName,
			},
		)
		if err != nil {
			return fmt.Errorf("failed to list EndpointSlices for %s/%s: %w", GPUOperatorNamespace, serviceName, err)
		}

		for _, slice := range sliceList.Items {
			for _, endpoint := range slice.Endpoints {
				if endpoint.Conditions.Ready != nil && *endpoint.Conditions.Ready {
					return nil
				}
			}
		}

		time.Sleep(5 * time.Second)
	}

	return fmt.Errorf("timeout waiting for DCGM exporter service endpoints in namespace %s", GPUOperatorNamespace)
}
