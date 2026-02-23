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
	corev1 "k8s.io/api/core/v1"
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

// WaitForDCGMPods waits until at least one DCGM exporter pod is running
// and has a valid Pod IP, or until the provided timeout is reached.
//
// The function polls the Kubernetes API at regular intervals,
// checking for pods matching DCGM exporter labels.
//
// A pod is considered ready when:
//   - Status.Phase == Running
//   - PodIP is non-empty
//
// Returns:
//   - nil once a ready DCGM exporter pod is detected
//   - error if the timeout expires or if a Kubernetes API call fails
//
// This function is typically used after enabling DCGM via Helm
// to ensure pods are ready before metrics scraping begins.
// WaitForDCGMPods waits until at least one DCGM exporter pod is running
// and has a valid Pod IP, or until the provided timeout is reached.
//
// It checks pods matching the supported DCGM exporter labels and
// polls the Kubernetes API at regular intervals.
//
// Returns nil when a ready pod is found, or an error if timeout expires.
func WaitForDCGMPods(ctx context.Context, k8sClient client.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)

	// List of label selectors to support multiple DCGM exporter labels
	labelSelectors := []client.MatchingLabels{
		{LabelAppKubernetesName: LabelValueNvidiaDCGMExporter},
		{LabelApp: LabelValueDCGMExporter},
	}

	for time.Now().Before(deadline) {
		for _, labels := range labelSelectors {
			podList := &corev1.PodList{}
			if err := k8sClient.List(ctx, podList, labels); err != nil {
				return fmt.Errorf("listing pods with labels %v: %w", labels, err)
			}

			for _, pod := range podList.Items {
				if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
					return nil
				}
			}
		}

		time.Sleep(5 * time.Second)
	}

	return fmt.Errorf("timeout waiting for DCGM exporter pods")
}
