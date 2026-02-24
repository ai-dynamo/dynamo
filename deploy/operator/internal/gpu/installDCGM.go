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
	"strings"

	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/cli"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	GPUOperatorReleaseName = "gpu-operator"
	GPUOperatorNamespace   = "gpu-operator"
)

// EnsureDCGMEnabled checks if the GPU Operator is installed and whether DCGM is enabled.
//
// This function performs the following actions:
// 1. Initializes Helm configuration to interact with the cluster.
// 2. Verifies that the GPU Operator is installed by checking its Helm release.
// 3. If the GPU Operator is installed, it checks the Helm values to ensure that DCGM is enabled.
// 4. If DCGM is not enabled, it returns an error prompting the user to enable it manually.
// 5. If the GPU Operator is not installed, it returns an error prompting the user to install it first.
//
// Logs appropriate warning or success messages based on the status of GPU Operator and DCGM.
//
// Returns:
// - nil if DCGM is enabled and GPU Operator is installed.
// - An error if GPU Operator is not installed or DCGM is not enabled.
// TODO: set a Controller Runtime status flag to indicate DCGM is not enabled.
func EnsureDCGMEnabled(ctx context.Context) error {
	logger := log.FromContext(ctx)

	// Step 1: Check if GPU Operator is installed using Helm
	settings := cli.New()

	// Explicitly configure Helm environment
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

	// Step 2: Check if GPU Operator is installed
	get := action.NewGet(actionConfig)
	release, err := get.Run(GPUOperatorReleaseName)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			logger.Info("GPU Operator is not installed", "namespace", GPUOperatorNamespace)
			return fmt.Errorf("GPU Operator is not installed. Please install GPU Operator first.")
		}
		logger.Error(err, "Failed to fetch GPU Operator release with Helm")
		return fmt.Errorf("unable to fetch GPU Operator release: %w", err)
	}

	// Step 3: Check if DCGM is enabled within the GPU Operator Helm release values
	values := release.Config
	if values == nil {
		return fmt.Errorf("GPU Operator release does not have any config values set")
	}

	// Check if the 'dcgm' section exists and if it's enabled
	dcgmValues, ok := values["dcgm"].(map[string]interface{})
	if !ok || dcgmValues["enabled"] == nil || dcgmValues["enabled"].(bool) != true {
		logger.Info("DCGM is not enabled in the GPU Operator", "namespace", GPUOperatorNamespace)
		return fmt.Errorf("DCGM is not enabled in the GPU Operator. Please enable it manually.")
	}

	// Step 4: Log success if DCGM is enabled
	logger.Info("DCGM is enabled in the GPU Operator", "namespace", GPUOperatorNamespace)
	return nil
}
