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
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/cli"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// EnsureDCGMHelmRelease ensures that the DCGM exporter Helm release is installed in the given namespace.
//
// The function checks if a release named "dcgm-exporter" already exists, and if not:
//   - Initializes Helm using the specified namespace and driver.
//   - Locates the DCGM exporter chart from NVIDIA's Helm repo.
//   - Installs the chart with default values (serviceMonitor disabled).
//
// This function requires Helm client libraries and permissions to install Helm releases
// in the target namespace. It is idempotent: if the release already exists, no action is taken.
//
// Returns an error if Helm initialization, chart loading, or installation fails.
func EnsureDCGMHelmRelease(ctx context.Context, namespace string) error {
	settings := cli.New()
	actionConfig := new(action.Configuration)
	if err := actionConfig.Init(settings.RESTClientGetter(), namespace, os.Getenv("HELM_DRIVER"), func(format string, v ...interface{}) {}); err != nil {
		return err
	}

	// Skip if release exists
	get := action.NewGet(actionConfig)
	if _, err := get.Run("dcgm-exporter"); err == nil {
		return nil
	}

	install := action.NewInstall(actionConfig)
	install.ReleaseName = "dcgm-exporter"
	install.Namespace = namespace
	install.CreateNamespace = true
	install.ChartPathOptions.RepoURL = "https://nvidia.github.io/dcgm-exporter/helm-charts"

	chartPath, err := install.ChartPathOptions.LocateChart("dcgm-exporter", settings)
	if err != nil {
		return fmt.Errorf("locate chart: %w", err)
	}

	chart, err := loader.Load(chartPath)
	if err != nil {
		return err
	}

	values := map[string]interface{}{
		"serviceMonitor": map[string]interface{}{"enabled": false},
		"image": map[string]interface{}{
			"repository": "nvcr.io/nvidia/k8s/dcgm-exporter",
			"tag":        "4.5.2-4.8.1-distroless", // pin exact version
			"pullPolicy": "IfNotPresent",
		},
	}

	if _, err := install.Run(chart, values); err != nil {
		return fmt.Errorf("helm install failed: %w", err)
	}

	return nil
}

// WaitForDCGMPods waits until at least one DCGM exporter pod is running in the cluster.
//
// The function polls the Kubernetes API, listing pods labeled "app=dcgm-exporter"
// in the cluster (or namespace provided in the client). It returns as soon as a
// pod reaches the Running phase, or an error if the timeout elapses.
//
// Parameters:
//   - ctx: context for cancellation and deadlines.
//   - c: Kubernetes client used to list pods.
//   - timeout: maximum duration to wait for a pod to become running.
//
// Returns an error if the timeout is exceeded or if there is a failure listing pods.
func WaitForDCGMPods(ctx context.Context, c client.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods := &corev1.PodList{}
		if err := c.List(ctx, pods, client.MatchingLabels{"app": "dcgm-exporter"}); err != nil {
			return err
		}
		for _, pod := range pods.Items {
			if pod.Status.Phase == corev1.PodRunning {
				return nil
			}
		}
		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("timeout waiting for dcgm pods")
}
