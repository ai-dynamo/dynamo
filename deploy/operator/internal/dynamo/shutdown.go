/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// validateShutdownBudget validates worker budgets on the final pod spec,
// using the runtime's 30-second default when no literal override is present.
// podSpec must be non-nil. Runtime-only settings (envFrom, image ENV, command-line
// exports) cannot be resolved here; deployments using them must also declare a
// literal total budget in the pod template to validate their effective value.
func validateShutdownBudget(podSpec *corev1.PodSpec, isWorker bool) error {
	// Kubernetes counts preStop execution against the pod grace period too.
	grace := int64(30)
	if podSpec.TerminationGracePeriodSeconds != nil {
		grace = *podSpec.TerminationGracePeriodSeconds
	}

	// Validate after all container overrides have merged.
	for index, container := range podSpec.Containers {
		var canonical, legacy *corev1.EnvVar
		for i := range container.Env {
			env := &container.Env[i]
			switch env.Name {
			case "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS":
				canonical = env
			case "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT":
				legacy = env
			}
		}

		// Match runtime precedence: a non-empty canonical value wins.
		selected := legacy
		if canonical != nil && (canonical.ValueFrom != nil || strings.TrimSpace(canonical.Value) != "") {
			selected = canonical
		}
		// Only the primary worker inherits the worker default. Unrelated
		// containers and non-worker components require an explicit setting.
		if (selected == nil || (selected.ValueFrom == nil && strings.TrimSpace(selected.Value) == "")) && !(isWorker && index == 0) {
			continue
		}
		if selected != nil && selected.ValueFrom != nil {
			return fmt.Errorf("container %s: %s must be a literal to validate the shutdown budget", container.Name, selected.Name)
		}
		// Reject invalid configuration rather than silently validating a fallback.
		total := int64(30)
		if selected != nil && strings.TrimSpace(selected.Value) != "" {
			var err error
			total, err = strconv.ParseInt(strings.TrimSpace(selected.Value), 10, 64)
			if err != nil || total <= 0 || total > 315360000 {
				return fmt.Errorf("container %s: invalid %s=%q", container.Name, selected.Name, selected.Value)
			}
		}
		if grace < total+5 {
			return fmt.Errorf("container %s: terminationGracePeriodSeconds=%d must be at least %d (shutdown total plus 5s margin); include additional time for preStop hooks", container.Name, grace, total+5)
		}
	}
	return nil
}
