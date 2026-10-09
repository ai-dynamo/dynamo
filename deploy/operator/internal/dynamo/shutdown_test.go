/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestValidateShutdownBudget(t *testing.T) {
	// Regression: pod overrides can let Kubernetes kill a worker before its deadline.
	cases := []struct {
		name      string
		grace     int64
		env       []corev1.EnvVar
		wantError bool
	}{
		{"default total exceeds grace", 10, nil, true},
		{"default total fits", 35, nil, false},
		{"empty total uses default", 10, []corev1.EnvVar{{Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", Value: " "}}, true},
		{"exact margin", 35, []corev1.EnvVar{{Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", Value: "30"}}, false},
		{"insufficient margin", 34, []corev1.EnvVar{{Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", Value: "30"}}, true},
		{"legacy alias", 35, []corev1.EnvVar{{Name: "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", Value: " 30 "}}, false},
		{"canonical wins", 35, []corev1.EnvVar{{Name: "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", Value: "100"}, {Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", Value: "30"}}, false},
		{"invalid total", 60, []corev1.EnvVar{{Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", Value: "NaN"}}, true},
		{"unresolved total", 60, []corev1.EnvVar{{Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", ValueFrom: &corev1.EnvVarSource{}}}, true},
	}

	// Exercise the effective pod contract without mutating user overrides.
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Validate the final pod grace period against the configured worker total")
			pod := corev1.PodSpec{TerminationGracePeriodSeconds: ptr.To(tc.grace), Containers: []corev1.Container{{Name: "worker", Env: tc.env}}}
			if err := validateShutdownBudget(&pod, true); (err != nil) != tc.wantError {
				t.Fatalf("validation error = %v, want error = %v", err, tc.wantError)
			}
		})
	}
	// A logging sidecar must not impose the worker default on an explicitly
	// shorter worker budget; non-worker components do not inherit it either.
	pod := corev1.PodSpec{TerminationGracePeriodSeconds: ptr.To[int64](10), Containers: []corev1.Container{
		{Name: "worker", Env: []corev1.EnvVar{{Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", Value: "5"}}},
		{Name: "logger"},
	}}
	if err := validateShutdownBudget(&pod, true); err != nil {
		t.Fatal(err)
	}
	pod.Containers = []corev1.Container{{Name: "frontend"}}
	if err := validateShutdownBudget(&pod, false); err != nil {
		t.Fatal(err)
	}
}
