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
		{"existing template unchanged", 10, nil, false},
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
			if err := validateShutdownBudget(&pod); (err != nil) != tc.wantError {
				t.Fatalf("validation error = %v, want error = %v", err, tc.wantError)
			}
		})
	}
}
