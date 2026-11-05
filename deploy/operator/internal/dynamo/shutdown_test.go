/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestShutdownBudgetWarningsPreserveExistingPodSpecs(t *testing.T) {
	// An operator upgrade must still render workloads with previously accepted budgets.
	for _, tc := range []struct {
		name  string
		grace *int64
		env   []corev1.EnvVar
	}{
		{name: "defaults"},
		{name: "explicit thirty seconds", grace: ptr.To[int64](30)},
		{name: "legacy sixty seconds", env: []corev1.EnvVar{{Name: "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", Value: "60"}}},
		{name: "config map total", env: []corev1.EnvVar{{Name: "DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "settings"}, Key: "timeout"}}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						TerminationGracePeriodSeconds: tc.grace,
						Containers:                    []corev1.Container{{Name: commonconsts.MainContainerName, Image: "worker:test", Env: tc.env}},
					},
				},
			}
			pod, err := GenerateBasePodSpec(component, BackendFrameworkVLLM, &mockSecretsRetriever{}, "dgd", "default", RoleMain, 1, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			// Diagnostics must not change the grace period or the configured environment.
			wantGrace := int64(60)
			if tc.grace != nil {
				wantGrace = *tc.grace
			}
			require.Equal(t, wantGrace, *pod.TerminationGracePeriodSeconds)
			for _, env := range tc.env {
				require.Contains(t, pod.Containers[0].Env, env)
			}
		})
	}
}

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
