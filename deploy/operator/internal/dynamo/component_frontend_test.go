/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestFrontendDefaultsStartupProbeVersionGate(t *testing.T) {
	tests := []struct {
		name           string
		runtimeVersion *runtimeversion.Version
		wantEnabled    bool
	}{
		{name: "unknown legacy runtime"},
		{name: "older runtime", runtimeVersion: &runtimeversion.Version{Major: 1, Minor: 5, Patch: 9}},
		{name: "minimum supported runtime", runtimeVersion: &runtimeversion.Version{Major: 1, Minor: 6}, wantEnabled: true},
		{name: "newer runtime", runtimeVersion: &runtimeversion.Version{Major: 2}, wantEnabled: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Render frontend defaults for the resolved runtime version")
			container, err := NewFrontendDefaults().GetBaseContainer(ComponentContext{RuntimeVersion: tt.runtimeVersion})
			require.NoError(t, err)

			t.Log("Check the version-gated startup allowance")
			if tt.wantEnabled {
				require.Equal(t, &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{
						Path: "/live", Port: intstr.FromString("http"),
					}},
					PeriodSeconds: 10, TimeoutSeconds: 1, FailureThreshold: 30,
				}, container.StartupProbe)
			} else {
				require.Nil(t, container.StartupProbe)
			}
		})
	}
}

func TestFrontendSidecarStartupProbeOverride(t *testing.T) {
	t.Log("Provide a TCP startup probe for a frontend sidecar on runtime 1.6.0")
	override := &corev1.Probe{
		ProbeHandler:  corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(8000)}},
		PeriodSeconds: 5, FailureThreshold: 90,
	}
	pod := &corev1.PodSpec{Containers: []corev1.Container{{Name: "frontend", StartupProbe: override.DeepCopy()}}}

	t.Log("Merge defaults and check that the user probe replaces the entire default probe")
	err := mergeFrontendSidecarDefaults(pod, "frontend", ComponentContext{
		RuntimeVersion: &runtimeversion.Version{Major: 1, Minor: 6},
	}, &configv1alpha1.OperatorConfiguration{}, nil)
	require.NoError(t, err)
	require.Equal(t, override, pod.Containers[0].StartupProbe)
}
