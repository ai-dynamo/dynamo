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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
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
	tests := []struct {
		name    string
		handler corev1.ProbeHandler
		wantErr bool
	}{
		{name: "timing only", wantErr: true},
		{
			name: "multiple handlers",
			handler: corev1.ProbeHandler{
				HTTPGet:   &corev1.HTTPGetAction{Path: "/live", Port: intstr.FromInt(8000)},
				TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(8000)},
			},
			wantErr: true,
		},
		{name: "exec", handler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}}},
		{name: "http", handler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/ready", Port: intstr.FromInt(8000)}}},
		{name: "tcp", handler: corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(8000)}}},
		{name: "grpc", handler: corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: 8000}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Provide a frontend sidecar startup probe on runtime 1.6.0")
			override := &corev1.Probe{ProbeHandler: tt.handler, PeriodSeconds: 5, FailureThreshold: 90}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				RuntimeVersionOverride: "1.6.0", FrontendSidecar: ptr.To("frontend"),
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{
					{Name: commonconsts.MainContainerName, Image: "worker:1.6.0"},
					{Name: "frontend", Image: "frontend:1.6.0", StartupProbe: override.DeepCopy()},
				}}},
			}

			t.Log("Render the complete pod specification through the production entry point")
			pod, err := GenerateBasePodSpec(
				component, BackendFrameworkVLLM, nil, "test-dgd", "test-ns", RoleMain, 1,
				&configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove,
				"worker", nil, staticContainerGPUCount(0),
			)

			t.Log("Reject invalid handlers and preserve the complete valid override")
			if tt.wantErr {
				require.EqualError(t, err, `frontend sidecar "frontend" startupProbe must define exactly one handler`)
				require.Nil(t, pod)
				return
			}
			require.NoError(t, err)
			require.Len(t, pod.Containers, 2)
			require.Equal(t, "frontend", pod.Containers[1].Name)
			require.Equal(t, override, pod.Containers[1].StartupProbe)
		})
	}
}
