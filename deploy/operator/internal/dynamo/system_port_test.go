// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

func TestSystemPortRendering(t *testing.T) {
	for _, tt := range []struct {
		name         string
		kind         v1beta1.ComponentType
		value        string
		global       string
		source       bool
		host         bool
		native       bool
		ports        []corev1.ContainerPort
		probe        *corev1.Probe
		wantPort     int32
		wantErr      bool
		wantNoProbes bool
	}{
		{name: "worker-default", kind: v1beta1.ComponentTypeWorker, wantPort: 9090},
		{name: "runtime-default-clear-ports", kind: v1beta1.ComponentTypeWorker, native: true, ports: []corev1.ContainerPort{}, wantPort: 9090},
		{name: "mapped-host-port", kind: v1beta1.ComponentTypeWorker, value: "12345", ports: []corev1.ContainerPort{{Name: "system", ContainerPort: 9090, HostPort: 23456}}, wantPort: 12345},
		{name: "worker-custom", kind: v1beta1.ComponentTypeWorker, value: "12345", wantPort: 12345},
		{name: "planner-custom", kind: v1beta1.ComponentTypePlanner, value: "12345", wantPort: 12345},
		{name: "router-custom", kind: v1beta1.ComponentType(""), value: "12345", wantPort: 12345},
		{name: "router-default-port", kind: v1beta1.ComponentType(""), value: "9090", wantPort: 9090},
		{name: "global-custom", kind: v1beta1.ComponentTypeWorker, global: "12345", wantPort: 12345},
		{name: "component-precedence", kind: v1beta1.ComponentTypeWorker, global: "12346", value: "12345", wantPort: 12345},
		{name: "host-network", kind: v1beta1.ComponentTypeWorker, host: true, value: "12345", ports: []corev1.ContainerPort{{Name: "system", ContainerPort: 9090, HostPort: 9090}, {Name: "other", ContainerPort: 23456}}, wantPort: 12345},
		{name: "runtime-sidecar", kind: v1beta1.ComponentTypeWorker, native: true, host: true, value: "12345", wantPort: 12345},
		{name: "runtime-sidecar-clear-ports", kind: v1beta1.ComponentTypeWorker, native: true, value: "12345", ports: []corev1.ContainerPort{}, wantPort: 0},
		{name: "clear-ports", kind: v1beta1.ComponentTypeWorker, value: "12345", ports: []corev1.ContainerPort{}, wantPort: 0},
		{name: "add-to-other-ports", kind: v1beta1.ComponentType(""), value: "12345", ports: []corev1.ContainerPort{{Name: "http", ContainerPort: 8000}}, wantPort: 12345},
		{name: "source-with-port", kind: v1beta1.ComponentTypeWorker, source: true, ports: []corev1.ContainerPort{{Name: "system", ContainerPort: 12345}}, wantPort: 12345},
		{name: "source-without-port", kind: v1beta1.ComponentTypeWorker, source: true, wantErr: true},
		{name: "expansion-with-port", kind: v1beta1.ComponentTypeWorker, value: "$(PORT)", ports: []corev1.ContainerPort{{Name: "system", ContainerPort: 12345}}, wantPort: 12345},
		{name: "expansion-without-port", kind: v1beta1.ComponentTypeWorker, value: "$(PORT)", wantErr: true},
		{name: "hexadecimal", kind: v1beta1.ComponentTypeWorker, value: "0x3039", wantPort: 12345},
		{name: "ephemeral", kind: v1beta1.ComponentTypeWorker, value: "0", wantPort: 0, wantNoProbes: true},
		{name: "disabled", kind: v1beta1.ComponentTypeWorker, value: "-1", wantPort: 0, wantNoProbes: true},
		{name: "invalid", kind: v1beta1.ComponentTypeWorker, value: "invalid", wantErr: true},
		{name: "out-of-range", kind: v1beta1.ComponentTypeWorker, value: "32768", wantErr: true},
		{name: "authored-probe", kind: v1beta1.ComponentTypeWorker, value: "12345", probe: &corev1.Probe{HTTPGet: &corev1.HTTPGetAction{Path: "/engine-health", Port: intstr.FromInt32(23456)}}, wantPort: 12345},
		{name: "disabled-authored-probe", kind: v1beta1.ComponentTypeWorker, value: "-1", probe: &corev1.Probe{Exec: &corev1.ExecAction{Command: []string{"true"}}}, wantPort: 0},
		{name: "frontend-independent", kind: v1beta1.ComponentTypeFrontend, value: "12345", wantPort: 0},
		{name: "epp-independent", kind: v1beta1.ComponentTypeEPP, value: "12345", wantPort: 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Configure the runtime environment independently of port declarations")
			authored := corev1.Container{Name: "main", Image: "runtime:1.6.0", Ports: tt.ports, ReadinessProbe: tt.probe}
			if tt.value != "" {
				authored.Env = []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", Value: tt.value}}
			}
			if tt.source {
				authored.Env = []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "ports", Key: "system"}}}}
			}
			ps := corev1.PodSpec{HostNetwork: tt.host, Containers: []corev1.Container{authored}}
			if tt.native {
				authored.Name = "runtime"
				authored.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
				ps.InitContainers = []corev1.Container{authored}
				ps.Containers = []corev1.Container{{Name: "main", Image: "engine:1.6.0", Command: []string{"vllm-rs"}}}
			}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: tt.kind, PodTemplate: &corev1.PodTemplateSpec{Spec: ps}}
			graph := &v1beta1.DynamoGraphDeployment{Name: "system-port", Namespace: "default"}
			if tt.global != "" {
				graph.Spec.Env = []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", Value: tt.global}}
			}
			original := component.DeepCopy()

			t.Log("Render using the real graph and container defaulting path")
			pod, err := GeneratePodSpecForComponent(component, BackendFrameworkNoop, nil, graph, RoleMain, 1, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeLWS, "worker", nil, staticContainerGPUCount(0))
			require.Equal(t, original, component)
			if tt.wantErr {
				require.ErrorContains(t, err, "DYN_SYSTEM_PORT")
				return
			}
			require.NoError(t, err)
			runtime := &pod.Containers[0]
			if tt.native {
				runtime = &pod.InitContainers[0]
			}

			t.Log("Check listener declarations and preserve independent configuration")
			port := findContainerPort(runtime, "system")
			if tt.wantPort == 0 {
				require.Nil(t, port)
			} else {
				require.NotNil(t, port)
				require.Equal(t, tt.wantPort, port.ContainerPort)
				if tt.host {
					require.Equal(t, tt.wantPort, port.HostPort)
				}
			}
			if tt.ports != nil && len(tt.ports) == 0 && tt.value != "" {
				require.Empty(t, runtime.Ports)
			}
			if !tt.host && tt.ports != nil && len(tt.ports) > 0 {
				for _, authoredPort := range tt.ports {
					if authoredPort.Name == "system" {
						require.Equal(t, authoredPort.HostPort, port.HostPort)
					}
				}
			}
			if tt.wantNoProbes {
				require.Nil(t, runtime.StartupProbe)
				require.Nil(t, runtime.ReadinessProbe)
				require.Nil(t, runtime.LivenessProbe)
			}
			if tt.probe != nil {
				require.Equal(t, tt.probe, runtime.ReadinessProbe)
			}
			if tt.source {
				require.NotNil(t, findEnvVar(runtime.Env, "DYN_SYSTEM_PORT").ValueFrom)
			}
			if tt.native {
				require.Nil(t, findContainerPort(&pod.Containers[0], "system"))
			}
		})
	}
}

func TestElasticEPSystemPortFormats(t *testing.T) {
	t.Log("Start a leader health endpoint on an available custom port")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/live" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	address, err := url.Parse(server.URL)
	require.NoError(t, err)
	port, err := strconv.Atoi(address.Port())
	require.NoError(t, err)
	container := corev1.Container{Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray"}}
	require.True(t, injectElasticEPRayLaunchFlags(&container, RoleWorker, "worker", &LWSMultinodeDeployer{}))
	_, script, ok := strings.Cut(container.Args[0], `python3 -c "`)
	require.True(t, ok)
	script, _, ok = strings.Cut(script, `"`)
	require.True(t, ok)
	script = strings.ReplaceAll(script, "$(LWS_LEADER_ADDRESS)", address.Hostname())
	for _, value := range []string{address.Port(), "0" + address.Port(), fmt.Sprintf("0x%x", port), " " + address.Port() + " "} {
		t.Run(value, func(t *testing.T) {
			t.Log("Execute the generated Python health check with a resolved environment value")
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			command := exec.CommandContext(ctx, "python3", "-c", script)
			command.Env = append(os.Environ(), "DYN_SYSTEM_PORT="+value)
			output, err := command.CombinedOutput()
			require.NoError(t, err, string(output))
		})
	}
}
