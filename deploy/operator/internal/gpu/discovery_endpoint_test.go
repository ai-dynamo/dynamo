/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package gpu

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestBuildDCGMEndpoint(t *testing.T) {
	// Exercise the default host and custom paths for both address families.
	for _, tc := range []struct {
		name     string
		podIP    string
		template string
		want     string
	}{
		{name: "default IPv4", podIP: "10.0.0.1", want: "http://10.0.0.1:9400/metrics"},
		{name: "default IPv6", podIP: "fd00::1234", want: "http://[fd00::1234]:9400/metrics"},
		{name: "custom IPv4", podIP: "10.0.0.1", template: "https://{POD_IP}:9443/custom/metrics", want: "https://10.0.0.1:9443/custom/metrics"},
		{name: "custom IPv6", podIP: "fd00::1234", template: "https://{POD_IP}:9443/custom/metrics", want: "https://[fd00::1234]:9443/custom/metrics"},
		{name: "already bracketed IPv6", podIP: "fd00::1234", template: "http://[{POD_IP}]:9400/metrics", want: "http://[fd00::1234]:9400/metrics"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("DCGM_METRICS_ENDPOINT_TEMPLATE", tc.template)
			require.Equal(t, tc.want, buildDCGMEndpoint(tc.podIP))
		})
	}
}

func TestDiscoverGPUsFromDCGMIPv6(t *testing.T) {
	t.Log("Start a DCGM metrics endpoint on an IPv6 loopback listener")
	listener, err := net.Listen("tcp6", "[::1]:0")
	require.NoError(t, err)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintln(w, `# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",modelName="H100-SXM5-80GB",Hostname="gpu-node"} 50`)
	}))
	require.NoError(t, server.Listener.Close())
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)
	t.Setenv("DCGM_METRICS_ENDPOINT_TEMPLATE", fmt.Sprintf("http://{POD_IP}:%d/metrics", listener.Addr().(*net.TCPAddr).Port))

	t.Log("Discover a running exporter pod and scrape its IPv6 Pod IP")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcgm-exporter",
			Namespace: "default",
			Labels:    map[string]string{LabelApp: LabelValueDCGMExporter},
		},
		Spec: corev1.PodSpec{NodeName: "gpu-node"},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: "::1",
		},
	}
	info, err := NewGPUDiscovery(ScrapeMetricsEndpoint).DiscoverGPUsFromDCGM(t.Context(), newFakeClient(pod), nil)

	t.Log("Verify the inventory was parsed from the exporter response")
	require.NoError(t, err)
	require.Equal(t, "H100-SXM5-80GB", info.Model)
	require.Equal(t, 1, info.GPUsPerNode)
	require.Equal(t, 1, info.NodesWithGPUs)
}
