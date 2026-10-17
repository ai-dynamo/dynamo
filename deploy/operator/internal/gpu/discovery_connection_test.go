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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestScrapeMetricsEndpointClosesConnections(t *testing.T) {
	t.Log("Exercise scrapes whose fully read responses leave reusable connections")
	const metrics = `# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",modelName="H100-SXM5-80GB",Hostname="gpu-node"} 50`
	for _, tc := range []struct {
		name        string
		body        string
		errContains string
	}{
		{name: "success", body: metrics},
		{name: "invalid exposition", body: "not valid prometheus exposition\n", errContains: "parse prometheus metrics"},
		{name: "missing GPU metrics", body: "\n", errContains: "no GPUs detected"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Start a keep-alive metrics server and track its open TCP connections")
			var active, accepted atomic.Int32
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintln(w, tc.body)
			}))
			server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
				switch state {
				case http.StateNew:
					accepted.Add(1)
					active.Add(1)
				case http.StateClosed:
					active.Add(-1)
				}
			}
			server.Start()
			t.Cleanup(server.Close)

			t.Log("Repeat scrapes with their individual transports and verify their results")
			for range 3 {
				info, err := ScrapeMetricsEndpoint(t.Context(), server.URL)
				if tc.errContains == "" {
					require.NoError(t, err)
					require.Equal(t, 1, info.GPUsPerNode)
				} else {
					require.ErrorContains(t, err, tc.errContains)
				}
			}

			t.Log("Verify every scrape released its connection without stopping the server")
			require.EqualValues(t, 3, accepted.Load())
			require.Eventually(t, func() bool {
				return active.Load() == 0
			}, time.Second, 10*time.Millisecond, "connections remain open after completed scrapes")
		})
	}
}
