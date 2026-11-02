// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func noEnv(string) (string, bool) { return "", false }

// syncBuffer is a bytes.Buffer safe for the server's concurrent logging.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func freePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer l.Close()
	return l.Addr().String()
}

func TestRunConfigErrors(t *testing.T) {
	var out syncBuffer
	assert.Equal(t, 2, run(context.Background(), []string{"--store=etcd"}, noEnv, &out))
	assert.Contains(t, out.String(), "mcs-registry: --store")
	assert.Equal(t, 0, run(context.Background(), []string{"--help"}, noEnv, &out))
	assert.Contains(t, out.String(), "Usage of mcs-registry")
}

func TestRunDynamoDBNeedsRegion(t *testing.T) {
	t.Setenv("AWS_REGION", "")
	var out syncBuffer
	assert.Equal(t, 1, run(context.Background(), []string{"--store=dynamodb"}, noEnv, &out))
	assert.Contains(t, out.String(), "no AWS region")
}

func TestRunServesUntilCanceled(t *testing.T) {
	grpcAddr, httpAddr, adminAddr := freePort(t), freePort(t), freePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	var out syncBuffer
	done := make(chan int, 1)
	go func() {
		done <- run(ctx, []string{
			"--grpc-listen-address=" + grpcAddr, "--http-listen-address=" + httpAddr, "--admin-listen-address=" + adminAddr,
			"--shutdown-drain=0s", "--refresh-interval=200ms", "--log-level=debug",
		}, noEnv, &out)
	}()
	require.Eventually(t, func() bool {
		resp, err := http.Get(fmt.Sprintf("http://%s/readyz", adminAddr))
		if err != nil {
			return false
		}
		_ = resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 10*time.Second, 20*time.Millisecond)
	resp, err := http.Get(fmt.Sprintf("http://%s/v1/clustersets", httpAddr))
	require.NoError(t, err)
	_ = resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	cancel()
	select {
	case code := <-done:
		assert.Equal(t, 0, code)
	case <-time.After(10 * time.Second):
		t.Fatal("mcs-registry did not stop")
	}
	logs := out.String()
	assert.Contains(t, logs, `"msg":"starting mcs-registry"`)
	assert.Contains(t, logs, `"msg":"mcs-registry stopped"`)
	assert.True(t, strings.Contains(logs, `"service":"global-discovery-mcs"`))
}

func TestRunListenError(t *testing.T) {
	busy, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer busy.Close()
	var out syncBuffer
	code := run(context.Background(), []string{
		"--grpc-listen-address=" + busy.Addr().String(),
		"--http-listen-address=" + freePort(t), "--admin-listen-address=" + freePort(t),
	}, noEnv, &out)
	assert.Equal(t, 1, code)
	assert.Contains(t, out.String(), "start failed")
}

func TestNewLogger(t *testing.T) {
	_, err := newLogger("loud", &bytes.Buffer{})
	assert.Error(t, err)
}
