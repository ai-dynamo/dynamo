/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"context"
	"net"
	"testing"
)

func TestWaitForProxySignal(t *testing.T) {
	t.Log("Connect the local bridge to a simulated proxy control connection")
	bridge, proxy := net.Pipe()
	defer func() { _ = bridge.Close() }()
	defer func() { _ = proxy.Close() }()
	go func() {
		_, _ = proxy.Write([]byte{2})
	}()

	t.Log("Accept the expected proxy protocol signal")
	if err := waitForProxySignal(context.Background(), bridge, 2); err != nil {
		t.Fatalf("wait for proxy signal: %v", err)
	}
}

func TestWaitForProxySignalRejectsUnexpectedByte(t *testing.T) {
	t.Log("Send a proxy protocol signal for a different bridge state")
	bridge, proxy := net.Pipe()
	defer func() { _ = bridge.Close() }()
	defer func() { _ = proxy.Close() }()
	go func() {
		_, _ = proxy.Write([]byte{1})
	}()

	t.Log("Reject the signal instead of treating the tunnel as ready")
	if err := waitForProxySignal(context.Background(), bridge, 2); err == nil {
		t.Fatal("unexpected proxy signal was accepted")
	}
}
