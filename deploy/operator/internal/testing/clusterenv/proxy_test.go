/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"context"
	"errors"
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

func TestProxyStopAfterPortForwardExited(t *testing.T) {
	t.Log("Represent a port-forward process whose error was already observed by startup")
	forwardStop := make(chan struct{})
	forwardDone := make(chan struct{})
	forwardErr := errors.New("port-forward failed")
	close(forwardDone)
	proxy := &proxyRuntime{
		forwardStop: forwardStop,
		forwardDone: forwardDone,
		forwardErr:  forwardErr,
	}

	t.Log("Stop without waiting for a second delivery of the process result")
	err := proxy.stop()
	if !errors.Is(err, forwardErr) {
		t.Fatalf("stop proxy error = %v, want %v", err, forwardErr)
	}
	select {
	case <-forwardStop:
	default:
		t.Fatal("port-forward stop channel was not closed")
	}
}
