// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pagebroker

import (
	"bytes"
	"context"
	"errors"
	"net"
	"path/filepath"
	"testing"
	"time"
)

func TestRequestEncoding(t *testing.T) {
	got := request(1, "tx-1", "/checkpoint")
	want := []byte{0x08, 0x01, 0x12, 0x04, 't', 'x', '-', '1', 0x1a, 0x0b, '/', 'c', 'h', 'e', 'c', 'k', 'p', 'o', 'i', 'n', 't'}
	if !bytes.Equal(got, want) {
		t.Fatalf("request = %x, want %x", got, want)
	}
}

func TestClientSocket(t *testing.T) {
	if got := NewClient("").socket; got != defaultSocket {
		t.Fatalf("default socket = %q, want %q", got, defaultSocket)
	}
	if got := NewClient("/tmp/pagebroker.sock").socket; got != "/tmp/pagebroker.sock" {
		t.Fatalf("socket = %q", got)
	}
}

func TestVarintBounds(t *testing.T) {
	max := []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}
	if value, size := read(max); value != ^uint64(0) || size != len(max) {
		t.Fatalf("max varint = (%d, %d)", value, size)
	}
	overflow := append([]byte(nil), max...)
	overflow[len(overflow)-1] = 0x02
	if _, size := read(overflow); size != 0 {
		t.Fatalf("overflow varint size = %d", size)
	}
}

func TestCallCancellation(t *testing.T) {
	listener, err := net.Listen("unixpacket", filepath.Join(t.TempDir(), "pagebroker.sock"))
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	accepted := make(chan net.Conn, 1)
	go func() {
		connection, err := listener.Accept()
		if err == nil {
			accepted <- connection
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		_, err := call(ctx, listener.Addr().String(), 1, "tx-1", "/checkpoint")
		result <- err
	}()

	connection := <-accepted
	defer connection.Close()
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("call error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("call did not stop after context cancellation")
	}
}

func TestResponseParsing(t *testing.T) {
	encoded := append([]byte{0x08, 0x01}, field(3, []byte("/staging"))...)
	encoded = append(encoded, 0x51, 0, 0, 0, 0, 0, 0, 0, 0)
	encoded = append(encoded, 0x5d, 0, 0, 0, 0)
	encoded = append(encoded, 0x63, 0x08, 0x01, 0x64)
	encoded = append(encoded, field(4, []byte("/scratch"))...)

	got, err := parse(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !got.ok || got.staging != "/staging" || got.scratch != "/scratch" || got.err != "" {
		t.Fatalf("response = %+v", got)
	}
}
