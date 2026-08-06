// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pagebroker

import (
	"bytes"
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRequestEncoding(t *testing.T) {
	got := request(1, "tx-1", "/checkpoint", true)
	want := []byte{0x08, 0x01, 0x12, 0x04, 't', 'x', '-', '1', 0x1a, 0x0b, '/', 'c', 'h', 'e', 'c', 'k', 'p', 'o', 'i', 'n', 't', 0x20, 0x01}
	if !bytes.Equal(got, want) {
		t.Fatalf("request = %x, want %x", got, want)
	}
}

func TestDirectRequestDoesNotStage(t *testing.T) {
	got := request(1, "tx-1", "/checkpoint", false)
	if bytes.Contains(got, []byte{0x20, 0x01}) {
		t.Fatalf("direct request includes staging flag: %x", got)
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
		_, err := call(ctx, listener.Addr().String(), 1, "tx-1", "/checkpoint", false)
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

func TestStageKeepsSourcePathAndProviderConnection(t *testing.T) {
	root := t.TempDir()
	checkpoint := filepath.Join(root, "checkpoint")
	if err := os.Mkdir(checkpoint, 0755); err != nil {
		t.Fatal(err)
	}
	scratch := filepath.Join(root, "scratch")
	staged := filepath.Join(root, "staged")
	socket := filepath.Join(root, "pagebroker.sock")
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: socket, Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	go func() {
		connection, err := listener.AcceptUnix()
		if err != nil {
			return
		}
		defer connection.Close()
		buffer := make([]byte, 1024)
		_, _ = connection.Read(buffer)
		response := append(append([]byte{0x08, 0x01}, field(3, []byte(staged))...), field(4, []byte(scratch))...)
		_, _ = connection.Write(response)
		_, _ = connection.Write([]byte("provider-ready"))
	}()

	transaction, err := Stage(context.Background(), socket, checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	if transaction.StagingPath() != checkpoint {
		t.Fatalf("restore path = %q, want %q", transaction.StagingPath(), checkpoint)
	}
	files, err := transaction.Files()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, file := range files {
			_ = file.Close()
		}
	}()
	if len(files) != 3 {
		t.Fatalf("inherited files = %d, want provider, image, work", len(files))
	}
	providerMessage := make([]byte, 32)
	n, err := files[0].Read(providerMessage)
	if err != nil {
		t.Fatal(err)
	}
	if string(providerMessage[:n]) != "provider-ready" {
		t.Fatalf("provider message = %q", providerMessage[:n])
	}
}
