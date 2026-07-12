/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package cuda

import (
	"context"
	"encoding/binary"
	"errors"
	"net"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func daemonTestResponse(status int32, output, errorOutput string) []byte {
	packet := make([]byte, daemonResponseHeader+len(output)+len(errorOutput))
	binary.LittleEndian.PutUint32(packet[0:4], daemonProtocolMagic)
	binary.LittleEndian.PutUint16(packet[4:6], daemonProtocolVersion)
	binary.LittleEndian.PutUint16(packet[6:8], daemonResponseHeader)
	binary.LittleEndian.PutUint32(packet[8:12], uint32(status))
	binary.LittleEndian.PutUint32(packet[16:20], uint32(len(output)))
	binary.LittleEndian.PutUint32(packet[20:24], uint32(len(errorOutput)))
	copy(packet[daemonResponseHeader:], output)
	copy(packet[daemonResponseHeader+len(output):], errorOutput)
	return packet
}

func runDaemonHealthTestServer(t *testing.T, handler func(*net.UnixConn)) string {
	t.Helper()
	socket := filepath.Join(t.TempDir(), "helper.sock")
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: socket + ".health", Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go func() {
		conn, err := listener.AcceptUnix()
		if err == nil {
			handler(conn)
			_ = conn.Close()
		}
	}()
	return socket
}

func TestCommandRunnerUsesDaemonForAllFourActions(t *testing.T) {
	for _, action := range []string{actionLock, actionCheckpoint, actionRestore, actionUnlock} {
		t.Run(action, func(t *testing.T) {
			requestAction := make(chan uint16, 1)
			socket := runDaemonTestServer(t, func(conn *net.UnixConn) {
				request := make([]byte, daemonMaxRequest)
				read, err := conn.Read(request)
				if err != nil {
					t.Error(err)
					return
				}
				requestAction <- binary.LittleEndian.Uint16(request[8:10])
				_, _ = conn.Write(daemonTestResponse(0, "", ""))
				_ = read
			})
			storageDir := "/checkpoints/process-0000"
			if action == actionLock || action == actionUnlock {
				storageDir = ""
			}
			err := (commandHelperActionRunner{}).run(
				context.Background(), 42, action, "", types.CUDAStorageModePOSIX,
				storageDir, daemonTestSettings(socket), testDaemonIdentity(42), logr.Discard(),
			)
			if err != nil {
				t.Fatalf("run(%q) error = %v", action, err)
			}
			want := map[string]uint16{
				actionLock: daemonActionLock, actionCheckpoint: daemonActionCheckpoint,
				actionRestore: daemonActionRestore, actionUnlock: daemonActionUnlock,
			}[action]
			if got := <-requestAction; got != want {
				t.Fatalf("run(%q) daemon action = %d, want %d", action, got, want)
			}
		})
	}
}

func TestDaemonRequestActionsCarryIdentityWithoutTransferArguments(t *testing.T) {
	identity := testDaemonIdentity(42)
	for action, want := range map[string]uint16{
		actionLock:       daemonActionLock,
		actionCheckpoint: daemonActionCheckpoint,
		actionRestore:    daemonActionRestore,
		actionUnlock:     daemonActionUnlock,
	} {
		storageDir := "/checkpoints/process-0000"
		if action == actionLock || action == actionUnlock {
			storageDir = ""
		}
		packet, err := daemonRequest(
			42, action, "", storageDir, daemonTestSettings("/unused.sock"), identity,
		)
		if err != nil {
			t.Fatalf("daemonRequest(%q) error = %v", action, err)
		}
		if got := binary.LittleEndian.Uint16(packet[8:10]); got != want {
			t.Fatalf("daemonRequest(%q) action = %d, want %d", action, got, want)
		}
		if got := binary.LittleEndian.Uint64(packet[40:48]); got != identity.StartTimeTicks {
			t.Fatalf("daemonRequest(%q) starttime = %d, want %d", action, got, identity.StartTimeTicks)
		}
	}
}

func TestWaitForDaemonRetriesUntilProtocolHealthIsReady(t *testing.T) {
	socket := filepath.Join(t.TempDir(), "helper.sock")
	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)
		time.Sleep(150 * time.Millisecond)
		listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: socket + ".health", Net: "unixpacket"})
		if err != nil {
			t.Error(err)
			return
		}
		defer listener.Close()
		conn, err := listener.AcceptUnix()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		request := make([]byte, daemonMaxRequest)
		_, _ = conn.Read(request)
		response := daemonTestResponse(0, "ready\n", "")
		binary.LittleEndian.PutUint32(response[12:16], daemonCapabilityDeferredCUDA)
		_, _ = conn.Write(response)
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := WaitForDaemon(ctx, socket); err != nil {
		t.Fatalf("WaitForDaemon() error = %v", err)
	}
	<-serverDone
}

func TestRunDaemonActionSurfacesFatalResponse(t *testing.T) {
	socket := runDaemonTestServer(t, func(conn *net.UnixConn) {
		request := make([]byte, daemonMaxRequest)
		_, _ = conn.Read(request)
		response := daemonTestResponse(2, "", "partial transfer")
		binary.LittleEndian.PutUint32(response[12:16], daemonResponseFatal)
		_, _ = conn.Write(response)
	})
	err := runDaemonAction(
		context.Background(), 42, actionCheckpoint, "", "/checkpoints/process-0000",
		daemonTestSettings(socket), testDaemonIdentity(42), logr.Discard(),
	)
	if !errors.Is(err, errDaemonFatal) {
		t.Fatalf("runDaemonAction() error = %v, want fatal daemon state", err)
	}
}

func TestDaemonHealthyRequiresDeferredCUDACapability(t *testing.T) {
	for _, test := range []struct {
		name  string
		flags uint32
		ok    bool
	}{
		{name: "capable", flags: daemonCapabilityDeferredCUDA, ok: true},
		{name: "old protocol capability", flags: 0, ok: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			socket := runDaemonHealthTestServer(t, func(conn *net.UnixConn) {
				request := make([]byte, daemonMaxRequest)
				_, _ = conn.Read(request)
				response := daemonTestResponse(0, "ready\n", "")
				binary.LittleEndian.PutUint32(response[12:16], test.flags)
				_, _ = conn.Write(response)
			})
			err := daemonHealthy(context.Background(), socket)
			if (err == nil) != test.ok {
				t.Fatalf("daemonHealthy() error = %v, want success=%v", err, test.ok)
			}
		})
	}
}

func TestDaemonRequestCarriesProcessIdentity(t *testing.T) {
	identity := testDaemonIdentity(42)
	packet, err := daemonRequest(
		42,
		actionCheckpoint,
		"",
		"/checkpoints/process-0000",
		daemonTestSettings("/unused.sock"),
		identity,
	)
	if err != nil {
		t.Fatal(err)
	}
	cgroupSize := int(binary.LittleEndian.Uint32(packet[36:40]))
	if got := binary.LittleEndian.Uint64(packet[40:48]); got != identity.StartTimeTicks {
		t.Fatalf("start time = %d, want %d", got, identity.StartTimeTicks)
	}
	payloadOffset := daemonRequestHeader + len("/checkpoints/process-0000")
	if got := string(packet[payloadOffset : payloadOffset+cgroupSize]); got != identity.Cgroup {
		t.Fatalf("cgroup = %q, want %q", got, identity.Cgroup)
	}
}

func testDaemonIdentity(pid int) snapshotruntime.ProcessDetails {
	return snapshotruntime.ProcessDetails{
		ObservedPID:    pid,
		OutermostPID:   pid,
		InnermostPID:   pid,
		NamespacePIDs:  []int{pid},
		StartTimeTicks: 12345,
		Cgroup:         "0::/kubepods/test\n",
	}
}

func TestCommandRunnerFallsBackOnlyWhenDaemonUnavailableBeforeSend(t *testing.T) {
	settings := daemonTestSettings(filepath.Join(t.TempDir(), "missing.sock"))
	settings.DaemonFallback = true
	oldBinary := cudaCheckpointHelperBinary
	cudaCheckpointHelperBinary = "/bin/true"
	t.Cleanup(func() { cudaCheckpointHelperBinary = oldBinary })

	err := (commandHelperActionRunner{}).run(
		context.Background(), 42, actionCheckpoint, "", types.CUDAStorageModePOSIX,
		"/checkpoints/process-0000", settings, testDaemonIdentity(42), logr.Discard(),
	)
	if err != nil {
		t.Fatalf("configured pre-send fallback error = %v", err)
	}
}

func runDaemonTestServer(t *testing.T, handler func(*net.UnixConn)) string {
	t.Helper()
	socket := filepath.Join(t.TempDir(), "helper.sock")
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: socket, Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go func() {
		conn, err := listener.AcceptUnix()
		if err == nil {
			handler(conn)
			_ = conn.Close()
		}
	}()
	return socket
}

func daemonTestSettings(socket string) types.CUDATransferSettings {
	return types.CUDATransferSettings{
		BufferCount:  1,
		ChunkBytes:   types.DefaultCUDATransferChunkBytes,
		DaemonSocket: socket,
	}
}

func TestRunDaemonActionSuccess(t *testing.T) {
	socket := runDaemonTestServer(t, func(conn *net.UnixConn) {
		request := make([]byte, daemonMaxRequest)
		if _, err := conn.Read(request); err != nil {
			t.Error(err)
			return
		}
		_, _ = conn.Write(daemonTestResponse(0,
			`{"event":"cuda_custom_storage_transfer","helper_main_to_telemetry_seconds":0.000001}`+"\n", ""))
	})
	if err := runDaemonAction(
		context.Background(), 42, actionCheckpoint, "", "/checkpoints/process-0000",
		daemonTestSettings(socket), testDaemonIdentity(42), logr.Discard(),
	); err != nil {
		t.Fatalf("runDaemonAction() error = %v", err)
	}
}

func TestRunDaemonActionRejectsOversizedResponse(t *testing.T) {
	socket := runDaemonTestServer(t, func(conn *net.UnixConn) {
		request := make([]byte, daemonMaxRequest)
		_, _ = conn.Read(request)
		packet := daemonTestResponse(0, "", "")
		binary.LittleEndian.PutUint32(packet[16:20], daemonMaxResponse)
		_, _ = conn.Write(packet)
	})
	err := runDaemonAction(
		context.Background(), 42, actionCheckpoint, "", "/checkpoints/process-0000",
		daemonTestSettings(socket), testDaemonIdentity(42), logr.Discard(),
	)
	if err == nil || !strings.Contains(err.Error(), "payload lengths") {
		t.Fatalf("runDaemonAction() error = %v, want malformed bounded response error", err)
	}
}

func TestRunDaemonActionDisconnectIsNotUnavailable(t *testing.T) {
	socket := runDaemonTestServer(t, func(conn *net.UnixConn) {
		request := make([]byte, daemonMaxRequest)
		_, _ = conn.Read(request)
	})
	err := runDaemonAction(
		context.Background(), 42, actionCheckpoint, "", "/checkpoints/process-0000",
		daemonTestSettings(socket), testDaemonIdentity(42), logr.Discard(),
	)
	if err == nil || !strings.Contains(err.Error(), "will not be replayed") {
		t.Fatalf("runDaemonAction() error = %v, want ambiguous disconnect error", err)
	}
	if errors.Is(err, errDaemonUnavailable) {
		t.Fatalf("disconnect after request must not permit fallback: %v", err)
	}
}

func TestRunDaemonActionUnavailablePermitsConfiguredFallback(t *testing.T) {
	settings := daemonTestSettings(filepath.Join(t.TempDir(), "missing.sock"))
	err := runDaemonAction(
		context.Background(), 42, actionCheckpoint, "", "/checkpoints/process-0000",
		settings, testDaemonIdentity(42), logr.Discard(),
	)
	if !errors.Is(err, errDaemonUnavailable) {
		t.Fatalf("runDaemonAction() error = %v, want errDaemonUnavailable", err)
	}
}
