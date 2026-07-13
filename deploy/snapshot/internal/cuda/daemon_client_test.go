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

func daemonTestResponse(status int32, flags uint32) []byte {
	packet := make([]byte, daemonResponseHeader)
	binary.LittleEndian.PutUint32(packet[0:4], daemonProtocolMagic)
	binary.LittleEndian.PutUint16(packet[4:6], daemonProtocolVersion)
	binary.LittleEndian.PutUint16(packet[6:8], daemonResponseHeader)
	binary.LittleEndian.PutUint32(packet[8:12], uint32(status))
	binary.LittleEndian.PutUint32(packet[12:16], flags)
	return packet
}

func testDaemonIdentity(pid int) snapshotruntime.ProcessDetails {
	return snapshotruntime.ProcessDetails{
		ObservedPID: pid, OutermostPID: pid, InnermostPID: pid,
		NamespacePIDs: []int{pid}, StartTimeTicks: 12345,
		Cgroup: "0::/kubepods/test\n",
	}
}

func withOperationServer(t *testing.T, handler func(*net.UnixConn)) {
	t.Helper()
	socket := filepath.Join(t.TempDir(), "helper.sock")
	oldSocket := daemonSocketPath
	daemonSocketPath = socket
	t.Cleanup(func() { daemonSocketPath = oldSocket })
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: socket, Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go func() {
		conn, err := listener.AcceptUnix()
		if err != nil {
			return
		}
		defer conn.Close()
		handler(conn)
	}()
}

func TestCommandRunnerSendsDaemonOperations(t *testing.T) {
	identity := testDaemonIdentity(42)
	transfer := types.CUDATransferSettings{BufferCount: 2, ChunkBytes: 8 * 1024 * 1024}
	for _, test := range []struct {
		action, backend, deviceMap, storageDir string
		wantAction, wantBackend                uint16
		wantStorageDir                         string
		wantBufferCount                        uint32
		wantChunkBytes                         uint64
	}{
		{action: actionLock, backend: types.CUDAStorageModeLegacy, storageDir: "/ignored", wantAction: daemonActionLock, wantBackend: 1},
		{action: actionCheckpoint, backend: types.CUDAStorageModeLegacy, storageDir: "/ignored", wantAction: daemonActionCheckpoint, wantBackend: 1},
		{
			action: actionRestore, backend: types.CUDAStorageModePOSIX, deviceMap: "0=1",
			storageDir: "/checkpoints/process-0000", wantAction: daemonActionRestore, wantBackend: 2,
			wantStorageDir: "/checkpoints/process-0000", wantBufferCount: 2, wantChunkBytes: 8 * 1024 * 1024,
		},
		{action: actionUnlock, backend: types.CUDAStorageModePOSIX, storageDir: "/ignored", wantAction: daemonActionUnlock, wantBackend: 2},
	} {
		t.Run(test.action, func(t *testing.T) {
			request := make(chan []byte, 1)
			withOperationServer(t, func(conn *net.UnixConn) {
				packet := make([]byte, daemonMaxRequest)
				n, err := conn.Read(packet)
				if err != nil {
					t.Error(err)
					return
				}
				request <- packet[:n]
				_, _ = conn.Write(daemonTestResponse(0, 0))
			})
			if err := (commandHelperActionRunner{}).run(
				context.Background(), 42, test.action, test.deviceMap, test.backend, test.storageDir,
				transfer, identity, logr.Discard(),
			); err != nil {
				t.Fatalf("run() error = %v", err)
			}

			packet := <-request
			if got := binary.LittleEndian.Uint16(packet[8:10]); got != test.wantAction {
				t.Errorf("action = %d, want %d", got, test.wantAction)
			}
			if got := binary.LittleEndian.Uint16(packet[10:12]); got != test.wantBackend {
				t.Errorf("backend = %d, want %d", got, test.wantBackend)
			}
			if got := binary.LittleEndian.Uint32(packet[12:16]); got != 42 {
				t.Errorf("pid = %d, want 42", got)
			}
			if got := binary.LittleEndian.Uint32(packet[16:20]); got != test.wantBufferCount {
				t.Errorf("buffer count = %d, want %d", got, test.wantBufferCount)
			}
			if got := binary.LittleEndian.Uint64(packet[20:28]); got != test.wantChunkBytes {
				t.Errorf("chunk bytes = %d, want %d", got, test.wantChunkBytes)
			}
			deviceMapSize := int(binary.LittleEndian.Uint32(packet[28:32]))
			storageDirSize := int(binary.LittleEndian.Uint32(packet[32:36]))
			cgroupSize := int(binary.LittleEndian.Uint32(packet[36:40]))
			if got := binary.LittleEndian.Uint64(packet[40:48]); got != identity.StartTimeTicks {
				t.Errorf("start time = %d, want %d", got, identity.StartTimeTicks)
			}
			payload := packet[daemonRequestHeader:]
			if got := string(payload[:deviceMapSize]); got != test.deviceMap {
				t.Errorf("device map = %q, want %q", got, test.deviceMap)
			}
			payload = payload[deviceMapSize:]
			if got := string(payload[:storageDirSize]); got != test.wantStorageDir {
				t.Errorf("storage directory = %q, want %q", got, test.wantStorageDir)
			}
			payload = payload[storageDirSize:]
			if got := string(payload[:cgroupSize]); got != identity.Cgroup {
				t.Errorf("cgroup = %q, want %q", got, identity.Cgroup)
			}
		})
	}
}

func TestDaemonRequestRejectsBackendArgumentMismatch(t *testing.T) {
	identity := testDaemonIdentity(42)
	transfer := types.CUDATransferSettings{BufferCount: 1, ChunkBytes: types.DefaultCUDATransferChunkBytes}
	for _, test := range []struct {
		name, backend, storageDir string
	}{
		{name: "regular with directory", backend: types.CUDAStorageModeLegacy, storageDir: "/checkpoints/process-0000"},
		{name: "posix without directory", backend: types.CUDAStorageModePOSIX},
		{name: "unknown", backend: "auto"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := daemonRequest(42, actionCheckpoint, "", test.backend, test.storageDir, transfer, identity); err == nil {
				t.Fatal("daemonRequest() accepted mismatched backend arguments")
			}
		})
	}
}

func withHealthServer(t *testing.T, flags uint32) {
	t.Helper()
	socket := filepath.Join(t.TempDir(), "helper.sock")
	oldSocket := daemonSocketPath
	daemonSocketPath = socket
	t.Cleanup(func() { daemonSocketPath = oldSocket })
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: socket + ".health", Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go func() {
		conn, err := listener.AcceptUnix()
		if err != nil {
			return
		}
		defer conn.Close()
		request := make([]byte, daemonMaxRequest)
		_, _ = conn.Read(request)
		_, _ = conn.Write(daemonTestResponse(0, flags|daemonCapabilityDeferredCUDA))
	}()
}

func TestSelectCheckpointBackendFromCapability(t *testing.T) {
	for _, test := range []struct {
		name  string
		flags uint32
		want  string
	}{
		{name: "capable selects posix", flags: daemonCapabilityCustomStorage, want: types.CUDAStorageModePOSIX},
		{name: "incapable selects regular", want: types.CUDAStorageModeLegacy},
	} {
		t.Run(test.name, func(t *testing.T) {
			withHealthServer(t, test.flags)
			got, err := SelectCheckpointBackend(context.Background())
			if err != nil || got != test.want {
				t.Fatalf("SelectCheckpointBackend() = %q, %v; want %q", got, err, test.want)
			}
			manifest := types.NewCUDAManifest([]int{42}, []string{"GPU-aaa"}, got)
			if test.want == types.CUDAStorageModePOSIX && manifest.StorageMode != types.CUDAStorageModePOSIX {
				t.Fatalf("capable checkpoint manifest mode = %q, want posix", manifest.StorageMode)
			}
			if test.want == types.CUDAStorageModeLegacy && manifest.StorageMode != types.CUDAStorageModeLegacy {
				t.Fatalf("incapable checkpoint persisted mode = %q, want legacy", manifest.StorageMode)
			}
		})
	}
}

func TestValidateRestoreBackendObeysManifest(t *testing.T) {
	withHealthServer(t, daemonCapabilityCustomStorage)
	if err := ValidateRestoreBackend(context.Background(), types.CUDAStorageModeLegacy); err != nil {
		t.Fatalf("legacy restore on capable daemon failed: %v", err)
	}

	withHealthServer(t, 0)
	err := ValidateRestoreBackend(context.Background(), types.CUDAStorageModePOSIX)
	if err == nil || !strings.Contains(err.Error(), "requires daemon CustomStorage capability") {
		t.Fatalf("POSIX restore error = %v, want capability rejection", err)
	}
}

func TestRunDaemonActionMapsFatalResponse(t *testing.T) {
	withOperationServer(t, func(conn *net.UnixConn) {
		request := make([]byte, daemonMaxRequest)
		_, _ = conn.Read(request)
		_, _ = conn.Write(daemonTestResponse(2, daemonResponseFatal))
	})
	err := runDaemonAction(
		context.Background(), 42, actionCheckpoint, "", types.CUDAStorageModeLegacy, "",
		types.CUDATransferSettings{}, testDaemonIdentity(42), logr.Discard(),
	)
	if !errors.Is(err, errDaemonFatal) {
		t.Fatalf("runDaemonAction() error = %v, want errDaemonFatal", err)
	}
}

func TestRunDaemonActionRejectsInvalidResponses(t *testing.T) {
	for _, test := range []struct {
		name     string
		response func() []byte
		want     string
	}{
		{
			name: "malformed payload lengths",
			response: func() []byte {
				packet := daemonTestResponse(0, 0)
				binary.LittleEndian.PutUint32(packet[16:20], 1)
				return packet
			},
			want: "payload lengths",
		},
		{
			name: "oversized packet",
			response: func() []byte {
				return make([]byte, daemonMaxResponse+1)
			},
			want: "exceeded",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			withOperationServer(t, func(conn *net.UnixConn) {
				request := make([]byte, daemonMaxRequest)
				_, _ = conn.Read(request)
				_, _ = conn.Write(test.response())
			})
			err := runDaemonAction(
				context.Background(), 42, actionCheckpoint, "", types.CUDAStorageModeLegacy, "",
				types.CUDATransferSettings{}, testDaemonIdentity(42), logr.Discard(),
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("runDaemonAction() error = %v, want error containing %q", err, test.want)
			}
		})
	}
}

func TestRunDaemonActionDisconnectAfterSendIsNotReplayed(t *testing.T) {
	socket := filepath.Join(t.TempDir(), "helper.sock")
	oldSocket := daemonSocketPath
	daemonSocketPath = socket
	t.Cleanup(func() { daemonSocketPath = oldSocket })
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: socket, Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	accepted := make(chan int, 1)
	go func() {
		count := 0
		conn, acceptErr := listener.AcceptUnix()
		if acceptErr != nil {
			accepted <- count
			return
		}
		count++
		request := make([]byte, daemonMaxRequest)
		_, _ = conn.Read(request)
		_ = conn.Close()
		_ = listener.SetDeadline(time.Now().Add(100 * time.Millisecond))
		conn, acceptErr = listener.AcceptUnix()
		if acceptErr == nil {
			count++
			_ = conn.Close()
		}
		accepted <- count
	}()
	err = runDaemonAction(
		context.Background(), 42, actionCheckpoint, "", types.CUDAStorageModeLegacy, "",
		types.CUDATransferSettings{}, testDaemonIdentity(42), logr.Discard(),
	)
	if err == nil || !strings.Contains(err.Error(), "outcome is unknown and will not be replayed") {
		t.Fatalf("runDaemonAction() error = %v, want ambiguous non-replayable operation error", err)
	}
	if errors.Is(err, errDaemonUnavailable) {
		t.Fatalf("disconnect after request must not be treated as pre-send unavailability: %v", err)
	}
	if count := <-accepted; count != 1 {
		t.Fatalf("daemon accepted %d operation requests after an ambiguous disconnect, want 1", count)
	}
}
