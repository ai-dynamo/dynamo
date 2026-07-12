/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package cuda

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"os"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	daemonProtocolMagic   = uint32(0x50484344)
	daemonProtocolVersion = uint16(3)
	daemonRequestHeader   = 48
	daemonResponseHeader  = 24
	daemonMaxRequest      = 64 * 1024
	daemonMaxResponse     = 128 * 1024
	daemonMaxCgroup       = 4096

	daemonActionHealth     = uint16(0)
	daemonActionCheckpoint = uint16(1)
	daemonActionRestore    = uint16(2)
	daemonActionLock       = uint16(3)
	daemonActionUnlock     = uint16(4)

	daemonResponseFatal          = uint32(1 << 0)
	daemonCapabilityDeferredCUDA = uint32(1 << 1)
	daemonHealthWait             = 30 * time.Second
	daemonHealthRetryInterval    = 100 * time.Millisecond
)

var errDaemonUnavailable = errors.New("CUDA helper daemon unavailable")
var errDaemonFatal = errors.New("CUDA helper daemon entered fatal state")

func daemonRequest(
	pid int,
	action,
	deviceMap,
	storageDir string,
	transfer types.CUDATransferSettings,
	identity snapshotruntime.ProcessDetails,
) ([]byte, error) {
	var daemonAction uint16
	switch action {
	case "":
		daemonAction = daemonActionHealth
	case actionCheckpoint:
		daemonAction = daemonActionCheckpoint
	case actionRestore:
		daemonAction = daemonActionRestore
	case actionLock:
		daemonAction = daemonActionLock
	case actionUnlock:
		daemonAction = daemonActionUnlock
	default:
		return nil, fmt.Errorf("action %q is not supported by CUDA helper daemon", action)
	}
	health := daemonAction == daemonActionHealth
	if health {
		pid = 0
		identity = snapshotruntime.ProcessDetails{}
	} else if pid <= 0 || identity.OutermostPID != pid || identity.StartTimeTicks == 0 ||
		identity.Cgroup == "" || len(identity.Cgroup) > daemonMaxCgroup {
		return nil, errors.New("invalid CUDA helper daemon process identity")
	}
	if daemonAction == daemonActionLock || daemonAction == daemonActionUnlock {
		if deviceMap != "" || storageDir != "" {
			return nil, errors.New("CUDA helper daemon lock/unlock request has transfer arguments")
		}
		transfer = types.CUDATransferSettings{}
	} else if !health && (storageDir == "" || storageDir[0] != '/') {
		return nil, errors.New("invalid CUDA helper daemon storage directory")
	}
	if len(deviceMap)+len(storageDir)+len(identity.Cgroup) > daemonMaxRequest-daemonRequestHeader {
		return nil, errors.New("CUDA helper daemon request is too large")
	}
	packet := make([]byte, daemonRequestHeader+len(deviceMap)+len(storageDir)+len(identity.Cgroup))
	binary.LittleEndian.PutUint32(packet[0:4], daemonProtocolMagic)
	binary.LittleEndian.PutUint16(packet[4:6], daemonProtocolVersion)
	binary.LittleEndian.PutUint16(packet[6:8], daemonRequestHeader)
	binary.LittleEndian.PutUint16(packet[8:10], daemonAction)
	binary.LittleEndian.PutUint32(packet[12:16], uint32(pid))
	binary.LittleEndian.PutUint32(packet[16:20], uint32(transfer.BufferCount))
	binary.LittleEndian.PutUint64(packet[20:28], transfer.ChunkBytes)
	binary.LittleEndian.PutUint32(packet[28:32], uint32(len(deviceMap)))
	binary.LittleEndian.PutUint32(packet[32:36], uint32(len(storageDir)))
	binary.LittleEndian.PutUint32(packet[36:40], uint32(len(identity.Cgroup)))
	binary.LittleEndian.PutUint64(packet[40:48], identity.StartTimeTicks)
	copy(packet[daemonRequestHeader:], deviceMap)
	copy(packet[daemonRequestHeader+len(deviceMap):], storageDir)
	copy(packet[daemonRequestHeader+len(deviceMap)+len(storageDir):], identity.Cgroup)
	return packet, nil
}

func parseDaemonResponse(packet []byte) (int32, uint32, string, string, error) {
	if len(packet) < daemonResponseHeader || len(packet) > daemonMaxResponse {
		return 0, 0, "", "", fmt.Errorf("invalid CUDA helper daemon response size %d", len(packet))
	}
	if binary.LittleEndian.Uint32(packet[0:4]) != daemonProtocolMagic ||
		binary.LittleEndian.Uint16(packet[4:6]) != daemonProtocolVersion ||
		binary.LittleEndian.Uint16(packet[6:8]) != daemonResponseHeader {
		return 0, 0, "", "", errors.New("invalid CUDA helper daemon response header")
	}
	flags := binary.LittleEndian.Uint32(packet[12:16])
	if flags & ^(daemonResponseFatal|daemonCapabilityDeferredCUDA) != 0 {
		return 0, 0, "", "", errors.New("invalid CUDA helper daemon response flags")
	}
	outputSize := int(binary.LittleEndian.Uint32(packet[16:20]))
	errorSize := int(binary.LittleEndian.Uint32(packet[20:24]))
	if outputSize < 0 || errorSize < 0 || outputSize+errorSize != len(packet)-daemonResponseHeader {
		return 0, 0, "", "", errors.New("invalid CUDA helper daemon response payload lengths")
	}
	payload := packet[daemonResponseHeader:]
	return int32(binary.LittleEndian.Uint32(packet[8:12])),
		flags, string(payload[:outputSize]), string(payload[outputSize:]), nil
}

func daemonRPC(
	ctx context.Context,
	socket string,
	packet []byte,
) (int32, uint32, string, string, time.Duration, error) {
	dialer := net.Dialer{}
	conn, err := dialer.DialContext(ctx, "unixpacket", socket)
	if err != nil {
		return 0, 0, "", "", 0, fmt.Errorf("%w at %s: %v", errDaemonUnavailable, socket, err)
	}
	defer conn.Close()
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	defer stop()

	start := time.Now()
	written, err := conn.Write(packet)
	if err != nil || written != len(packet) {
		return 0, 0, "", "", time.Since(start),
			fmt.Errorf("CUDA helper daemon request write failed; operation outcome is unknown and will not be replayed: %w", err)
	}
	response := make([]byte, daemonMaxResponse+1)
	read, err := conn.Read(response)
	rpcWall := time.Since(start)
	if err != nil {
		if errors.Is(err, os.ErrDeadlineExceeded) || ctx.Err() != nil {
			return 0, 0, "", "", rpcWall, fmt.Errorf("CUDA helper daemon RPC canceled after %s: %w", rpcWall, ctx.Err())
		}
		return 0, 0, "", "", rpcWall,
			fmt.Errorf("CUDA helper daemon disconnected after request; operation outcome is unknown and will not be replayed: %w", err)
	}
	if read > daemonMaxResponse {
		return 0, 0, "", "", rpcWall, fmt.Errorf("CUDA helper daemon response exceeded %d bytes", daemonMaxResponse)
	}
	status, flags, stdout, stderr, err := parseDaemonResponse(response[:read])
	if err != nil {
		return 0, 0, "", "", rpcWall, err
	}
	return status, flags, stdout, stderr, rpcWall, nil
}

func runDaemonAction(
	ctx context.Context,
	pid int,
	action,
	deviceMap,
	storageDir string,
	transfer types.CUDATransferSettings,
	identity snapshotruntime.ProcessDetails,
	log logr.Logger,
) error {
	packet, err := daemonRequest(pid, action, deviceMap, storageDir, transfer, identity)
	if err != nil {
		return err
	}
	status, flags, stdout, stderr, rpcWall, err := daemonRPC(ctx, transfer.DaemonSocket, packet)
	if err != nil {
		return err
	}
	output := stdout + stderr
	if status != 0 {
		if flags&daemonResponseFatal != 0 {
			return fmt.Errorf("%w: CUDA helper daemon %s failed for pid %d after %s with CUDA status %d (output: %s)",
				errDaemonFatal, action, pid, rpcWall, status, output)
		}
		return fmt.Errorf("CUDA helper daemon %s failed for pid %d after %s with CUDA status %d (output: %s)",
			action, pid, rpcWall, status, output)
	}
	if action == actionLock || action == actionUnlock {
		log.V(1).Info("CUDA helper daemon action succeeded",
			"pid", pid,
			"action", action,
			"daemon_rpc_wall_duration", rpcWall,
			"output", output,
		)
		return nil
	}
	telemetry := parseCustomStorageTelemetry(output, rpcWall)
	values := []any{
		"pid", pid,
		"action", action,
		"transport", "daemon",
		"daemon_rpc_wall_duration", rpcWall,
		"helper_telemetry_status", telemetry.status,
	}
	if telemetry.status == "valid" {
		values = append(values, "helper_operation_to_telemetry_duration", telemetry.helperMainDuration)
	} else {
		values = append(values, "helper_telemetry_error", telemetry.err)
	}
	log.Info("CUDA custom-storage transfer succeeded", append(values, "output", output)...)
	return nil
}

func daemonHealthy(ctx context.Context, socket string) error {
	packet, err := daemonRequest(0, "", "", "", types.CUDATransferSettings{}, snapshotruntime.ProcessDetails{})
	if err != nil {
		return err
	}
	status, flags, _, stderr, _, err := daemonRPC(ctx, socket+".health", packet)
	if err != nil {
		return err
	}
	if status != 0 || flags&daemonCapabilityDeferredCUDA == 0 {
		return fmt.Errorf("CUDA helper daemon health/capability check failed: status=%d flags=%#x error=%s",
			status, flags, stderr)
	}
	return nil
}

// WaitForDaemon waits for a protocol-level health and deferred-CUDA capability response.
func WaitForDaemon(ctx context.Context, socket string) error {
	if socket == "" {
		return nil
	}
	waitCtx, cancel := context.WithTimeout(ctx, daemonHealthWait)
	defer cancel()
	ticker := time.NewTicker(daemonHealthRetryInterval)
	defer ticker.Stop()
	var lastErr error
	for {
		if err := daemonHealthy(waitCtx, socket); err == nil {
			return nil
		} else {
			lastErr = err
		}
		select {
		case <-waitCtx.Done():
			return fmt.Errorf("wait for CUDA helper daemon at %s: %w (last error: %v)", socket, waitCtx.Err(), lastErr)
		case <-ticker.C:
		}
	}
}
