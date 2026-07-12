package cuda

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

var cudaCheckpointHelperBinary = "/usr/local/bin/cuda-checkpoint-helper"

const (
	actionLock       = "lock"
	actionCheckpoint = "checkpoint"
	actionRestore    = "restore"
	actionUnlock     = "unlock"
)

type helperActionRunner interface {
	run(
		context.Context,
		int,
		string,
		string,
		string,
		string,
		types.CUDATransferSettings,
		snapshotruntime.ProcessDetails,
		logr.Logger,
	) error
	state(context.Context, int) (string, error)
}

type commandHelperActionRunner struct{}

type identityValidatingRunner struct {
	runner     helperActionRunner
	procRoot   string
	identities map[int]snapshotruntime.ProcessDetails
}

type customStorageTelemetry struct {
	Event                        string          `json:"event"`
	HelperMainToTelemetrySeconds json.RawMessage `json:"helper_main_to_telemetry_seconds"`
}

type customStorageTelemetryParse struct {
	status             string
	err                string
	helperMainDuration time.Duration
}

func parseCustomStorageTelemetry(output string, processWall time.Duration) customStorageTelemetryParse {
	sawMalformedJSON := false
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var telemetry customStorageTelemetry
		if err := json.Unmarshal([]byte(line), &telemetry); err != nil {
			sawMalformedJSON = true
			continue
		}
		if telemetry.Event != "cuda_custom_storage_transfer" {
			continue
		}
		if len(telemetry.HelperMainToTelemetrySeconds) == 0 ||
			string(telemetry.HelperMainToTelemetrySeconds) == "null" {
			return customStorageTelemetryParse{
				status: "missing-duration",
				err:    "expected helper_main_to_telemetry_seconds",
			}
		}
		var seconds json.Number
		if err := json.Unmarshal(telemetry.HelperMainToTelemetrySeconds, &seconds); err != nil {
			return customStorageTelemetryParse{
				status: "invalid-duration",
				err:    "helper_main_to_telemetry_seconds is not a number",
			}
		}
		value, err := strconv.ParseFloat(seconds.String(), 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return customStorageTelemetryParse{
				status: "invalid-duration",
				err:    "helper_main_to_telemetry_seconds is not finite",
			}
		}
		if value < 0 {
			return customStorageTelemetryParse{
				status: "invalid-duration",
				err:    "helper_main_to_telemetry_seconds is negative",
			}
		}

		// The helper prints six fractional digits, so tolerate at most one
		// microsecond of upward rounding without adding to processWall.
		const roundingToleranceSeconds = 1e-6
		processWallSeconds := processWall.Seconds()
		if value > processWallSeconds+roundingToleranceSeconds {
			return customStorageTelemetryParse{
				status: "duration-exceeds-process-wall",
				err:    "helper_main_to_telemetry_seconds exceeds process wall duration",
			}
		}
		if value >= processWallSeconds ||
			value*float64(time.Second) >= float64(math.MaxInt64) {
			return customStorageTelemetryParse{
				status:             "valid",
				helperMainDuration: processWall,
			}
		}
		return customStorageTelemetryParse{
			status:             "valid",
			helperMainDuration: time.Duration(value * float64(time.Second)),
		}
	}
	if sawMalformedJSON {
		return customStorageTelemetryParse{
			status: "malformed-json",
			err:    "malformed JSON telemetry output",
		}
	}
	return customStorageTelemetryParse{
		status: "event-absent",
		err:    "cuda_custom_storage_transfer event not found",
	}
}

func customStorageSuccessLogValues(
	pid int,
	action string,
	processWall time.Duration,
	output string,
	telemetry customStorageTelemetryParse,
) []any {
	values := []any{
		"pid", pid,
		"action", action,
		"duration", processWall,
		"helper_process_wall_duration", processWall,
		"helper_telemetry_status", telemetry.status,
	}
	if telemetry.status == "valid" {
		values = append(
			values,
			"helper_main_to_telemetry_duration", telemetry.helperMainDuration,
			"helper_process_overhead_duration", processWall-telemetry.helperMainDuration,
		)
	} else {
		values = append(values, "helper_telemetry_error", telemetry.err)
	}
	return append(values, "output", output)
}

func (commandHelperActionRunner) run(
	ctx context.Context,
	pid int,
	action,
	deviceMap,
	storageMode,
	storageDir string,
	transferSettings types.CUDATransferSettings,
	identity snapshotruntime.ProcessDetails,
	log logr.Logger,
) error {
	if transferSettings.DaemonSocket != "" {
		if identity.StartTimeTicks == 0 || identity.Cgroup == "" {
			captured, err := snapshotruntime.ReadProcessDetails(snapshotruntime.HostProcPath, pid)
			if err != nil {
				return fmt.Errorf("capture host PID %d identity for CUDA helper daemon: %w", pid, err)
			}
			identity = captured
		}
		daemonStorageDir := storageDir
		if action == actionLock || action == actionUnlock {
			daemonStorageDir = ""
		}
		err := runDaemonAction(ctx, pid, action, deviceMap, daemonStorageDir, transferSettings, identity, log)
		if err == nil || !transferSettings.DaemonFallback || !errors.Is(err, errDaemonUnavailable) {
			return err
		}
		log.Info("CUDA helper daemon unavailable before request; using configured one-shot fallback",
			"socket", transferSettings.DaemonSocket,
			"pid", pid,
			"action", action,
			"error", err,
		)
	}
	return runAction(ctx, pid, action, deviceMap, storageMode, storageDir, transferSettings, log)
}

func (commandHelperActionRunner) state(ctx context.Context, pid int) (string, error) {
	return getState(ctx, pid)
}

func (r identityValidatingRunner) run(
	ctx context.Context,
	pid int,
	action,
	deviceMap,
	storageMode,
	storageDir string,
	transferSettings types.CUDATransferSettings,
	_ snapshotruntime.ProcessDetails,
	log logr.Logger,
) error {
	expected, ok := r.identities[pid]
	if !ok {
		return fmt.Errorf("missing expected process identity for host PID %d", pid)
	}
	if err := snapshotruntime.ValidateProcessIdentity(r.procRoot, expected); err != nil {
		return fmt.Errorf("validate host PID %d immediately before CUDA %s: %w", pid, action, err)
	}
	return r.runner.run(ctx, pid, action, deviceMap, storageMode, storageDir, transferSettings, expected, log)
}

func (r identityValidatingRunner) state(ctx context.Context, pid int) (string, error) {
	return r.runner.state(ctx, pid)
}

func getState(ctx context.Context, pid int) (string, error) {
	cmd := exec.CommandContext(ctx, cudaCheckpointHelperBinary, "--get-state", "--pid", strconv.Itoa(pid))
	output, err := cmd.CombinedOutput()
	state := strings.TrimSpace(string(output))
	if err != nil {
		return "", fmt.Errorf("cuda-checkpoint-helper --get-state failed for pid %d: %w (output: %s)", pid, err, state)
	}
	if state == "" {
		return "", fmt.Errorf("cuda-checkpoint-helper --get-state returned empty state for pid %d", pid)
	}
	return state, nil
}

func helperActionArgs(pid int, action, deviceMap, storageMode, storageDir string, transferSettings types.CUDATransferSettings) []string {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	if storageMode == "posix" {
		args = append(
			args,
			"--storage-mode", storageMode,
			"--storage-dir", storageDir,
			"--transfer-buffer-count", strconv.Itoa(transferSettings.BufferCount),
			"--transfer-chunk-bytes", strconv.FormatUint(transferSettings.ChunkBytes, 10),
		)
	}
	return args
}

func runAction(ctx context.Context, pid int, action, deviceMap, storageMode, storageDir string, transferSettings types.CUDATransferSettings, log logr.Logger) error {
	args := helperActionArgs(pid, action, deviceMap, storageMode, storageDir, transferSettings)
	cmd := exec.CommandContext(ctx, cudaCheckpointHelperBinary, args...)
	details := snapshotruntime.ProcessDetails{
		ObservedPID:   pid,
		OutermostPID:  pid,
		InnermostPID:  pid,
		NamespacePIDs: []int{pid},
	}
	if process, err := snapshotruntime.ReadProcessDetails("/proc", pid); err == nil {
		details = process
	}
	start := time.Now()
	output, err := cmd.CombinedOutput()
	duration := time.Since(start)
	out := strings.TrimSpace(string(output))
	if err != nil {
		log.Error(err, "cuda-checkpoint-helper command failed",
			"pid", pid,
			"outermost_pid", details.OutermostPID,
			"innermost_pid", details.InnermostPID,
			"cmdline", details.Cmdline,
			"action", action,
			"duration", duration,
			"output", out,
		)
		return fmt.Errorf("cuda-checkpoint-helper %v failed for pid %d after %s: %w (output: %s)", args, pid, duration, err, out)
	}
	if storageMode == "posix" && (action == actionCheckpoint || action == actionRestore) {
		telemetry := parseCustomStorageTelemetry(out, duration)
		log.Info("CUDA custom-storage transfer succeeded",
			customStorageSuccessLogValues(pid, action, duration, out, telemetry)...,
		)
		return nil
	}
	log.V(1).Info("cuda-checkpoint-helper command succeeded",
		"pid", pid,
		"outermost_pid", details.OutermostPID,
		"innermost_pid", details.InnermostPID,
		"cmdline", details.Cmdline,
		"action", action,
		"duration", duration,
		"output", out,
	)
	return nil
}
