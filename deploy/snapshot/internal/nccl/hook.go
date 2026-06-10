package nccl

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
)

const (
	actionPrepare = "nccl_prepare"
	actionRestore = "nccl_restore"

	defaultActionTimeout = 5 * time.Minute
	hookReadyTimeout     = 10 * time.Second
	hookReadyInterval    = 100 * time.Millisecond
)

type PhaseTimings struct {
	TotalDuration time.Duration
}

func PrepareProcessTree(ctx context.Context, pids []int, log logr.Logger) (PhaseTimings, error) {
	return runProcessTreeAction(ctx, pids, actionPrepare, log)
}

func RestoreProcessTree(ctx context.Context, pids []int, procRoot string, log logr.Logger) (PhaseTimings, error) {
	return runProcessTreeActionWithProcRoot(ctx, pids, procRoot, actionRestore, log)
}

func PrepareConfiguredProcessTree(ctx context.Context, pids []int, log logr.Logger) (PhaseTimings, bool, error) {
	return runConfiguredProcessTreeAction(ctx, pids, snapshotruntime.HostProcPath, actionPrepare, log)
}

func RestoreConfiguredProcessTree(ctx context.Context, pids []int, procRoot string, log logr.Logger) (PhaseTimings, bool, error) {
	return runConfiguredProcessTreeAction(ctx, pids, procRoot, actionRestore, log)
}

func runConfiguredProcessTreeAction(ctx context.Context, pids []int, procRoot, action string, log logr.Logger) (PhaseTimings, bool, error) {
	var timings PhaseTimings
	configured, err := waitConfiguredHooks(ctx, pids, procRoot)
	if err != nil {
		return timings, false, err
	}
	if len(configured) == 0 {
		log.V(1).Info("Skipping NCCL checkpoint hook action; no configured hook sockets found",
			"action", action,
			"pids", pids,
		)
		return timings, false, nil
	}
	timings, err = runProcessTreeActionWithProcRoot(ctx, configured, procRoot, action, log)
	return timings, true, err
}

func waitConfiguredHooks(ctx context.Context, pids []int, procRoot string) ([]int, error) {
	deadline := time.NewTimer(hookReadyTimeout)
	defer deadline.Stop()
	ticker := time.NewTicker(hookReadyInterval)
	defer ticker.Stop()

	for {
		configured, missing := configuredHookState(pids, procRoot)
		if len(missing) == 0 {
			return configured, nil
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-deadline.C:
			return nil, fmt.Errorf("snapshot hook sockets missing for NCCL checkpoint configured PIDs %v", missing)
		case <-ticker.C:
		}
	}
}

func configuredHookState(pids []int, procRoot string) ([]int, []int) {
	var configured []int
	var missing []int
	for _, pid := range pids {
		if HookPresent(procRoot, pid) {
			configured = append(configured, pid)
			continue
		}
		if hasNCCLCheckpointConfig(procRoot, pid) {
			missing = append(missing, pid)
		}
	}
	return configured, missing
}

func runProcessTreeAction(ctx context.Context, pids []int, action string, log logr.Logger) (PhaseTimings, error) {
	return runProcessTreeActionWithProcRoot(ctx, pids, snapshotruntime.HostProcPath, action, log)
}

func runProcessTreeActionWithProcRoot(ctx context.Context, pids []int, procRoot, action string, log logr.Logger) (PhaseTimings, error) {
	var timings PhaseTimings
	start := time.Now()
	if len(pids) == 0 {
		return timings, nil
	}

	actionCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	errCh := make(chan error, len(pids))
	for _, pid := range pids {
		pid := pid
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := runProcessAction(actionCtx, procRoot, pid, action, log); err != nil {
				errCh <- err
				cancel()
			}
		}()
	}
	wg.Wait()
	close(errCh)
	if err := <-errCh; err != nil {
		timings.TotalDuration = time.Since(start)
		return timings, err
	}
	timings.TotalDuration = time.Since(start)
	return timings, nil
}

func runProcessAction(ctx context.Context, procRoot string, pid int, action string, log logr.Logger) error {
	details := snapshotruntime.ReadProcessDetailsOrDefault(procRoot, pid)
	socketPath := hookSocketPath(procRoot, details)
	actionCtx, cancel := context.WithTimeout(ctx, defaultActionTimeout)
	defer cancel()

	start := time.Now()
	var d net.Dialer
	conn, err := d.DialContext(actionCtx, "unix", socketPath)
	if err != nil {
		return fmt.Errorf("connect snapshot hook for pid %d (%s): %w", pid, socketPath, err)
	}
	defer conn.Close()
	if deadline, ok := actionCtx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	}
	cancelDone := make(chan struct{})
	go func() {
		select {
		case <-actionCtx.Done():
			_ = conn.SetDeadline(time.Now())
		case <-cancelDone:
		}
	}()
	defer close(cancelDone)
	if _, err := fmt.Fprintf(conn, "%s\n", action); err != nil {
		return fmt.Errorf("send snapshot hook action %s to pid %d: %w", action, pid, err)
	}
	response, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		return fmt.Errorf("read snapshot hook action %s response from pid %d: %w", action, pid, err)
	}
	response = strings.TrimSpace(response)
	if !strings.HasPrefix(response, "ok ") {
		return fmt.Errorf("snapshot hook action %s failed for pid %d: %s", action, pid, response)
	}
	fields := strings.Fields(response)
	if len(fields) != 2 {
		return fmt.Errorf("snapshot hook action %s returned malformed response for pid %d: %q", action, pid, response)
	}
	result, err := strconv.Atoi(fields[1])
	if err != nil {
		return fmt.Errorf("snapshot hook action %s returned non-integer result for pid %d: %q", action, pid, response)
	}
	if result != 0 {
		return fmt.Errorf("snapshot hook action %s returned NCCL result %d for pid %d", action, result, pid)
	}
	log.V(1).Info("snapshot hook action succeeded",
		"pid", pid,
		"outermost_pid", details.OutermostPID,
		"innermost_pid", details.InnermostPID,
		"cmdline", details.Cmdline,
		"action", action,
		"duration", time.Since(start),
	)
	return nil
}

func hookSocketPath(procRoot string, details snapshotruntime.ProcessDetails) string {
	controlDir := filepath.Join(procRoot, strconv.Itoa(details.ObservedPID), "root", "snapshot-control")
	pid := details.InnermostPID
	if pid <= 0 {
		pid = details.ObservedPID
	}
	return filepath.Join(controlDir, "snapshot-hook", fmt.Sprintf("%d.sock", pid))
}

func HookPresent(procRoot string, pid int) bool {
	details := snapshotruntime.ReadProcessDetailsOrDefault(procRoot, pid)
	_, err := os.Stat(hookSocketPath(procRoot, details))
	return err == nil
}

func hasNCCLCheckpointConfig(procRoot string, pid int) bool {
	data, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "environ"))
	if err != nil {
		return false
	}
	for _, entry := range strings.Split(string(data), "\x00") {
		if strings.HasPrefix(entry, "NCCL_CHECKPOINT_KVS_PATH=") ||
			strings.Contains(entry, "libnccl-checkpoint") ||
			strings.Contains(entry, "dynamo-snapshot-hook") {
			return true
		}
	}
	return false
}
