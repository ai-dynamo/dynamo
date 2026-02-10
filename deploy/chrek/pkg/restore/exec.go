package restore

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criulog"
	"github.com/sirupsen/logrus"
)

const (
	defaultRestoreTimeout = 5 * time.Minute
)

// restoreViaExec performs external restore by executing criu-helper via nsenter.
func restoreViaExec(ctx context.Context, opts *RestoreOptions, log *logrus.Entry) (int, error) {
	log.WithFields(logrus.Fields{
		"target_pid": opts.TargetPID,
		"checkpoint": opts.CheckpointPath,
	}).Info("Starting external restore via nsenter + criu-helper")

	helperArgs := buildHelperArgs(opts, log)
	nsenterArgs := buildNsenterArgs(opts.TargetPID, helperArgs)

	stdout, err := executeNsenter(ctx, nsenterArgs, opts, log)
	if err != nil {
		return 0, err
	}

	pid, err := parseRestoredPID(stdout)
	if err != nil {
		return 0, err
	}

	log.WithField("pid", pid).Info("Restored PID from criu-helper")
	return pid, nil
}

func buildHelperArgs(opts *RestoreOptions, log *logrus.Entry) []string {
	args := []string{
		"--checkpoint", opts.CheckpointPath,
		fmt.Sprintf("--log-level=%d", opts.LogLevel),
		"--log-file", opts.LogFile,
		"--cgroup-root", "/",
	}

	if opts.WorkDir != "" {
		args = append(args, "--work-dir", opts.WorkDir)
		log.WithField("work_dir", opts.WorkDir).Info("Using custom work directory")
	}

	if opts.LibDir != "" {
		args = append(args, "--lib-dir", opts.LibDir)
		log.WithField("lib_dir", opts.LibDir).Info("Using custom plugin library directory")
	}

	if opts.TcpEstablished {
		args = append(args, "--tcp-established")
	}
	if opts.TcpClose {
		args = append(args, "--tcp-close")
	}

	args = append(args, "--verbose")
	return args
}

func buildNsenterArgs(targetPID int, helperArgs []string) []string {
	nsenterArgs := []string{
		"-t", fmt.Sprintf("%d", targetPID),
		"-a", // all namespaces
		"--",
		"/usr/local/bin/criu-helper",
	}
	return append(nsenterArgs, helperArgs...)
}

func executeNsenter(ctx context.Context, nsenterArgs []string, opts *RestoreOptions, log *logrus.Entry) (string, error) {
	restoreCtx, cancel := context.WithTimeout(ctx, defaultRestoreTimeout)
	defer cancel()

	cmd := exec.CommandContext(restoreCtx, "nsenter", nsenterArgs...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = log.Logger.Out

	log.Info("Executing criu-helper via nsenter")
	execStart := time.Now()
	err := cmd.Run()
	duration := time.Since(execStart)

	if errors.Is(restoreCtx.Err(), context.DeadlineExceeded) {
		return "", fmt.Errorf("restore timed out after %v", defaultRestoreTimeout)
	}

	if err != nil {
		log.WithFields(logrus.Fields{
			"error":    err,
			"duration": duration,
		}).Error("criu-helper via nsenter failed")

		if stdout.Len() > 0 {
			log.WithField("stdout", stdout.String()).Error("Helper stdout:")
		}

		criulog.LogErrorsWithWorkDir(opts.CheckpointPath, opts.WorkDir, opts.LogFile, log)
		return "", fmt.Errorf("criu-helper via nsenter failed: %w", err)
	}

	log.WithField("duration", duration).Info("criu-helper via nsenter completed successfully")
	return stdout.String(), nil
}

func parseRestoredPID(stdout string) (int, error) {
	pidRegex := regexp.MustCompile(`RESTORED_PID=(\d+)`)
	matches := pidRegex.FindStringSubmatch(stdout)
	if len(matches) < 2 {
		return 0, fmt.Errorf("criu-helper did not output RESTORED_PID")
	}

	pid, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0, fmt.Errorf("invalid PID from criu-helper: %w", err)
	}

	return pid, nil
}
