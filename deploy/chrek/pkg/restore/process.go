package restore

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// MonitorProcess monitors the restored process and returns its exit code.
// It blocks until the process exits. Does not forward stdout/stderr.
// For output forwarding, use ForwardProcessOutput instead.
func MonitorProcess(pid int, log *logrus.Entry) int {
	log.WithField("pid", pid).Info("Monitoring restored process")

	for {
		// Check if process still exists by sending signal 0
		proc, err := os.FindProcess(pid)
		if err != nil {
			log.WithError(err).Error("Failed to find process")
			return 1
		}

		err = proc.Signal(syscall.Signal(0))
		if err != nil {
			// Process has exited
			log.WithField("pid", pid).Info("Restored process exited")

			// Try to read exit status from /proc/<pid>/stat
			// If process is gone, assume exit code 0
			exitCode := getExitCode(pid)
			log.WithField("exit_code", exitCode).Info("Restored process exit status")
			return exitCode
		}

		time.Sleep(time.Second)
	}
}

// ForwardProcessOutput forwards the stdout and stderr of a restored process
// to our own stdout/stderr via /proc/<pid>/fd/1 and /proc/<pid>/fd/2.
// This ensures logs from the restored process appear in kubectl logs.
// Returns the exit code of the process.
func ForwardProcessOutput(pid int, log *logrus.Entry) int {
	log.WithField("pid", pid).Info("Forwarding output from restored process")

	// Try to open the process's stdout and stderr via /proc
	stdoutPath := fmt.Sprintf("/proc/%d/fd/1", pid)
	stderrPath := fmt.Sprintf("/proc/%d/fd/2", pid)

	// Channel to signal when copying goroutines should stop
	done := make(chan struct{})

	// Forward stdout
	go forwardFD(stdoutPath, os.Stdout, "stdout", log, done)

	// Forward stderr
	go forwardFD(stderrPath, os.Stderr, "stderr", log, done)

	// Wait for process to exit
	exitCode := waitForProcess(pid, log)

	// Signal goroutines to stop
	close(done)

	// Give goroutines a moment to flush any remaining output
	time.Sleep(100 * time.Millisecond)

	return exitCode
}

// forwardFD copies data from a file descriptor path to a writer.
// It handles the case where the FD may not be readable.
func forwardFD(fdPath string, dst io.Writer, name string, log *logrus.Entry, done <-chan struct{}) {
	// Try to open the FD path
	src, err := os.Open(fdPath)
	if err != nil {
		log.WithError(err).WithField("path", fdPath).Debug("Could not open process FD for forwarding")
		return
	}
	defer src.Close()

	// Check what kind of file this is
	stat, err := src.Stat()
	if err != nil {
		log.WithError(err).WithField("path", fdPath).Debug("Could not stat process FD")
		return
	}

	log.WithFields(logrus.Fields{
		"name": name,
		"mode": stat.Mode().String(),
		"path": fdPath,
	}).Debug("Forwarding process output")

	// Copy data until done or EOF
	buf := make([]byte, 4096)
	for {
		select {
		case <-done:
			return
		default:
			// Set a read deadline to allow checking done channel periodically
			src.SetReadDeadline(time.Now().Add(100 * time.Millisecond))

			n, err := src.Read(buf)
			if n > 0 {
				dst.Write(buf[:n])
			}
			if err != nil {
				if os.IsTimeout(err) {
					continue
				}
				if err != io.EOF {
					log.WithError(err).WithField("name", name).Debug("Error reading from process FD")
				}
				return
			}
		}
	}
}

// waitForProcess waits for a process to exit and returns its exit code.
func waitForProcess(pid int, log *logrus.Entry) int {
	for {
		// Check if process still exists by sending signal 0
		proc, err := os.FindProcess(pid)
		if err != nil {
			log.WithError(err).Error("Failed to find process")
			return 1
		}

		err = proc.Signal(syscall.Signal(0))
		if err != nil {
			// Process has exited
			log.WithField("pid", pid).Info("Restored process exited")

			// Try to get exit status
			exitCode := getExitCode(pid)
			log.WithField("exit_code", exitCode).Info("Restored process exit status")
			return exitCode
		}

		time.Sleep(100 * time.Millisecond)
	}
}

// getExitCode attempts to get the exit code of a process.
// Returns 0 if unable to determine the exit code.
func getExitCode(pid int) int {
	// Try to wait for the process (only works if we're the parent)
	proc, err := os.FindProcess(pid)
	if err != nil {
		return 0
	}

	// Try waitpid with WNOHANG - this may not work for non-child processes
	var wstatus syscall.WaitStatus
	wpid, err := syscall.Wait4(pid, &wstatus, syscall.WNOHANG, nil)
	if err == nil && wpid == pid {
		if wstatus.Exited() {
			return wstatus.ExitStatus()
		}
		if wstatus.Signaled() {
			return 128 + int(wstatus.Signal())
		}
	}

	// If we can't wait on it, check if it's still running
	if proc.Signal(syscall.Signal(0)) != nil {
		// Process is gone, assume clean exit
		return 0
	}

	return 0
}

// SetupSignalForwarding sets up signal forwarding to the restored process.
// Returns a cleanup function that should be called when done.
func SetupSignalForwarding(pid int, log *logrus.Entry) func() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT, syscall.SIGQUIT)

	done := make(chan struct{})

	go func() {
		select {
		case sig := <-sigChan:
			log.WithFields(logrus.Fields{
				"signal": sig,
				"pid":    pid,
			}).Info("Forwarding signal to restored process")

			proc, err := os.FindProcess(pid)
			if err == nil {
				proc.Signal(sig)
			}
		case <-done:
			return
		}
	}()

	return func() {
		signal.Stop(sigChan)
		close(done)
	}
}

// WaitForPidFile waits for the CRIU PID file to be created and returns the PID.
func WaitForPidFile(pidFile string, timeout time.Duration, log *logrus.Entry) (int, error) {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		data, err := os.ReadFile(pidFile)
		if err == nil {
			pidStr := strings.TrimSpace(string(data))
			pid, err := strconv.Atoi(pidStr)
			if err == nil && pid > 0 {
				return pid, nil
			}
		}
		time.Sleep(100 * time.Millisecond)
	}

	return 0, fmt.Errorf("timeout waiting for PID file %s after %v", pidFile, timeout)
}

// ExecColdStart execs the cold start command (ColdStartArgs), replacing the current process.
// If no args are provided, falls back to sleep infinity.
func ExecColdStart(cfg *config.RestoreConfig, log *logrus.Entry) error {
	if len(cfg.ColdStartArgs) == 0 {
		log.Warn("No cold start command provided, sleeping indefinitely")
		return ExecArgs([]string{"sleep", "infinity"}, log)
	}

	log.WithField("cmd", cfg.ColdStartArgs).Info("Executing cold start command")
	return ExecArgs(cfg.ColdStartArgs, log)
}

// ExecArgs replaces the current process with the given command and arguments.
// Uses syscall.Exec for proper PID 1 behavior in containers.
func ExecArgs(args []string, log *logrus.Entry) error {
	if len(args) == 0 {
		return fmt.Errorf("empty command")
	}

	// Find the executable path
	path, err := exec.LookPath(args[0])
	if err != nil {
		return fmt.Errorf("command not found: %s: %w", args[0], err)
	}

	log.WithFields(logrus.Fields{
		"path": path,
		"args": args,
	}).Debug("Replacing process via syscall.Exec")

	// Replace current process with the command
	return syscall.Exec(path, args, os.Environ())
}
