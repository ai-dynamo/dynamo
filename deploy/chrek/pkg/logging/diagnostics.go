package logging

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// LogProcessDiagnostics logs process state and CRIU restore log for debugging a failed restore.
func LogProcessDiagnostics(procRoot string, pid int, restoreLogPath string, log logr.Logger) {
	entry := log.WithValues("restored_pid", pid)

	// Process status and cmdline
	pidStr := strconv.Itoa(pid)
	if data, err := os.ReadFile(filepath.Join(procRoot, pidStr, "status")); err == nil {
		entry.Info("Process status", "content", strings.TrimSpace(string(data)))
	}
	if data, err := os.ReadFile(filepath.Join(procRoot, pidStr, "cmdline")); err == nil {
		cmdline := strings.TrimSpace(strings.ReplaceAll(string(data), "\x00", " "))
		if cmdline != "" {
			entry.Info("Process cmdline", "cmdline", cmdline)
		}
	}

	// Exit code from /proc/stat
	if data, err := os.ReadFile(filepath.Join(procRoot, pidStr, "stat")); err == nil {
		if ws, err := common.ParseProcExitCode(string(data)); err == nil {
			entry.Info("Process exit code", "exit_status", ws.ExitStatus(), "term_signal", ws.Signal(), "core_dumped", ws.CoreDump())
		}
	}

	// PID 1 children in restored namespace
	if data, err := os.ReadFile(filepath.Join(procRoot, "1", "task", "1", "children")); err == nil {
		entry.Info("PID 1 children", "children", strings.TrimSpace(string(data)))
	}

	// CRIU restore log summary
	logRestoreLog(restoreLogPath, entry)
}

// LogRestoreErrors finds the CRIU restore.log and logs key lines from it.
func LogRestoreErrors(checkpointPath, workDir string, log logr.Logger) {
	// Try workdir first, then checkpoint dir
	for _, dir := range []string{workDir, checkpointPath} {
		if dir == "" {
			continue
		}
		logPath := filepath.Join(dir, "restore.log")
		if _, err := os.Stat(logPath); err == nil {
			logRestoreLog(logPath, log)
			return
		}
	}
}

// logRestoreLog extracts key lines and tail from a CRIU restore log file.
func logRestoreLog(path string, log logr.Logger) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	lines := strings.Split(string(data), "\n")

	// Extract error/warning/notable lines
	var keyLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		lower := strings.ToLower(trimmed)
		if strings.Contains(lower, "error") ||
			strings.Contains(lower, "warn") ||
			strings.Contains(lower, "fail") ||
			strings.Contains(lower, "cuda") ||
			strings.Contains(lower, "restore finished successfully") {
			keyLines = append(keyLines, trimmed)
			if len(keyLines) >= 80 {
				break
			}
		}
	}
	if len(keyLines) > 0 {
		log.Info("CRIU restore key lines", "path", path, "lines", strings.Join(keyLines, " | "))
	}

	// Last 40 non-empty lines
	var tail []string
	for i := len(lines) - 1; i >= 0 && len(tail) < 40; i-- {
		if trimmed := strings.TrimSpace(lines[i]); trimmed != "" {
			tail = append(tail, trimmed)
		}
	}
	slices.Reverse(tail)
	if len(tail) > 0 {
		log.Info("CRIU restore tail", "path", path, "lines", strings.Join(tail, " | "))
	}
}
