// Package criulog provides shared CRIU log reading and error reporting utilities.
package criulog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

const (
	defaultLogCopyPerm = 0644
	defaultLogDirPerm  = 0755
)

// LogErrors reads a CRIU log file and logs all lines as errors.
// If CRIU_LOG_DIR environment variable is set, copies the log there.
func LogErrors(logPath string, logger interface{}) {
	data, err := os.ReadFile(logPath)
	if err != nil {
		logWarning(logger, err, logPath)
		return
	}

	logError(logger, "=== CRIU RESTORE LOG START ===")
	for _, line := range strings.Split(string(data), "\n") {
		if line != "" {
			logError(logger, line)
		}
	}
	logError(logger, "=== CRIU RESTORE LOG END ===")

	copyLogToSharedDir(data, logger)
}

// LogErrorsWithWorkDir reads CRIU log from workDir if set, otherwise from checkpointPath.
func LogErrorsWithWorkDir(checkpointPath, workDir, logFile string, logger interface{}) {
	logBase := checkpointPath
	if workDir != "" {
		logBase = workDir
	}
	LogErrors(filepath.Join(logBase, logFile), logger)
}

// copyLogToSharedDir copies log data to CRIU_LOG_DIR if configured.
func copyLogToSharedDir(data []byte, logger interface{}) {
	logDir := os.Getenv("CRIU_LOG_DIR")
	if logDir == "" {
		return
	}

	if err := os.MkdirAll(logDir, defaultLogDirPerm); err != nil {
		return
	}

	destPath := filepath.Join(logDir, fmt.Sprintf("restore-%d.log", time.Now().Unix()))
	if err := os.WriteFile(destPath, data, defaultLogCopyPerm); err == nil {
		logInfo(logger, "CRIU log copied to shared directory", destPath)
	}
}

// logError logs an error message, adapting to either logrus.Entry or logrus.Logger.
func logError(logger interface{}, msg string) {
	switch l := logger.(type) {
	case *logrus.Entry:
		l.Error(msg)
	case *logrus.Logger:
		l.Error(msg)
	}
}

// logWarning logs a warning with error and path field.
func logWarning(logger interface{}, err error, path string) {
	switch l := logger.(type) {
	case *logrus.Entry:
		l.WithError(err).WithField("path", path).Warn("Could not read CRIU log file")
	case *logrus.Logger:
		l.WithError(err).WithField("path", path).Warn("Could not read CRIU log file")
	}
}

// logInfo logs an info message with path field.
func logInfo(logger interface{}, msg, path string) {
	switch l := logger.(type) {
	case *logrus.Entry:
		l.WithField("path", path).Info(msg)
	case *logrus.Logger:
		l.WithField("path", path).Info(msg)
	}
}
