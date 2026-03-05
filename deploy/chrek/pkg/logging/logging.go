// Package logging provides shared logger configuration for chrek binaries.
package logging

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-logr/logr"
	"github.com/go-logr/zapr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// ConfigureLogger creates a logr.Logger from the CHREK_LOG_LEVEL environment variable.
// output controls where log lines are written ("stdout" or "stderr").
// Supported levels: trace, debug, info, warn, error. Defaults to info.
func ConfigureLogger(output string) logr.Logger {
	level := strings.TrimSpace(strings.ToLower(os.Getenv("CHREK_LOG_LEVEL")))
	if level == "" {
		level = "info"
	}

	zapLevel := zapcore.InfoLevel
	var parseErr error
	switch level {
	case "trace", "debug":
		zapLevel = zapcore.DebugLevel
	case "info":
		zapLevel = zapcore.InfoLevel
	case "warn", "warning":
		zapLevel = zapcore.WarnLevel
	case "error":
		zapLevel = zapcore.ErrorLevel
	default:
		parseErr = fmt.Errorf("invalid level %q", level)
	}

	if output == "" {
		output = "stdout"
	}

	zapCfg := zap.Config{
		Level:            zap.NewAtomicLevelAt(zapLevel),
		Development:      false,
		Encoding:         "console",
		EncoderConfig:    zap.NewProductionEncoderConfig(),
		OutputPaths:      []string{output},
		ErrorOutputPaths: []string{"stderr"},
	}
	zapCfg.EncoderConfig.EncodeTime = zapcore.RFC3339NanoTimeEncoder
	zapCfg.EncoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder
	zapLog, err := zapCfg.Build()
	if err != nil {
		zapLog, _ = zap.NewDevelopment()
	}

	log := zapr.NewLogger(zapLog)
	if parseErr != nil {
		log.WithName("setup").Info("Invalid CHREK_LOG_LEVEL, falling back to info", "value", level, "error", parseErr)
	}
	return log
}
