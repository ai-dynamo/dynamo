// Package main provides the restore-entrypoint binary for self-restoring placeholder containers.
// This binary replaces the shell script restore-entrypoint.sh with a Go implementation
// that uses the go-criu library for CRIU operations.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/restore"
)

func main() {
	// Set up logging
	log := logrus.New()
	log.SetOutput(os.Stdout)
	log.SetFormatter(&logrus.TextFormatter{
		FullTimestamp:   true,
		TimestampFormat: "2006-01-02 15:04:05",
	})

	// Load configuration from environment
	cfg := restore.ConfigFromEnv()

	// Set log level based on DEBUG flag
	if cfg.Debug {
		log.SetLevel(logrus.DebugLevel)
	} else {
		log.SetLevel(logrus.InfoLevel)
	}

	entry := log.WithField("component", "restore-entrypoint")

	// Set up context with signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		sig := <-sigChan
		entry.WithField("signal", sig).Info("Received shutdown signal")
		cancel()
	}()

	// Run the restore entrypoint
	if err := restore.Run(ctx, cfg, entry); err != nil {
		entry.WithError(err).Fatal("Restore entrypoint failed")
	}
}
