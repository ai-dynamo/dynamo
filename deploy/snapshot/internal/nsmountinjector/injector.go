// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nsmountinjector stages the agent's restore binaries inside a
// placeholder container's mount namespace before CRIU restore can run.
//
// The agent's binary bundle is bind-mounted read-only from agentBinDir into
// SnapshotBinDir inside the target namespace, via the ns-bind-mount C helper
// (cmd/ns-bind-mount). The placeholder image ships none of these binaries.
package nsmountinjector

import (
	"context"
	"errors"
	"path/filepath"

	"github.com/go-logr/logr"
)

const (
	// agentBinDir is the host-side directory in the agent image holding the
	// binaries to inject.
	agentBinDir = "/snapshot-binaries"

	// SnapshotBinDir is where that bundle appears inside the placeholder
	// namespace. This is the bundle-layout contract: nsrestore resolves criu,
	// the criu plugin directory, and the cuda-checkpoint helpers underneath it,
	// and deploy/snapshot/Dockerfile builds the bundle to match.
	SnapshotBinDir = "/tmp" + agentBinDir
)

// Config is the static configuration for binary injection.
type Config struct {
	// SourceDir is the host-side directory where agent binaries live.
	SourceDir string
	// DestinationDir is the path inside the placeholder namespace where binaries appear.
	DestinationDir string
}

// WithDefaults returns a copy of c with empty fields filled from package defaults.
func (c Config) WithDefaults() Config {
	if c.SourceDir == "" {
		c.SourceDir = agentBinDir
	}
	if c.DestinationDir == "" {
		c.DestinationDir = SnapshotBinDir
	}
	return c
}

// Validate returns an error if any required Config field is empty.
func (c Config) Validate() error {
	if c.SourceDir == "" {
		return errors.New("nsmountinjector.Config: SourceDir must not be empty")
	}
	if c.DestinationDir == "" {
		return errors.New("nsmountinjector.Config: DestinationDir must not be empty")
	}
	return nil
}

// Handle represents a live binary injection into a placeholder namespace.
// The caller must call Cleanup after nsrestore returns.
type Handle interface {
	// BinPath returns the in-namespace absolute path to the named binary.
	// Example: handle.BinPath("nsrestore") → "/tmp/snapshot-binaries/nsrestore"
	BinPath(name string) string

	// Cleanup unmounts the injected directory from the target namespace.
	// Idempotent — safe to call from a defer even if Inject partially failed.
	Cleanup(ctx context.Context) error
}

// Injector mounts the agent's restore binaries into a placeholder container's
// mount namespace and returns a handle for cleanup.
type Injector interface {
	Inject(ctx context.Context, pid int) (Handle, error)
}

// NSMountInjector implements Injector on top of the ns-bind-mount helper.
type NSMountInjector struct {
	cfg     Config
	mounter mounter
	log     logr.Logger
}

// New returns an NSMountInjector backed by the ns-bind-mount binary at its
// default location. cfg is normalised with WithDefaults then validated.
// It errors if cfg is invalid or the helper binary is missing, so a
// misconfigured node fails at startup rather than at the first restore.
func New(cfg Config, log logr.Logger) (*NSMountInjector, error) {
	m, err := newExecMounter(defaultBinaryPath, log)
	if err != nil {
		return nil, err
	}
	return newWithMounter(cfg, m, log)
}

// newWithMounter is the test seam: it takes an arbitrary mounter so tests can
// exercise Inject without the ns-bind-mount subprocess.
func newWithMounter(cfg Config, m mounter, log logr.Logger) (*NSMountInjector, error) {
	cfg = cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &NSMountInjector{cfg: cfg, mounter: m, log: log}, nil
}

// Inject bind-mounts cfg.SourceDir into cfg.DestinationDir inside the mount
// namespace of pid. The caller must call Handle.Cleanup when done.
func (i *NSMountInjector) Inject(ctx context.Context, pid int) (Handle, error) {
	i.log.Info("injecting agent bundle into placeholder namespace", "pid", pid, "src", i.cfg.SourceDir, "dst", i.cfg.DestinationDir)

	handle, err := i.mounter.Mount(ctx, pid, i.cfg.SourceDir, i.cfg.DestinationDir, MountOptions{ReadOnly: true})
	if err != nil {
		return nil, err
	}

	i.log.Info("agent bundle mounted", "pid", pid, "dst", handle.TargetPath())
	return &injectionHandle{mount: handle}, nil
}

// injectionHandle adapts a MountHandle to the bundle-oriented Handle surface.
type injectionHandle struct {
	mount MountHandle
}

func (h *injectionHandle) BinPath(name string) string {
	return filepath.Join(h.mount.TargetPath(), name)
}

func (h *injectionHandle) Cleanup(ctx context.Context) error {
	return h.mount.Unmount(ctx)
}
