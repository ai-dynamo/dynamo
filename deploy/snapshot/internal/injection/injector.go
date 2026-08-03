// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package injection stages the mounts required inside a placeholder container's
// mount namespace before CRIU restore can run.
package injection

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsbindmount"
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
		return errors.New("injection.Config: SourceDir must not be empty")
	}
	if c.DestinationDir == "" {
		return errors.New("injection.Config: DestinationDir must not be empty")
	}
	return nil
}

// InjectionHandle represents a live binary injection into a placeholder namespace.
// The caller must call Cleanup after nsrestore returns.
type InjectionHandle interface {
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
	Inject(ctx context.Context, pid int) (InjectionHandle, error)
}

// NSMountInjector implements Injector using a nsbindmount.Mounter.
type NSMountInjector struct {
	cfg     Config
	mounter nsbindmount.Mounter
	log     logr.Logger
}

// New returns an NSMountInjector. cfg is normalised with WithDefaults then
// validated; an error is returned if the result is invalid.
func New(cfg Config, mounter nsbindmount.Mounter, log logr.Logger) (*NSMountInjector, error) {
	cfg = cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &NSMountInjector{cfg: cfg, mounter: mounter, log: log}, nil
}

// Inject bind-mounts cfg.SourceDir into cfg.DestinationDir inside the mount
// namespace of pid. The caller must call InjectionHandle.Cleanup when done.
func (i *NSMountInjector) Inject(ctx context.Context, pid int) (InjectionHandle, error) {
	i.log.Info("injecting agent bundle into placeholder namespace", "pid", pid, "src", i.cfg.SourceDir, "dst", i.cfg.DestinationDir)

	handle, err := i.mounter.Mount(ctx, pid, i.cfg.SourceDir, i.cfg.DestinationDir, nsbindmount.MountOptions{ReadOnly: true})
	if err != nil {
		return nil, fmt.Errorf("mount agent bundle into placeholder: %w", err)
	}

	i.log.Info("agent bundle mounted", "pid", pid, "dst", handle.TargetPath())
	return &injectionHandle{mount: handle}, nil
}

// injectionHandle is the concrete InjectionHandle returned by Inject.
type injectionHandle struct {
	mount nsbindmount.MountHandle
}

func (h *injectionHandle) BinPath(name string) string {
	return filepath.Join(h.mount.TargetPath(), name)
}

func (h *injectionHandle) Cleanup(ctx context.Context) error {
	return h.mount.Unmount(ctx)
}
