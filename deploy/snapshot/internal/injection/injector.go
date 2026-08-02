// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package injection stages the mounts required inside a placeholder container's
// mount namespace before CRIU restore can run.
package injection

import (
	"context"
	"fmt"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsbindmount"
)

// Injector stages all mounts needed inside a placeholder container's namespace
// before CRIU restore runs. It holds the agent binary bundle, CRIU libs, and
// checkpoint directory — all read-only.
type Injector struct {
	mounter nsbindmount.Mounter
	log     logr.Logger
}

// New returns an Injector backed by the given Mounter.
func New(mounter nsbindmount.Mounter, log logr.Logger) *Injector {
	return &Injector{mounter: mounter, log: log}
}

// Inject bind-mounts the agent binary bundle (binaries, CRIU, and all
// dependencies) from agentBinDir into SnapshotBinDir inside the mount
// namespace of pid. Returns a cleanup func that unmounts it.
func (i *Injector) Inject(ctx context.Context, pid int) (func() error, error) {
	i.log.Info("injecting agent bundle into placeholder namespace", "pid", pid)

	cleanup, err := i.mounter.Mount(ctx, pid, agentBinDir, SnapshotBinDir, nsbindmount.MountOptions{ReadOnly: true})
	if err != nil {
		return nil, fmt.Errorf("mount agent bundle into placeholder: %w", err)
	}

	i.log.Info("agent bundle mounted", "pid", pid, "dst", SnapshotBinDir)
	return cleanup, nil
}
