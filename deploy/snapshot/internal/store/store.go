// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package store mirrors snapshot checkpoint directory trees to and from a remote
// object store. A checkpoint artifact is a directory (manifest.yaml plus CRIU
// images, overlay diffs, and CUDA state). Mirroring that tree to object storage
// lets checkpoint and restore pods share artifacts without a RWX volume,
// enabling scalable, multi-cluster checkpoint storage on any S3 endpoint.
//
// TODO(future): the artifact data path runs over HTTP today. minio-go already
// supports an RDMA / GPUDirect transport — wire it up for artifact transfers to
// cut restore latency and bypass the host bounce buffer where the object store
// and fabric support it.
package store

import (
	"context"
	"fmt"
	"strings"
)

// Backend identifiers. The empty string and BackendPVC both mean "no remote
// store" — the local staging directory is the durable store (legacy behavior).
const (
	BackendPVC = "pvc"
	BackendS3  = "s3"
)

// ArtifactStore mirrors a local checkpoint directory tree to and from a remote
// object store. Implementations key objects under a caller-supplied prefix that
// already encodes the checkpoint ID and artifact version
// (e.g. "<checkpointId>/versions/<n>").
type ArtifactStore interface {
	// Upload mirrors every regular file under localDir to keyPrefix. The
	// manifest object is written last so Exists only reports true once the
	// full artifact has landed.
	Upload(ctx context.Context, localDir, keyPrefix string) error
	// Download mirrors every object under keyPrefix into localDir, creating
	// parent directories as needed.
	Download(ctx context.Context, keyPrefix, localDir string) error
	// Exists reports whether a complete artifact (its manifest) is present
	// under keyPrefix.
	Exists(ctx context.Context, keyPrefix string) (bool, error)
	// Remove deletes every object under keyPrefix. A missing prefix is not an
	// error.
	Remove(ctx context.Context, keyPrefix string) error
	// Backend names the store implementation, for logging.
	Backend() string
}

// Config selects and configures an artifact store backend.
type Config struct {
	// Backend is the store type. "" or "pvc" means no remote store.
	Backend string
	// S3 configures the S3 backend when Backend is "s3".
	S3 S3Config
}

// New builds the artifact store for cfg. It returns (nil, nil) when no remote
// backend is configured, signaling callers to use the local staging directory
// directly (the PVC-backed legacy path).
func New(cfg Config) (ArtifactStore, error) {
	switch strings.TrimSpace(cfg.Backend) {
	case "", BackendPVC:
		return nil, nil
	case BackendS3:
		return newS3Store(cfg.S3)
	default:
		return nil, fmt.Errorf("unsupported artifact store backend %q", cfg.Backend)
	}
}

// normalizePrefix trims surrounding slashes and appends a single trailing slash
// so object keys join cleanly. An empty prefix stays empty.
func normalizePrefix(p string) string {
	p = strings.Trim(strings.TrimSpace(p), "/")
	if p == "" {
		return ""
	}
	return p + "/"
}
