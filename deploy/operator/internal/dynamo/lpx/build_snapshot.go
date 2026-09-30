/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"time"
)

// errInvalidBuildManifest separates invalid compiler input from acquisition failures.
var errInvalidBuildManifest = errors.New("invalid build manifest")

const (
	// This domain identifies snapshots of manifest bytes without a file inventory.
	buildSnapshotIdentityVersion = "dynamo-lpx-compiler-snapshot/v4"
	// Bound the single manifest to the Model Express transport's metadata ceiling.
	maxBuildSnapshotMetadataBytes = modelExpressMaxMessageSize
	buildSnapshotTimeout          = 30 * time.Second
)

// BuildSnapshot pairs a validated build with the digest of its compiler manifest.
// Callers must treat the build as immutable and copy it before runtime configuration.
type BuildSnapshot struct {
	contentID string
	build     *Build
}

// AcquireBuildSnapshot reads and validates the immutable compiler manifest once.
// The receiver must be non-nil and is not mutated. Payload files are not inspected.
func (r *defaultModelRegistry) AcquireBuildSnapshot(ctx context.Context, id string) (*BuildSnapshot, error) {
	// Bound the manifest read and resolve the build's provider-specific locator.
	ctx, cancel := context.WithTimeout(ctx, buildSnapshotTimeout)
	defer cancel()
	refURL, err := r.BuildURL(id)
	if err != nil {
		return nil, err
	}

	// Fetch only the required manifest, preserving read and transport failures.
	manifestData, err := r.readBuildFileBounded(ctx, refURL, gbuildManifestV2CapnpFile, maxBuildSnapshotMetadataBytes)
	if err != nil {
		return nil, err
	}

	// Validate compiler input before publishing a snapshot to projection consumers.
	manifest, err := decodeGbuildManifestV2(manifestData)
	if err != nil {
		return nil, fmt.Errorf("%w for %q: %w", errInvalidBuildManifest, refURL.String(), err)
	}
	build, err := buildFromGbuildManifestV2(refURL.String(), manifest)
	if err != nil {
		return nil, fmt.Errorf("%w for %q: %w", errInvalidBuildManifest, refURL.String(), err)
	}

	// Hash the exact manifest bytes in their own domain, independent of filenames and locator.
	hash := sha256.New()
	_, _ = hash.Write([]byte(buildSnapshotIdentityVersion + "\x00"))
	_, _ = hash.Write(manifestData)
	return &BuildSnapshot{
		contentID: fmt.Sprintf("sha256:%x", hash.Sum(nil)),
		build:     build,
	}, nil
}
