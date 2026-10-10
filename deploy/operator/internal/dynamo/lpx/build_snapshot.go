/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	// ErrBuildSnapshotInconsistent identifies a build locator whose contents
	// cannot be treated as one immutable build snapshot.
	ErrBuildSnapshotInconsistent = errors.New("immutable build snapshot is inconsistent")
)

const (
	// Preserve the versioned digest domain so terminology changes do not alter
	// immutable build content identities.
	buildSnapshotIdentityVersion = "dynamo-lpx-compiler-snapshot/v3"
	// The existing Model Express transport admits messages up to this size.
	// Reuse that compatibility ceiling as one total retained control-plane
	// metadata budget instead of permitting an unbounded sum of streamed
	// chunks or local files.
	maxBuildSnapshotMetadataBytes = modelExpressMaxMessageSize
	buildSnapshotTimeout          = 30 * time.Second
)

// BuildSnapshot is a provider-fenced copy of the exact manifest consumed by
// the LPX workload projection.
type BuildSnapshot struct {
	ref           string
	contentID     string
	manifestBytes []byte
	format        buildContractFormat
	inventory     []string
}

// AcquireBuildSnapshot fences the selected compiler contract with inventories
// and duplicate reads. The receiver must be non-nil and is not mutated.
func (r *defaultModelRegistry) AcquireBuildSnapshot(ctx context.Context, id string) (*BuildSnapshot, error) {
	// Bound all metadata RPCs for one snapshot acquisition.
	ctx, cancel := context.WithTimeout(ctx, buildSnapshotTimeout)
	defer cancel()

	refURL, err := r.BuildURL(id)
	if err != nil {
		return nil, err
	}

	pre, err := r.listBuildFiles(ctx, refURL)
	if err != nil {
		return nil, err
	}

	// Select the unique binary entry point before either bounded read.
	format, err := selectBuildContract(pre)
	if err != nil {
		return nil, fmt.Errorf("%w: immutable build %q: %w", ErrBuildSnapshotInconsistent, refURL.String(), err)
	}
	manifestPath, err := format.filename()
	if err != nil {
		return nil, err
	}
	manifestData, readErr := r.readBuildFileBounded(ctx, refURL, manifestPath, maxBuildSnapshotMetadataBytes)
	if readErr != nil {
		if errors.Is(readErr, errBuildFileTooLarge) {
			return nil, fmt.Errorf(
				"%w: compiler metadata %s in %q exceeds its bounded acquisition budget: %w",
				ErrBuildSnapshotInconsistent,
				manifestPath,
				refURL.String(),
				readErr,
			)
		}
		if isDefinitiveBuildMemberAbsence(readErr) {
			return nil, fmt.Errorf(
				"%w: compiler metadata %s disappeared from %q after inventory: %w",
				ErrBuildSnapshotInconsistent,
				manifestPath,
				refURL.String(),
				readErr,
			)
		}
		return nil, readErr
	}

	// Verify the build inventory did not change after reading compiler metadata.
	post, err := r.listBuildFiles(ctx, refURL)
	if err != nil {
		if isDefinitiveBuildMemberAbsence(err) {
			return nil, fmt.Errorf(
				"%w: build inventory disappeared from %q while acquiring it: %w",
				ErrBuildSnapshotInconsistent,
				refURL.String(),
				err,
			)
		}
		return nil, err
	}
	if !slices.Equal(pre, post) {
		return nil, fmt.Errorf(
			"%w: build inventory changed while acquiring %q",
			ErrBuildSnapshotInconsistent,
			refURL.String(),
		)
	}

	// Re-read the compiler manifest after verifying the build inventory.
	verifiedData, readErr := r.readBuildFileBounded(
		ctx,
		refURL,
		manifestPath,
		maxBuildSnapshotMetadataBytes,
	)
	if readErr != nil {
		if errors.Is(readErr, errBuildFileTooLarge) {
			return nil, fmt.Errorf(
				"%w: compiler metadata %s grew beyond its bounded acquisition budget in %q: %w",
				ErrBuildSnapshotInconsistent,
				manifestPath,
				refURL.String(),
				readErr,
			)
		}
		if isDefinitiveBuildMemberAbsence(readErr) {
			return nil, fmt.Errorf(
				"%w: compiler metadata %s disappeared from %q after inventory verification: %w",
				ErrBuildSnapshotInconsistent,
				manifestPath,
				refURL.String(),
				readErr,
			)
		}
		return nil, readErr
	}
	if !bytes.Equal(manifestData, verifiedData) {
		return nil, fmt.Errorf(
			"%w: compiler metadata %s changed while acquiring %q",
			ErrBuildSnapshotInconsistent,
			manifestPath,
			refURL.String(),
		)
	}

	// Bind the stable inventory and manifest bytes into the snapshot identity.
	hash := sha256.New()
	writeBuildSnapshotField(hash, []byte(buildSnapshotIdentityVersion))
	writeBuildSnapshotField(hash, []byte("inventory"))
	writeBuildSnapshotField(hash, []byte(strconv.Itoa(len(pre))))
	for _, path := range pre {
		writeBuildSnapshotField(hash, []byte("inventory-path"))
		writeBuildSnapshotField(hash, []byte(path))
	}
	writeBuildSnapshotField(hash, []byte("compiler-metadata"))
	writeBuildSnapshotField(hash, []byte("1"))
	writeBuildSnapshotField(hash, []byte("metadata-path"))
	writeBuildSnapshotField(hash, []byte(manifestPath))
	writeBuildSnapshotField(hash, []byte("metadata-bytes"))
	writeBuildSnapshotField(hash, manifestData)
	return &BuildSnapshot{
		ref:           refURL.String(),
		contentID:     fmt.Sprintf("sha256:%x", hash.Sum(nil)),
		manifestBytes: manifestData,
		format:        format,
		inventory:     pre,
	}, nil
}

func isDefinitiveBuildMemberAbsence(err error) bool {
	return errors.Is(err, os.ErrNotExist) ||
		errors.Is(err, syscall.ENOTDIR) ||
		status.Code(err) == codes.NotFound
}

func writeBuildSnapshotField(hash interface{ Write([]byte) (int, error) }, value []byte) {
	_, _ = fmt.Fprintf(hash, "%d:", len(value))
	_, _ = hash.Write(value)
}

// NormalizedBuildSnapshot is the validated, source-independent projection input.
type NormalizedBuildSnapshot struct {
	contentID string
	build     *Build
	format    buildContractFormat
}

// normalizeBuildSnapshot validates and lowers a successfully acquired, non-nil
// snapshot into its normalized build representation. The snapshot is not mutated.
func normalizeBuildSnapshot(snapshot *BuildSnapshot) (NormalizedBuildSnapshot, error) {
	// Decode only the selected contract; malformed split data cannot reach the legacy reader.
	var build *Build
	var err error
	switch snapshot.format {
	case buildContractManifestV2:
		manifest, decodeErr := decodeGbuildManifestV2(snapshot.manifestBytes)
		if decodeErr != nil {
			return NormalizedBuildSnapshot{}, decodeErr
		}
		build, err = buildFromGbuildManifestV2(snapshot.ref, manifest)
	case buildContractDeploymentV1:
		deployment, decodeErr := decodeGbuildDeploymentV1(snapshot.manifestBytes)
		if decodeErr != nil {
			return NormalizedBuildSnapshot{}, decodeErr
		}
		build, err = buildFromGbuildDeploymentV1(snapshot.ref, deployment, snapshot.inventory)
	default:
		return NormalizedBuildSnapshot{}, fmt.Errorf("unsupported build contract format %d", snapshot.format)
	}
	if err != nil {
		return NormalizedBuildSnapshot{}, err
	}
	return NormalizedBuildSnapshot{contentID: snapshot.contentID, build: build, format: snapshot.format}, nil
}

func normalizeModelPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	path = filepath.ToSlash(filepath.Clean(path))
	return strings.TrimPrefix(path, "./")
}

// normalizeBuildFilePaths consumes a nonnil owned path slice and canonicalizes it in place.
func normalizeBuildFilePaths(ctx context.Context, paths []string) ([]string, error) {
	// Normalize and retain compiler evidence while validating every listed path.
	for index, rawPath := range paths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		path := normalizeModelPath(rawPath)
		if path == "" || path == "." || filepath.IsAbs(path) || path == ".." || strings.HasPrefix(path, "../") {
			return nil, fmt.Errorf("build inventory contains invalid relative path %q", rawPath)
		}
		paths[index] = path
	}

	// Canonical ordering makes duplicate paths adjacent without a second index.
	// Sorting itself is not interruptible; observe cancellation in the following checks.
	sort.Strings(paths)
	for i := 1; i < len(paths); i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if paths[i] == paths[i-1] {
			return nil, fmt.Errorf("build inventory repeats relative path %q", paths[i])
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return paths, nil
}
