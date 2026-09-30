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
	"net/url"
	"os"
	"syscall"
	"time"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
	format    buildContractFormat
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

	// Discover only the two supported entry points; never inventory payload files.
	entries, err := r.buildContractFiles(ctx, refURL)
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("%w: missing build contract: require %s or %s", os.ErrNotExist, gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile)
	}
	format, err := selectBuildContract(entries)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", errInvalidBuildManifest, err)
	}
	manifestPath, err := format.filename()
	if err != nil {
		return nil, err
	}
	manifestData, err := r.readBuildFileBounded(ctx, refURL, manifestPath, maxBuildSnapshotMetadataBytes)
	if err != nil {
		return nil, err
	}

	// Validate the selected contract exactly once, without fallback on invalid split bytes.
	var build *Build
	switch format {
	case buildContractManifestV2:
		manifest, decodeErr := decodeGbuildManifestV2(manifestData)
		if decodeErr != nil {
			err = decodeErr
		} else {
			build, err = buildFromGbuildManifestV2(refURL.String(), manifest)
		}
	case buildContractDeploymentV1:
		deployment, decodeErr := decodeGbuildDeploymentV1(manifestData)
		if decodeErr != nil {
			err = decodeErr
		} else {
			build, err = buildFromGbuildDeploymentV1(refURL.String(), deployment)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("%w for %q: %w", errInvalidBuildManifest, refURL.String(), err)
	}

	// Hash the exact manifest bytes in their own domain, independent of filenames and locator.
	hash := sha256.New()
	_, _ = hash.Write([]byte(buildSnapshotIdentityVersion + "\x00"))
	_, _ = hash.Write([]byte(manifestPath + "\x00"))
	_, _ = hash.Write(manifestData)
	return &BuildSnapshot{
		contentID: fmt.Sprintf("sha256:%x", hash.Sum(nil)),
		build:     build,
		format:    format,
	}, nil
}

// buildContractFiles discovers supported entry points without scanning local payload trees.
// The receiver and buildURL must be non-nil.
func (r *defaultModelRegistry) buildContractFiles(ctx context.Context, buildURL *url.URL) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	candidates := []string{gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile}
	if buildURL.Scheme == BuildSchemeFile {
		var paths []string
		for _, name := range candidates {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			_, err := os.Lstat(buildURL.JoinPath(name).Path)
			if isDefinitiveBuildMemberAbsence(err) {
				continue
			}
			if err != nil {
				return nil, err
			}
			paths = append(paths, name)
		}
		return paths, nil
	}
	if r.mxClient == nil {
		return nil, fmt.Errorf("Model Express client is required for GCS model registry reads of %q", buildURL.String())
	}
	// Ask Model Express only for supported control-plane entry points.
	listed, err := r.mxClient.ListModelFiles(ctx, &modelpb.ModelFilesRequest{
		ModelName: buildURL.String(), Provider: modelpb.ModelProvider_GCS,
		FileSelector: &modelpb.ModelFileSelector{Paths: candidates},
	})
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, file := range listed.GetFiles() {
		if name := file.GetRelativePath(); name == candidates[0] || name == candidates[1] {
			paths = append(paths, name)
		}
	}
	return paths, nil
}

func isDefinitiveBuildMemberAbsence(err error) bool {
	return errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) || status.Code(err) == codes.NotFound
}
