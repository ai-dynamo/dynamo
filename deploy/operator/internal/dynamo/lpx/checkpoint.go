/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
)

// Checkpoint is a Hugging Face model repository pinned to a commit. Only
// checkpointFromDeploymentV1 constructs one, so its fields are validated.
type Checkpoint struct {
	// Model is the Hugging Face repository ID, such as openai/gpt-oss-20b.
	Model string
	// Revision is the full 40-character commit SHA of the repository snapshot.
	Revision string
}

// Key returns the download-status identity of a non-nil checkpoint.
func (c Checkpoint) Key() string {
	return c.Model + "@" + c.Revision
}

// SnapshotPath returns the directory that holds a non-nil validated
// checkpoint's files. Model Express does not report local paths, so this
// follows the Hugging Face cache layout that Model Express writes at the root
// of model storage: models--<org>--<name>/snapshots/<commit>. Build-contract
// validation restricts model and revision to safe path components.
func (c Checkpoint) SnapshotPath(modelStoragePath string) string {
	repository := "models--" + strings.ReplaceAll(c.Model, "/", "--")
	return filepath.Join(modelStoragePath, repository, "snapshots", c.Revision)
}

// These follow the Hugging Face repository-ID and commit rules.
var (
	checkpointModelPattern    = regexp.MustCompile(`^([A-Za-z0-9_]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?/)?[A-Za-z0-9_]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?$`)
	checkpointRevisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// checkpointFromDeploymentV1 returns the deployment contract's validated
// checkpoint, or nil when the build records none.
func checkpointFromDeploymentV1(manifest deploymentcapnpv1.Deployment) (*Checkpoint, error) {
	if !manifest.HasCheckpoint() {
		return nil, nil
	}
	raw, err := manifest.Checkpoint()
	if err != nil {
		return nil, fmt.Errorf("reading %s checkpoint: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if raw.Provider() != deploymentcapnpv1.CheckpointProvider_huggingface {
		return nil, fmt.Errorf("%s checkpoint.provider %d is not supported", gbuildDeploymentV1CapnpFile, raw.Provider())
	}
	model, err := raw.Model()
	if err != nil {
		return nil, fmt.Errorf("reading %s checkpoint.model: %w", gbuildDeploymentV1CapnpFile, err)
	}
	revision, err := raw.Revision()
	if err != nil {
		return nil, fmt.Errorf("reading %s checkpoint.revision: %w", gbuildDeploymentV1CapnpFile, err)
	}

	// Reject identities that Hugging Face rejects or that escape the cache layout.
	if !checkpointModelPattern.MatchString(model) || strings.Contains(model, "--") ||
		strings.Contains(model, "..") || strings.HasSuffix(model, ".git") {
		return nil, fmt.Errorf("%s checkpoint.model %q is not a valid Hugging Face repository ID", gbuildDeploymentV1CapnpFile, model)
	}
	if !checkpointRevisionPattern.MatchString(revision) {
		return nil, fmt.Errorf("%s checkpoint.revision %q is not a full commit SHA", gbuildDeploymentV1CapnpFile, revision)
	}
	return &Checkpoint{Model: model, Revision: revision}, nil
}

// WithoutBuildCheckpoints returns a registry whose builds record no checkpoint,
// resolved workloads neither download nor project one. registry is non-nil.
func WithoutBuildCheckpoints(registry ModelRegistry) ModelRegistry {
	return withoutBuildCheckpoints{ModelRegistry: registry}
}

type withoutBuildCheckpoints struct {
	ModelRegistry
}

func (r withoutBuildCheckpoints) AcquireBuild(ctx context.Context, id string) (*Build, error) {
	build, err := r.ModelRegistry.AcquireBuild(ctx, id)
	if err != nil || (build.checkpoint == nil) {
		return build, err
	}
	stripped := *build
	stripped.checkpoint = nil
	return &stripped, nil
}
