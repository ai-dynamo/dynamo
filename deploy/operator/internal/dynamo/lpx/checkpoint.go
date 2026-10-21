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
	corev1 "k8s.io/api/core/v1"
)

const cyborgWeightsPathEnv = "CYBORG_WEIGHTS_PATH"

// Checkpoint is a Hugging Face model repository pinned to a commit. Only
// checkpointFromDeploymentV1 constructs one, so its fields are validated.
type Checkpoint struct {
	// Model is the Hugging Face repository ID, such as openai/gpt-oss-20b.
	Model string
	// Revision is the full 40-character commit SHA of the repository snapshot.
	Revision string
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
// valid or not, so resolved workloads neither download, project, nor reject
// one. registry is non-nil.
func WithoutBuildCheckpoints(registry ModelRegistry) ModelRegistry {
	return withoutBuildCheckpoints{ModelRegistry: registry}
}

type withoutBuildCheckpoints struct {
	ModelRegistry
}

func (r withoutBuildCheckpoints) AcquireBuild(ctx context.Context, id string) (*Build, error) {
	build, err := r.ModelRegistry.AcquireBuild(ctx, id)
	if err != nil || (build.checkpoint == nil && build.checkpointErr == nil) {
		return build, err
	}
	stripped := *build
	stripped.checkpoint, stripped.checkpointErr = nil, nil
	return &stripped, nil
}

// CheckpointKey returns the download-status identity of a non-nil checkpoint.
func CheckpointKey(checkpoint *Checkpoint) string {
	return checkpoint.Model + "@" + checkpoint.Revision
}

// checkpointSnapshotPath returns the directory that holds a non-nil validated
// checkpoint's files. Model Express does not report local paths, so this
// follows the Hugging Face cache layout that Model Express writes at the root
// of model storage: models--<org>--<name>/snapshots/<commit>. Build-contract
// validation restricts model and revision to safe path components.
func checkpointSnapshotPath(checkpoint *Checkpoint, modelStoragePath string) string {
	repository := "models--" + strings.ReplaceAll(checkpoint.Model, "/", "--")
	return filepath.Join(modelStoragePath, repository, "snapshots", checkpoint.Revision)
}

func applyCyborgWeightsPath(container *corev1.Container, projection *ModelProjection, modelStoragePath string) {
	if projection.checkpoint == nil {
		return
	}

	// Kubernetes expands environment references in order; publish the checkpoint before authored bindings.
	env := make([]corev1.EnvVar, 0, len(container.Env)+1)
	env = append(env, corev1.EnvVar{
		Name:  cyborgWeightsPathEnv,
		Value: checkpointSnapshotPath(projection.checkpoint, modelStoragePath),
	})
	for _, variable := range container.Env {
		if variable.Name != cyborgWeightsPathEnv {
			env = append(env, variable)
		}
	}
	container.Env = env
}
