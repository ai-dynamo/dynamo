/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	corev1 "k8s.io/api/core/v1"
)

const cyborgWeightsPathEnv = "CYBORG_WEIGHTS_PATH"

// These match the LPXCheckpoint admission rules, which a manifest checkpoint does not pass through.
var (
	checkpointModelPattern    = regexp.MustCompile(`^([A-Za-z0-9_]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?/)?[A-Za-z0-9_]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?$`)
	checkpointRevisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// checkpointFromManifestV2 returns the manifest's validated checkpoint, or nil
// when the manifest records none.
func checkpointFromManifestV2(manifest manifestcapnpv2.Manifest) (*dynamov1beta1.LPXCheckpoint, error) {
	if !manifest.HasCheckpoint() {
		return nil, nil
	}
	raw, err := manifest.Checkpoint()
	if err != nil {
		return nil, fmt.Errorf("reading %s checkpoint: %w", gbuildManifestV2CapnpFile, err)
	}
	if raw.Provider() != manifestcapnpv2.CheckpointProvider_huggingFace {
		return nil, fmt.Errorf("%s checkpoint.provider %d is not supported", gbuildManifestV2CapnpFile, raw.Provider())
	}
	model, err := raw.Model()
	if err != nil {
		return nil, fmt.Errorf("reading %s checkpoint.model: %w", gbuildManifestV2CapnpFile, err)
	}
	revision, err := raw.Revision()
	if err != nil {
		return nil, fmt.Errorf("reading %s checkpoint.revision: %w", gbuildManifestV2CapnpFile, err)
	}

	// Reject identities that Hugging Face rejects or that escape the cache layout.
	if !checkpointModelPattern.MatchString(model) || strings.Contains(model, "--") ||
		strings.Contains(model, "..") || strings.HasSuffix(model, ".git") {
		return nil, fmt.Errorf("%s checkpoint.model %q is not a valid Hugging Face repository ID", gbuildManifestV2CapnpFile, model)
	}
	if !checkpointRevisionPattern.MatchString(revision) {
		return nil, fmt.Errorf("%s checkpoint.revision %q is not a full commit SHA", gbuildManifestV2CapnpFile, revision)
	}
	return &dynamov1beta1.LPXCheckpoint{
		Provider: dynamov1beta1.LPXCheckpointProviderHuggingFace,
		Model:    model,
		Revision: revision,
	}, nil
}

// CheckpointKey returns the download-status identity of a non-nil checkpoint.
func CheckpointKey(checkpoint *dynamov1beta1.LPXCheckpoint) string {
	return checkpoint.Model + "@" + checkpoint.Revision
}

// checkpointSnapshotPath returns the directory that holds a non-nil validated
// checkpoint's files. Model Express does not report local paths, so this
// follows the Hugging Face cache layout that Model Express writes at the root
// of model storage: models--<org>--<name>/snapshots/<commit>. Admission and
// manifest validation restrict model and revision to safe path components.
func checkpointSnapshotPath(checkpoint *dynamov1beta1.LPXCheckpoint, modelStoragePath string) string {
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
