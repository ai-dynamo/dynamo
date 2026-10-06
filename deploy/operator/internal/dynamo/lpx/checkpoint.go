/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"path/filepath"
	"strings"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

const cyborgWeightsPathEnv = "CYBORG_WEIGHTS_PATH"

// CheckpointKey returns the download-status identity of a non-nil checkpoint.
func CheckpointKey(checkpoint *dynamov1beta1.LPXCheckpoint) string {
	return checkpoint.Model + "@" + checkpoint.Revision
}

// checkpointSnapshotPath returns the directory that holds a non-nil admitted
// checkpoint's files. Model Express does not report local paths, so this
// follows the Hugging Face cache layout that Model Express writes at the root
// of model storage: models--<org>--<name>/snapshots/<commit>. Admission
// restricts model and revision to safe path components.
func checkpointSnapshotPath(checkpoint *dynamov1beta1.LPXCheckpoint, modelStoragePath string) string {
	repository := "models--" + strings.ReplaceAll(checkpoint.Model, "/", "--")
	return filepath.Join(modelStoragePath, repository, "snapshots", checkpoint.Revision)
}

// applyCyborgWeightsPath projects the projection's checkpoint directory into one
// Cyborg container. Without a checkpoint, the container's authored environment
// is unchanged.
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
