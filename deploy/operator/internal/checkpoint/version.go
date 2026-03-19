/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package checkpoint

import (
	"path/filepath"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const DefaultArtifactVersion = "1"

func ArtifactVersion(obj metav1.Object) string {
	if obj == nil {
		return DefaultArtifactVersion
	}
	annotations := obj.GetAnnotations()
	if annotations != nil {
		version := strings.TrimSpace(annotations[consts.KubeAnnotationCheckpointArtifactVersion])
		if version != "" {
			return version
		}
	}
	return DefaultArtifactVersion
}

func EffectiveArtifactVersion(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) string {
	if ckpt == nil {
		return DefaultArtifactVersion
	}
	if version := strings.TrimSpace(ArtifactVersion(ckpt)); version != DefaultArtifactVersion || hasExplicitArtifactVersion(ckpt) {
		return version
	}
	if version := versionFromJobName(ckpt.Status.JobName, ckpt.Status.IdentityHash); version != "" {
		return version
	}
	if version := versionFromLocation(ckpt.Status.Location); version != "" {
		return version
	}
	if ckpt.Status.JobName != "" || ckpt.Status.Location != "" || ckpt.Status.CreatedAt != nil {
		return ""
	}
	return DefaultArtifactVersion
}

func ObservedArtifactVersion(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) string {
	if ckpt == nil {
		return DefaultArtifactVersion
	}
	if version := versionFromJobName(ckpt.Status.JobName, ckpt.Status.IdentityHash); version != "" {
		return version
	}
	if version := versionFromLocation(ckpt.Status.Location); version != "" {
		return version
	}
	if ckpt.Status.JobName != "" || ckpt.Status.Location != "" || ckpt.Status.CreatedAt != nil {
		return ""
	}
	return EffectiveArtifactVersion(ckpt)
}

func DesiredCheckpointJobName(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, hash string) string {
	return CheckpointJobName(hash, EffectiveArtifactVersion(ckpt))
}

func CheckpointJobName(hash, version string) string {
	if version == "" {
		return "checkpoint-job-" + hash
	}
	return "checkpoint-job-" + hash + "-" + version
}

func hasExplicitArtifactVersion(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) bool {
	if ckpt == nil || ckpt.Annotations == nil {
		return false
	}
	return strings.TrimSpace(ckpt.Annotations[consts.KubeAnnotationCheckpointArtifactVersion]) != ""
}

func versionFromJobName(jobName, hash string) string {
	if jobName == "" || hash == "" {
		return ""
	}
	legacyJobName := "checkpoint-job-" + hash
	if jobName == legacyJobName {
		return ""
	}
	prefix := legacyJobName + "-"
	if !strings.HasPrefix(jobName, prefix) {
		return ""
	}
	version := strings.TrimSpace(strings.TrimPrefix(jobName, prefix))
	if version == "" {
		return ""
	}
	return version
}

func versionFromLocation(location string) string {
	if location == "" {
		return ""
	}
	dir := filepath.Clean(location)
	version := filepath.Base(dir)
	if version == "." || version == string(filepath.Separator) || version == "versions" {
		return ""
	}
	if filepath.Base(filepath.Dir(dir)) != "versions" {
		return ""
	}
	return version
}
