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
	"strings"

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

func CheckpointJobName(hash, version string) string {
	version = strings.TrimSpace(version)
	if version == "" {
		version = DefaultArtifactVersion
	}
	return "checkpoint-job-" + hash + "-" + version
}
