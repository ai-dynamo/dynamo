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

package crds

import (
	"time"

	corev1 "k8s.io/api/core/v1"
)

type DynamoNimRequestData struct {
	DynamoNimVersionTag string `json:"bentoTag"`
	DownloadURL           string `json:"downloadUrl,omitempty"`

	ImageBuildTimeout *time.Duration `json:"imageBuildTimeout,omitempty"`

	ImageBuilderExtraPodMetadata   *ExtraPodMetadata            `json:"imageBuilderExtraPodMetadata,omitempty"`
	ImageBuilderExtraPodSpec       *ExtraPodSpec                `json:"imageBuilderExtraPodSpec,omitempty"`
	ImageBuilderExtraContainerEnv  []corev1.EnvVar              `json:"imageBuilderExtraContainerEnv,omitempty"`
	ImageBuilderContainerResources *corev1.ResourceRequirements `json:"imageBuilderContainerResources,omitempty"`

	DockerConfigJSONSecretName string `json:"dockerConfigJsonSecretName,omitempty"`

	DownloaderContainerEnvFrom []corev1.EnvFromSource `json:"downloaderContainerEnvFrom,omitempty"`
}

type DynamoNimRequestConfigurationV1Alpha1 struct {
	Data    DynamoNimRequestData `json:"data,omitempty"`
	Version string                 `json:"version"`
}
