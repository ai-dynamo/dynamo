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

package consts

const (
	DefaultETCDTimeoutSeconds              = 5
	DefaultETCDDialKeepaliveTimeSeconds    = 30
	DefaultETCDDialKeepaliveTimeoutSeconds = 10

	HPADefaultMaxReplicas = 10

	HPACPUDefaultAverageUtilization = 80

	YataiDebugImg             = "yatai.ai/yatai-infras/debug"
	YataiKubectlNamespace     = "default"
	YataiKubectlContainerName = "main"
	YataiKubectlImage         = "yatai.ai/yatai-infras/k8s"

	TracingContextKey = "tracing-context"
	// nolint: gosec
	YataiApiTokenHeaderName = "X-YATAI-API-TOKEN"

	YataiOrganizationHeaderName = "X-Yatai-Organization"
	NgcOrganizationHeaderName   = "Nv-Ngc-Org"
	NgcUserHeaderName           = "Nv-Actor-Id"

	DefaultUserId = "default"
	DefaultOrgId  = "default"

	BentoServicePort          = 3000
	BentoContainerDefaultPort = 3000
	BentoServicePortName      = "http"
	BentoContainerPortName    = "http"

	NoneStr = "None"

	AmazonS3Endpoint = "s3.amazonaws.com"

	YataiImageBuilderComponentName = "yatai-image-builder"
	YataiDeploymentComponentName   = "yatai-deployment"

	// nolint: gosec
	YataiK8sBotApiTokenName = "yatai-k8s-bot"

	YataiBentoDeploymentComponentApiServer = "api-server"
	YataiBentoDeploymentComponentRunner    = "runner"

	InternalImagesBentoDownloaderDefault    = "quay.io/bentoml/bento-downloader:0.0.3"
	InternalImagesCurlDefault               = "quay.io/bentoml/curl:0.0.1"
	InternalImagesKanikoDefault             = "quay.io/bentoml/kaniko:1.9.1"
	InternalImagesMetricsTransformerDefault = "quay.io/bentoml/yatai-bento-metrics-transformer:0.0.3"
	InternalImagesBuildkitDefault           = "quay.io/bentoml/buildkit:master"
	InternalImagesBuildkitRootlessDefault   = "quay.io/bentoml/buildkit:master-rootless"
	InternalImagesBuildahDefault            = "quay.io/bentoml/bentoml-buildah:0.0.1"
)
