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
	EnvYataiEndpoint    = "YATAI_ENDPOINT"
	EnvYataiClusterName = "YATAI_CLUSTER_NAME"
	// nolint: gosec
	EnvYataiApiToken = "YATAI_API_TOKEN"

	EnvBentoServicePort = "PORT"

	// tracking envars
	EnvYataiVersion       = "YATAI_T_VERSION"
	EnvYataiOrgUID        = "YATAI_T_ORG_UID"
	EnvYataiDeploymentUID = "YATAI_T_DEPLOYMENT_UID"
	EnvYataiClusterUID    = "YATAI_T_CLUSTER_UID"

	EnvYataiBentoDeploymentName      = "YATAI_BENTO_DEPLOYMENT_NAME"
	EnvYataiBentoDeploymentNamespace = "YATAI_BENTO_DEPLOYMENT_NAMESPACE"
	EnvDynamoDeploymentTag           = "DYNAMO_DEPLOYMENT_TAG"

	EnvS3Endpoint   = "S3_ENDPOINT"
	EnvS3Region     = "S3_REGION"
	EnvS3BucketName = "S3_BUCKET_NAME"
	EnvS3AccessKey  = "S3_ACCESS_KEY"
	// nolint:gosec
	EnvS3SecretKey = "S3_SECRET_KEY"
	EnvS3Secure    = "S3_SECURE"

	EnvDockerRegistryServer          = "DOCKER_REGISTRY_SERVER"
	EnvDockerRegistryInClusterServer = "DOCKER_REGISTRY_IN_CLUSTER_SERVER"
	EnvDockerRegistryUsername        = "DOCKER_REGISTRY_USERNAME"
	// nolint:gosec
	EnvDockerRegistryPassword            = "DOCKER_REGISTRY_PASSWORD"
	EnvDockerRegistrySecure              = "DOCKER_REGISTRY_SECURE"
	EnvDockerRegistryBentoRepositoryName = "DOCKER_REGISTRY_BENTO_REPOSITORY_NAME"
	EnvDockerRegistryModelRepositoryName = "DOCKER_REGISTRY_MODEL_REPOSITORY_NAME"

	EnvInternalImagesBentoDownloader    = "INTERNAL_IMAGES_BENTO_DOWNLOADER"
	EnvInternalImagesCurl               = "INTERNAL_IMAGES_CURL"
	EnvInternalImagesKaniko             = "INTERNAL_IMAGES_KANIKO"
	EnvInternalImagesMetricsTransformer = "INTERNAL_IMAGES_METRICS_TRANSFORMER"
	EnvInternalImagesBuildkit           = "INTERNAL_IMAGES_BUILDKIT"
	EnvInternalImagesBuildkitRootless   = "INTERNAL_IMAGES_BUILDKIT_ROOTLESS"
	EnvInternalImagesBuildah            = "INTERNAL_IMAGES_BUILDAH"

	EnvYataiSystemNamespace       = "YATAI_SYSTEM_NAMESPACE"
	EnvYataiImageBuilderNamespace = "YATAI_IMAGE_BUILDER_NAMESPACE"
	EnvYataiDeploymentNamespace   = "YATAI_DEPLOYMENT_NAMESPACE"
	EnvBentoDeploymentNamespaces  = "BENTO_DEPLOYMENT_NAMESPACES"
	EnvImageBuildersNamespace     = "IMAGE_BUILDERS_NAMESPACE"

	EnvAWSAccessKeyID     = "AWS_ACCESS_KEY_ID"
	EnvGCPAccessKeyID     = "GCP_ACCESS_KEY_ID"
	EnvAWSSecretAccessKey = "AWS_SECRET_ACCESS_KEY"
	EnvGCPSecretAccessKey = "GCP_SECRET_ACCESS_KEY"

	EnvAWSECRWithIAMRole = "AWS_ECR_WITH_IAM_ROLE"
	EnvAWSECRRegion      = "AWS_ECR_REGION"
)
