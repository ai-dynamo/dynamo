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

package config

import (
	"context"
	"os"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/consts"
)

func GetDynamoImageBuilderNamespace(ctx context.Context) (namespace string, err error) {
	return os.Getenv(consts.EnvDynamoImageBuilderNamespace), nil
}

type DockerRegistryConfig struct {
	DynamoComponentsRepositoryName string `yaml:"dynamo_components_repository_name"`
	Server                         string `yaml:"server"`
	SecretName                     string `yaml:"secret_name"`
	Secure                         bool   `yaml:"secure"`
}

func GetDockerRegistryConfig() *DockerRegistryConfig {
	return &DockerRegistryConfig{
		DynamoComponentsRepositoryName: os.Getenv(consts.EnvDockerRegistryDynamoComponentsRepositoryName),
		Server:                         os.Getenv(consts.EnvDockerRegistryServer),
		SecretName:                     os.Getenv(consts.EnvDockerRegistrySecret),
		Secure:                         os.Getenv(consts.EnvDockerRegistrySecure) == "true",
	}
}

type ApiStoreConfig struct {
	Endpoint    string `yaml:"endpoint"`
	ClusterName string `yaml:"cluster_name"`
	ApiToken    string `yaml:"api_token"`
}

func GetApiStoreConfig(ctx context.Context) (conf *ApiStoreConfig, err error) {
	return &ApiStoreConfig{
		Endpoint:    os.Getenv(consts.EnvApiStoreEndpoint),
		ClusterName: os.Getenv(consts.EnvApiStoreClusterName),
		ApiToken:    os.Getenv(consts.EnvApiStoreApiToken),
	}, nil
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

type InternalImages struct {
	DynamoComponentsDownloader string
	Kaniko                     string
	Buildkit                   string
	BuildkitRootless           string
}

func GetInternalImages() (conf *InternalImages) {
	conf = &InternalImages{}
	conf.DynamoComponentsDownloader = getEnv(consts.EnvInternalImagesDynamoComponentsDownloader, consts.InternalImagesDynamoComponentsDownloaderDefault)
	conf.Kaniko = getEnv(consts.EnvInternalImagesKaniko, consts.InternalImagesKanikoDefault)
	conf.Buildkit = getEnv(consts.EnvInternalImagesBuildkit, consts.InternalImagesBuildkitDefault)
	conf.BuildkitRootless = getEnv(consts.EnvInternalImagesBuildkitRootless, consts.InternalImagesBuildkitRootlessDefault)
	return
}
