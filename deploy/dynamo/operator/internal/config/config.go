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

func GetYataiImageBuilderNamespace(ctx context.Context) (namespace string, err error) {
	return os.Getenv(consts.EnvYataiImageBuilderNamespace), nil
}

type DockerRegistryConfig struct {
	BentoRepositoryName string `yaml:"bento_repository_name"`
	ModelRepositoryName string `yaml:"model_repository_name"`
	Server              string `yaml:"server"`
	InClusterServer     string `yaml:"in_cluster_server"`
	Username            string `yaml:"username"`
	Password            string `yaml:"password"`
	Secure              bool   `yaml:"secure"`
}

func GetDockerRegistryConfig() (conf *DockerRegistryConfig, err error) {
	return &DockerRegistryConfig{
		BentoRepositoryName: os.Getenv(consts.EnvDockerRegistryBentoRepositoryName),
		ModelRepositoryName: os.Getenv(consts.EnvDockerRegistryModelRepositoryName),
		Server:              os.Getenv(consts.EnvDockerRegistryServer),
		InClusterServer:     os.Getenv(consts.EnvDockerRegistryInClusterServer),
		Username:            os.Getenv(consts.EnvDockerRegistryUsername),
		Password:            os.Getenv(consts.EnvDockerRegistryPassword),
		Secure:              os.Getenv(consts.EnvDockerRegistrySecure) == "true",
	}, nil
}

type YataiConfig struct {
	Endpoint    string `yaml:"endpoint"`
	ClusterName string `yaml:"cluster_name"`
	ApiToken    string `yaml:"api_token"`
}

func GetYataiConfig(ctx context.Context) (conf *YataiConfig, err error) {
	return &YataiConfig{
		Endpoint:    os.Getenv(consts.EnvYataiEndpoint),
		ClusterName: os.Getenv(consts.EnvYataiClusterName),
		ApiToken:    os.Getenv(consts.EnvYataiApiToken),
	}, nil
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

type InternalImages struct {
	BentoDownloader    string
	Kaniko             string
	MetricsTransformer string
	Buildkit           string
	BuildkitRootless   string
}

func GetInternalImages() (conf *InternalImages) {
	conf = &InternalImages{}
	conf.BentoDownloader = getEnv(consts.EnvInternalImagesBentoDownloader, consts.InternalImagesBentoDownloaderDefault)
	conf.Kaniko = getEnv(consts.EnvInternalImagesKaniko, consts.InternalImagesKanikoDefault)
	conf.MetricsTransformer = getEnv(consts.EnvInternalImagesMetricsTransformer, consts.InternalImagesMetricsTransformerDefault)
	conf.Buildkit = getEnv(consts.EnvInternalImagesBuildkit, consts.InternalImagesBuildkitDefault)
	conf.BuildkitRootless = getEnv(consts.EnvInternalImagesBuildkitRootless, consts.InternalImagesBuildkitRootlessDefault)
	return
}
