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

package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type AWSS3Schema struct {
	BentosBucketName string `json:"bentos_bucket_name"`
	ModelsBucketName string `json:"models_bucket_name"`
	Region           string `json:"region"`
}

type AWSECRSchema struct {
	AccountId            string `json:"account_id"`
	BentosRepositoryName string `json:"bentos_repository_name"`
	ModelsRepositoryName string `json:"models_repository_name"`
	Password             string `json:"password"`
	Region               string `json:"region"`
}

type OrganizationConfigAWSSchema struct {
	AccessKeyId     string        `json:"access_key_id"`
	SecretAccessKey string        `json:"secret_access_key"`
	ECR             *AWSECRSchema `json:"ecr"`
	S3              *AWSS3Schema  `json:"s3"`
}

type OrganizationDockerRegistrySchema struct {
	BentosRepositoryURI string `json:"bentos_repository_uri"`
	ModelsRepositoryURI string `json:"models_repository_uri"`
	Server              string `json:"server"`
	Username            string `json:"username"`
	Password            string `json:"password"`
	Secure              bool   `json:"secure"`
}

type OrganizationS3Schema struct {
	Endpoint         string `json:"endpoint"`
	AccessKey        string `json:"access_key"`
	SecretKey        string `json:"secret_key"`
	Secure           bool   `json:"secure"`
	Region           string `json:"region"`
	BentosBucketName string `json:"bentos_bucket_name"`
	ModelsBucketName string `json:"models_bucket_name"`
}

type OrganizationConfigSchema struct {
	MajorClusterUid      string                            `json:"major_cluster_uid"`
	AWS                  *OrganizationConfigAWSSchema      `json:"aws,omitempty"`
	DockerRegistry       *OrganizationDockerRegistrySchema `json:"docker_registry,omitempty"`
	S3                   *OrganizationS3Schema             `json:"s3,omitempty"`
	TransmissionStrategy *TransmissionStrategy             `json:"transmission_strategy,omitempty"`
}

func (c *OrganizationConfigSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), c)
}

func (c *OrganizationConfigSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
