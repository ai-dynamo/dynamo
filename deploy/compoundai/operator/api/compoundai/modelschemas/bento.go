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

type TransmissionStrategy string

const (
	TransmissionStrategyPresignedURL TransmissionStrategy = "presigned_url"
	TransmissionStrategyProxy        TransmissionStrategy = "proxy"
)

type BentoUploadStatus string

const (
	BentoUploadStatusPending   BentoUploadStatus = "pending"
	BentoUploadStatusUploading BentoUploadStatus = "uploading"
	BentoUploadStatusSuccess   BentoUploadStatus = "success"
	BentoUploadStatusFailed    BentoUploadStatus = "failed"
)

type ImageBuildStatus string

const (
	ImageBuildStatusPending  ImageBuildStatus = "pending"
	ImageBuildStatusBuilding ImageBuildStatus = "building"
	ImageBuildStatusSuccess  ImageBuildStatus = "success"
	ImageBuildStatusFailed   ImageBuildStatus = "failed"
)

type BentoApiSchema struct {
	Route  string `json:"route"`
	Doc    string `json:"doc"`
	Input  string `json:"input"`
	Output string `json:"output"`
}

type BentoRunnerResourceSchema struct {
	CPU             *float64           `json:"cpu"`
	NvidiaGPU       *float64           `json:"nvidia_gpu"`
	CustomResources map[string]float64 `json:"custom_resources"`
}

type BentoRunnerSchema struct {
	Name           string                     `json:"name"`
	RunnableType   string                     `json:"runnable_type"`
	Models         []string                   `json:"models"`
	ResourceConfig *BentoRunnerResourceSchema `json:"resource_config"`
}

type BentoManifestSchema struct {
	Service        string                    `json:"service"`
	BentomlVersion string                    `json:"bentoml_version"`
	Apis           map[string]BentoApiSchema `json:"apis"`
	Models         []string                  `json:"models"`
	Runners        []BentoRunnerSchema       `json:"runners"`
	SizeBytes      uint                      `json:"size_bytes"`
}

func (c *BentoManifestSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal(value.([]byte), c)
}

func (c *BentoManifestSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
