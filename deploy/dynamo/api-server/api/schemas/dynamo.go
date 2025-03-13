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

package schemas

type DynamoNimApiSchema struct {
	Route  string `json:"route"`
	Doc    string `json:"doc"`
	Input  string `json:"input"`
	Output string `json:"output"`
}

type DynamoNimManifestSchema struct {
	Service         string                        `json:"service"`
	DynamoAiVersion string                        `json:"bentoml_version"`
	Apis            map[string]DynamoNimApiSchema `json:"apis"`
	SizeBytes       uint                          `json:"size_bytes"`
}

type TransmissionStrategy string

const (
	TransmissionStrategyPresignedURL TransmissionStrategy = "presigned_url"
	TransmissionStrategyProxy        TransmissionStrategy = "proxy"
)

type DynamoNimVersionUploadStatus string

const (
	DynamoNimVersionUploadStatusPending   DynamoNimVersionUploadStatus = "pending"
	DynamoNimVersionUploadStatusUploading DynamoNimVersionUploadStatus = "uploading"
	DynamoNimVersionUploadStatusSuccess   DynamoNimVersionUploadStatus = "success"
	DynamoNimVersionUploadStatusFailed    DynamoNimVersionUploadStatus = "failed"
)

type ImageBuildStatus string

const (
	ImageBuildStatusPending  ImageBuildStatus = "pending"
	ImageBuildStatusBuilding ImageBuildStatus = "building"
	ImageBuildStatusSuccess  ImageBuildStatus = "success"
	ImageBuildStatusFailed   ImageBuildStatus = "failed"
)
