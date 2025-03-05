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

type ModelUploadStatus string

const (
	ModelUploadStatusPending   ModelUploadStatus = "pending"
	ModelUploadStatusUploading ModelUploadStatus = "uploading"
	ModelUploadStatusSuccess   ModelUploadStatus = "success"
	ModelUploadStatusFailed    ModelUploadStatus = "failed"
)

type ModelManifestSchema struct {
	BentomlVersion string                 `json:"bentoml_version"`
	ApiVersion     string                 `json:"api_version"`
	Module         string                 `json:"module"`
	Metadata       map[string]interface{} `json:"metadata"`
	Context        map[string]interface{} `json:"context"`
	Options        map[string]interface{} `json:"options"`
	SizeBytes      uint                   `json:"size_bytes"`
}

func (c *ModelManifestSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal(value.([]byte), c)
}

func (c *ModelManifestSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
