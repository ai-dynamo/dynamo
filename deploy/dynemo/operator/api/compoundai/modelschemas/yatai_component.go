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

type YataiComponentName string

const (
	YataiComponentNameDeployment   YataiComponentName = "deployment"
	YataiComponentNameImageBuilder YataiComponentName = "image-builder"
	YataiComponentNameServerless   YataiComponentName = "serverless"
	YataiComponentNameFunction     YataiComponentName = "function"
	YataiComponentNameJob          YataiComponentName = "job"
)

type YataiComponentManifestSchema struct {
	SelectorLabels   map[string]string `json:"selector_labels,omitempty"`
	LatestCRDVersion string            `json:"latest_crd_version,omitempty"`
}

func (c *YataiComponentManifestSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal(value.([]byte), c)
}

func (c *YataiComponentManifestSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
