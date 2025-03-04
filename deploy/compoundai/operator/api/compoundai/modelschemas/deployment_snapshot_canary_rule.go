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

type DeploymentTargetCanaryRuleType string

const (
	DeploymentTargetCanaryRuleTypeWeight DeploymentTargetCanaryRuleType = "weight"
	DeploymentTargetCanaryRuleTypeHeader DeploymentTargetCanaryRuleType = "header"
	DeploymentTargetCanaryRuleTypeCookie DeploymentTargetCanaryRuleType = "cookie"
)

type DeploymentTargetCanaryRule struct {
	Type DeploymentTargetCanaryRuleType `json:"type" enum:"weight,header,cookie"`

	Weight      *uint   `json:"weight"`
	Header      *string `json:"header"`
	Cookie      *string `json:"cookie"`
	HeaderValue *string `json:"header_value"`
}

type DeploymentTargetCanaryRules []*DeploymentTargetCanaryRule

func (c *DeploymentTargetCanaryRules) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), c)
}

func (c *DeploymentTargetCanaryRules) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
