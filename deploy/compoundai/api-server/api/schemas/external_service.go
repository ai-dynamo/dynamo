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

import (
	"encoding/json"
)

type ExternalService struct {
	DeploymentSelectorKey   string `json:"-"`
	DeploymentSelectorValue string `json:"-"`
}

// UnmarshalJSON handles snake_case to struct mapping
func (e *ExternalService) UnmarshalJSON(data []byte) error {
	var temp map[string]interface{}
	if err := json.Unmarshal(data, &temp); err != nil {
		return err
	}

	if val, ok := temp["deployment_selector_key"].(string); ok {
		e.DeploymentSelectorKey = val
	}
	if val, ok := temp["deployment_selector_value"].(string); ok {
		e.DeploymentSelectorValue = val
	}
	return nil
}

// MarshalJSON converts the struct to camelCase
func (e ExternalService) MarshalJSON() ([]byte, error) {
	temp := map[string]interface{}{
		"deploymentSelectorKey":   e.DeploymentSelectorKey,
		"deploymentSelectorValue": e.DeploymentSelectorValue,
	}
	return json.Marshal(temp)
}
