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
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

type DeploymentTargetType string

const (
	DeploymentTargetTypeStable DeploymentTargetType = "stable"
	DeploymentTargetTypeCanary DeploymentTargetType = "canary"
)

type DeploymentStrategy string

const (
	DeploymentStrategyRollingUpdate               DeploymentStrategy = "RollingUpdate"
	DeploymentStrategyRecreate                    DeploymentStrategy = "Recreate"
	DeploymentStrategyRampedSlowRollout           DeploymentStrategy = "RampedSlowRollout"
	DeploymentStrategyBestEffortControlledRollout DeploymentStrategy = "BestEffortControlledRollout"
)

var DeploymentTargetTypeAddrs = map[DeploymentTargetType]string{
	DeploymentTargetTypeStable: "stb",
	DeploymentTargetTypeCanary: "cnr",
}

type DeploymentTargetHPAConf struct {
	CPU         *int32  `json:"cpu,omitempty"`
	GPU         *int32  `json:"gpu,omitempty"`
	Memory      *string `json:"memory,omitempty"`
	QPS         *int64  `json:"qps,omitempty"`
	MinReplicas *int32  `json:"min_replicas,omitempty"`
	MaxReplicas *int32  `json:"max_replicas,omitempty"`
}

type DeploymentOverrides struct {
	ColdStartTimeout *int32 `json:"cold_start_timeout"`
}

type DeploymentTargetConfig struct {
	KubeResourceUid                        string                     `json:"kubeResourceUid"`
	KubeResourceVersion                    string                     `json:"kubeResourceVersion"`
	Resources                              *Resources                 `json:"resources"`
	HPAConf                                *DeploymentTargetHPAConf   `json:"hpa_conf,omitempty"`
	EnableIngress                          *bool                      `json:"enable_ingress,omitempty"`
	EnableStealingTrafficDebugMode         *bool                      `json:"enable_stealing_traffic_debug_mode,omitempty"`
	EnableDebugMode                        *bool                      `json:"enable_debug_mode,omitempty"`
	EnableDebugPodReceiveProductionTraffic *bool                      `json:"enable_debug_pod_receive_production_traffic,omitempty"`
	DeploymentStrategy                     *DeploymentStrategy        `json:"deployment_strategy,omitempty"`
	ExternalServices                       map[string]ExternalService `json:"external_services,omitempty"`
	DeploymentOverrides                    *DeploymentOverrides       `json:"DeploymentOverrides,omitempty"`
}

type CreateDeploymentTargetSchema struct {
	CompoundNim string                  `json:"bento_repository"`
	Version     string                  `json:"bento"`
	Config      *DeploymentTargetConfig `json:"config"`
}

func (c *DeploymentTargetConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}

	var data []byte
	switch v := value.(type) {
	case string:
		data = []byte(v)
	case []byte:
		data = v
	default:
		return fmt.Errorf("unsupported type: %T", value)
	}

	return json.Unmarshal(data, c)
}

func (c *DeploymentTargetConfig) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}

type DeploymentTargetTypeSchema struct {
	Type string `json:"type" enum:"stable,canary"`
}

type DeploymentTargetSchema struct {
	ResourceSchema
	DeploymentTargetTypeSchema
	Creator            *UserSchema                   `json:"creator"`
	CompoundNimVersion *CompoundNimVersionFullSchema `json:"bento"`
	Config             *DeploymentTargetConfig       `json:"config"`
}

type DeploymentTargetListSchema struct {
	BaseListSchema
	Items []*DeploymentTargetSchema `json:"items"`
}
