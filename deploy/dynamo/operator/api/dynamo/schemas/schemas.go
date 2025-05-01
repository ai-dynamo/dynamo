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
	"errors"
	"time"
)

type DynamoComponent struct {
	PresignedDownloadUrl string                `json:"presigned_download_url"`
	TransmissionStrategy *TransmissionStrategy `json:"transmission_strategy"`
}

type TransmissionStrategy string

const (
	TransmissionStrategyPresignedURL TransmissionStrategy = "presigned_url"
	TransmissionStrategyProxy        TransmissionStrategy = "proxy"
)

type Duration time.Duration

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	switch value := v.(type) {
	case float64:
		*d = Duration(time.Duration(value))
	case string:
		tmp, err := time.ParseDuration(value)
		if err != nil {
			return err
		}
		*d = Duration(tmp)
	default:
		return errors.New("invalid duration")
	}
	return nil
}

type DeploymentStrategy string

const (
	DeploymentStrategyRollingUpdate               DeploymentStrategy = "RollingUpdate"
	DeploymentStrategyRecreate                    DeploymentStrategy = "Recreate"
	DeploymentStrategyRampedSlowRollout           DeploymentStrategy = "RampedSlowRollout"
	DeploymentStrategyBestEffortControlledRollout DeploymentStrategy = "BestEffortControlledRollout"
)

type DockerRegistrySchema struct {
	DynamoRepositoryURI string `json:"dynamoRepositoryURI"`
	Server              string `json:"server"`
	Username            string `json:"username"`
	Password            string `json:"password"`
	SecretName          string `json:"secretName"`
	Secure              bool   `json:"secure"`
}
