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

package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/dynamo/operator/api/dynamo/modelschemas"

type DeploymentTargetTypeSchema struct {
	Type modelschemas.DeploymentTargetType `json:"type" enum:"stable,canary"`
}

type DeploymentTargetSchema struct {
	ResourceSchema
	DeploymentTargetTypeSchema
	Creator     *UserSchema                               `json:"creator"`
	Bento       *BentoFullSchema                          `json:"bento"`
	CanaryRules *modelschemas.DeploymentTargetCanaryRules `json:"canary_rules"`
	Config      *modelschemas.DeploymentTargetConfig      `json:"config"`
}

type DeploymentTargetListSchema struct {
	BaseListSchema
	Items []*DeploymentTargetSchema `json:"items"`
}

type CreateDeploymentTargetSchema struct {
	DeploymentTargetTypeSchema
	BentoRepository string                                    `json:"bento_repository"`
	Bento           string                                    `json:"bento"`
	CanaryRules     *modelschemas.DeploymentTargetCanaryRules `json:"canary_rules"`
	Config          *modelschemas.DeploymentTargetConfig      `json:"config"`
}
