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

type BentoRepositorySchema struct {
	ResourceSchema
	Creator      *UserSchema         `json:"creator"`
	Organization *OrganizationSchema `json:"organization"`
	LatestBento  *BentoSchema        `json:"latest_bento"`
	NBentos      uint                `json:"n_bentos"`
	NDeployments uint                `json:"n_deployments"`
	LatestBentos []*BentoSchema      `json:"latest_bentos"`
	Description  string              `json:"description"`
}

type BentoRepositoryListSchema struct {
	BaseListSchema
	Items []*BentoRepositorySchema `json:"items"`
}

type BentoRepositoryWithLatestDeploymentsSchema struct {
	BentoRepositorySchema
	LatestDeployments []*DeploymentSchema `json:"latest_deployments"`
}

type BentoRepositoryWithLatestDeploymentsListSchema struct {
	BaseListSchema
	Items []*BentoRepositoryWithLatestDeploymentsSchema `json:"items"`
}

type CreateBentoRepositorySchema struct {
	Name        string                        `json:"name"`
	Description string                        `json:"description"`
	Labels      modelschemas.LabelItemsSchema `json:"labels"`
}

type UpdateBentoRepositorySchema struct {
	Description *string                        `json:"description"`
	Labels      *modelschemas.LabelItemsSchema `json:"labels,omitempty"`
}
