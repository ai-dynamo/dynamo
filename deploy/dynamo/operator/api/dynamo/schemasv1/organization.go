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

type OrganizationSchema struct {
	ResourceSchema
	Creator     *UserSchema `json:"creator"`
	Description string      `json:"description"`
}

type OrganizationFullSchema struct {
	OrganizationSchema
	Config **modelschemas.OrganizationConfigSchema `json:"config"`
}

type OrganizationListSchema struct {
	BaseListSchema
	Items []*OrganizationSchema `json:"items"`
}

type UpdateOrganizationSchema struct {
	Description *string                                 `json:"description"`
	Config      **modelschemas.OrganizationConfigSchema `json:"config"`
}

type CreateOrganizationSchema struct {
	Name        string                                 `json:"name"`
	Description string                                 `json:"description"`
	Config      *modelschemas.OrganizationConfigSchema `json:"config"`
}
