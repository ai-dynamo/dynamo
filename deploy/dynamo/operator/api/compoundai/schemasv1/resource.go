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

type IResourceSchema interface {
	GetType() modelschemas.ResourceType
	GetName() string
}

type ResourceSchema struct {
	BaseSchema
	Name         string                        `json:"name"`
	ResourceType modelschemas.ResourceType     `json:"resource_type" enum:"user,organization,cluster,bento_repository,bento,deployment,deployment_revision,model_repository,model,api_token"`
	Labels       modelschemas.LabelItemsSchema `json:"labels"`
}

func (r ResourceSchema) GetType() modelschemas.ResourceType {
	return r.ResourceType
}

func (r ResourceSchema) GetName() string {
	return r.Name
}

func (s *ResourceSchema) TypeName() string {
	return string(s.ResourceType)
}
