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

type IResourceSchema interface {
	GetType() ResourceType
	GetName() string
}

type ResourceSchema struct {
	BaseSchema
	Name         string            `json:"name"`
	Labels       []LabelItemSchema `json:"labels"`
	ResourceType ResourceType      `json:"resource_type" enum:"user,organization,cluster,compound_nim,compound_nim_version,deployment,deployment_revision,model_repository,model,api_token"`
}

func (r ResourceSchema) GetType() ResourceType {
	return r.ResourceType
}

func (r ResourceSchema) GetName() string {
	return r.Name
}

func (s *ResourceSchema) TypeName() string {
	return string(s.ResourceType)
}

type ResourceItem struct {
	CPU    string            `json:"cpu,omitempty"`
	Memory string            `json:"memory,omitempty"`
	GPU    string            `json:"gpu,omitempty"`
	Custom map[string]string `json:"custom,omitempty"`
}

type Resources struct {
	Requests *ResourceItem `json:"requests,omitempty"`
	Limits   *ResourceItem `json:"limits,omitempty"`
}
