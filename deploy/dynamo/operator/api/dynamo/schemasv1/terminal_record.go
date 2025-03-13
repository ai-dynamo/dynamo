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

type TerminalRecordSchema struct {
	ResourceSchema
	Creator       *UserSchema         `json:"creator"`
	Organization  *OrganizationSchema `json:"organization"`
	Cluster       *ClusterSchema      `json:"cluster"`
	Deployment    *DeploymentSchema   `json:"deployment"`
	Resource      *ResourceSchema     `json:"resource"`
	PodName       string              `json:"pod_name"`
	ContainerName string              `json:"container_name"`
}

type TerminalRecordListSchema struct {
	BaseListSchema
	Items []*TerminalRecordSchema `json:"items"`
}
