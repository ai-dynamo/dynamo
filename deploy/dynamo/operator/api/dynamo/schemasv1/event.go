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

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/operator/api/dynamo/modelschemas"
)

type EventSchema struct {
	BaseSchema
	Resource        interface{}              `json:"resource,omitempty"`
	Name            string                   `json:"name,omitempty"`
	Status          modelschemas.EventStatus `json:"status,omitempty"`
	OperationName   string                   `json:"operation_name,omitempty"`
	ApiTokenName    string                   `json:"api_token_name,omitempty"`
	Creator         *UserSchema              `json:"creator,omitempty"`
	CreatedAt       time.Time                `json:"created_at,omitempty"`
	ResourceDeleted bool                     `json:"resource_deleted,omitempty"`
}

type EventListSchema struct {
	BaseListSchema
	Items []*EventSchema `json:"items"`
}
