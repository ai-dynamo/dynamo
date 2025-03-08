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

package mocks

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
)

var mockedUid = "nvid1a11-1234-5678-9abc-def012345678"

func DefaultUser() *schemas.UserSchema {
	return &schemas.UserSchema{
		ResourceSchema: schemas.ResourceSchema{
			BaseSchema: schemas.BaseSchema{
				Uid:       mockedUid,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: nil,
			},
			Name: "nvidia-user",
		},
		FirstName: "Dynamo",
		LastName:  "AI",
		Email:     "dynamo@nvidia.com",
	}
}

func DefaultOrg() *schemas.OrganizationSchema {
	return &schemas.OrganizationSchema{
		ResourceSchema: schemas.ResourceSchema{
			BaseSchema: schemas.BaseSchema{
				Uid:       mockedUid,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: nil,
			},
			Name:         "nvidia-org",
			ResourceType: schemas.ResourceTypeOrganization,
			Labels:       []schemas.LabelItemSchema{},
		},
		Description: "nvidia-org-desc",
	}
}

func DefaultOrgMember() *schemas.OrganizationMemberSchema {
	return &schemas.OrganizationMemberSchema{
		BaseSchema: schemas.BaseSchema{
			Uid:       mockedUid,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			DeletedAt: nil,
		},
		Role:         schemas.MemberRoleAdmin,
		Creator:      DefaultUser(),
		User:         *DefaultUser(),
		Organization: *DefaultOrg(),
	}
}
