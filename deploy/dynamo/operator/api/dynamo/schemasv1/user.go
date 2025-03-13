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

import "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/modelschemas"

type UserSchema struct {
	ResourceSchema
	FirstName    string `json:"first_name"`
	LastName     string `json:"last_name"`
	Email        string `json:"email"`
	AvatarUrl    string `json:"avatar_url"`
	IsSuperAdmin bool   `json:"is_super_admin"`
}

type UserListSchema struct {
	BaseListSchema
	Items []*UserSchema `json:"items"`
}

type RegisterUserSchema struct {
	Name      string `json:"name" validate:"required"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Email     string `json:"email" validate:"required"`
	Password  string `json:"password" validate:"required"`
}

type LoginUserSchema struct {
	NameOrEmail string `json:"name_or_email" validate:"required"`
	Password    string `json:"password" validate:"required"`
}

type UpdateUserSchema struct {
	FirstName string `json:"first_name" validate:"required"`
	LastName  string `json:"last_name" validate:"required"`
}

type ResetPasswordSchema struct {
	CurrentPassword string `json:"current_password"`
	NewPassword     string `json:"new_password"`
}

type CreateUserSchema struct {
	Name     string                  `json:"name" validate:"required"`
	Email    string                  `json:"email" validate:"required"`
	Password string                  `json:"password" validate:"required"`
	Role     modelschemas.MemberRole `json:"role" enum:"guest,developer,admin"`
}
