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

package models

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type IBaseModel interface {
	GetId() uint
	GetUid() string
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetDeletedAt() gorm.DeletedAt
}

type BaseModel struct {
	gorm.Model
	Uid uuid.UUID `json:"uid" gorm:"type:uuid;default:gen_random_uuid()"`
}

func (b *BaseModel) GetId() uint {
	return b.ID
}

func (b *BaseModel) GetUid() string {
	return b.Uid.String()
}

func (b *BaseModel) GetCreatedAt() time.Time {
	return b.CreatedAt
}

func (b *BaseModel) GetUpdatedAt() time.Time {
	return b.UpdatedAt
}

func (b *BaseModel) GetDeletedAt() gorm.DeletedAt {
	return b.DeletedAt
}
