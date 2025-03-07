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

package converters

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
)

func ToBaseSchema(base models.BaseModel) schemas.BaseSchema {
	var deletedAt *time.Time
	deletedAt_ := base.GetDeletedAt()
	if deletedAt_.Valid {
		deletedAt = &deletedAt_.Time
	}
	return schemas.BaseSchema{
		Uid:       base.GetUid(),
		CreatedAt: base.GetCreatedAt(),
		UpdatedAt: base.GetUpdatedAt(),
		DeletedAt: deletedAt,
	}
}
