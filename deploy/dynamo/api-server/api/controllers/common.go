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

package controllers

import (
	"errors"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/gin-gonic/gin"
)

const OwnershipInfoKey = "_ownershipInfoKey"

func GetOwnershipInfo(ctx *gin.Context) (*schemas.OwnershipSchema, error) {
	ownership_ := ctx.Value(OwnershipInfoKey)
	if ownership_ == nil {
		return nil, consts.ErrNotFound
	}

	ownership, ok := ownership_.(*schemas.OwnershipSchema)
	if !ok {
		return nil, errors.New("current ownership is not an ownership struct")
	}

	return ownership, nil
}
