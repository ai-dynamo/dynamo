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

package services

import (
	"fmt"
	"strings"

	"gorm.io/gorm"
)

type BaseListOption struct {
	Start             *uint
	Count             *uint
	Search            *string
	Keywords          *[]string
	KeywordFieldNames *[]string
}

func (opt BaseListOption) BindQueryWithLimit(query *gorm.DB) *gorm.DB {
	if opt.Count != nil {
		query = query.Limit(int(*opt.Count))
	}
	if opt.Start != nil {
		query = query.Offset(int(*opt.Start))
	}
	return query
}

func (opt BaseListOption) BindQueryWithKeywords(query *gorm.DB, tableName string) *gorm.DB {
	tableName = query.Statement.Quote(tableName)
	keywordFieldNames := []string{"name"}
	if opt.KeywordFieldNames != nil {
		keywordFieldNames = *opt.KeywordFieldNames
	}
	if opt.Search != nil && *opt.Search != "" {
		sqlPieces := make([]string, 0, len(keywordFieldNames))
		args := make([]interface{}, 0, len(keywordFieldNames))
		for _, keywordFieldName := range keywordFieldNames {
			keywordFieldName = query.Statement.Quote(keywordFieldName)
			sqlPieces = append(sqlPieces, fmt.Sprintf("%s.%s LIKE ?", tableName, keywordFieldName))
			args = append(args, fmt.Sprintf("%%%s%%", *opt.Search))
		}
		query = query.Where(fmt.Sprintf("(%s)", strings.Join(sqlPieces, " OR ")), args...)
	}
	if opt.Keywords != nil {
		sqlPieces := make([]string, 0, len(keywordFieldNames))
		args := make([]interface{}, 0, len(keywordFieldNames))
		for _, keywordFieldName := range keywordFieldNames {
			keywordFieldName = query.Statement.Quote(keywordFieldName)
			sqlPieces_ := make([]string, 0, len(*opt.Keywords))
			for _, keyword := range *opt.Keywords {
				sqlPieces_ = append(sqlPieces_, fmt.Sprintf("%s.%s LIKE ?", tableName, keywordFieldName))
				args = append(args, fmt.Sprintf("%%%s%%", keyword))
			}
			sqlPieces = append(sqlPieces, fmt.Sprintf("(%s)", strings.Join(sqlPieces_, " AND ")))
		}
		query = query.Where(fmt.Sprintf("(%s)", strings.Join(sqlPieces, " OR ")), args...)
	}
	return query
}
