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

package utils

import (
	"net/url"
	"path"
)

func UrlJoin(baseUrl string, extra string, params ...map[string]string) string {
	u, err := url.Parse(baseUrl)
	if err != nil {
		return baseUrl
	}
	u.Path = path.Join(u.Path, extra)
	q := u.Query()
	for _, p := range params {
		for k, v := range p {
			q.Add(k, v)
		}
	}
	u.RawQuery = q.Encode()
	return u.String()
}

func UrlJoinWithQuery(baseUrl string, extra string, query url.Values) string {
	u, err := url.Parse(baseUrl)
	if err != nil {
		return baseUrl
	}
	u.Path = path.Join(u.Path, extra)
	u.RawQuery = query.Encode()
	return u.String()
}
