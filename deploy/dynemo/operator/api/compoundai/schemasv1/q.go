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
	"strings"

	"github.com/huandu/xstrings"
)

const ValueQMe = "@me"
const KeyQKeywords = "__keywords"
const KeyQIn = "in"

type Q string

func (q Q) ToMap() map[string]interface{} {
	res := map[string]interface{}{}
	for _, piece := range strings.Split(string(q), " ") {
		piece = strings.TrimSpace(piece)
		if piece == "" {
			continue
		}
		var k string
		var v string
		if !strings.Contains(piece, ":") {
			k = KeyQKeywords
			v = piece
		} else {
			k, _, v = xstrings.Partition(piece, ":")
			if v == "" {
				continue
			}
			if k == "is" {
				res[v] = true
				continue
			}
			if k == "not" {
				res[v] = false
				continue
			}
		}
		v_, ok := res[k]
		if !ok {
			v_ = make([]string, 0)
		}
		v_ = append(v_.([]string), v)
		res[k] = v_
	}
	return res
}
