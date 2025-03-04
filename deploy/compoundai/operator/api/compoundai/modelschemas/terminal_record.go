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

package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type RecordType string

const (
	RecordTypeInput  RecordType = "i"
	RecordTypeOutput RecordType = "o"
)

type TerminalRecordEnv struct {
	TERM  string `json:"term"`
	SHELL string `json:"shell"`
}

type TerminalRecordMeta struct {
	Version   uint               `json:"version"`
	Width     uint16             `json:"width"`
	Height    uint16             `json:"height"`
	Timestamp int64              `json:"timestamp"`
	Env       *TerminalRecordEnv `json:"env"`
}

func (m *TerminalRecordMeta) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), m)
}

func (m *TerminalRecordMeta) Value() (driver.Value, error) {
	if m == nil {
		return nil, nil
	}
	return json.Marshal(m)
}
