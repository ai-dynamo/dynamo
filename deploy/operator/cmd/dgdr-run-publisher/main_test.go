/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestWriteTerminationMessage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "termination.log")
	message := strings.Repeat("failure-🙂", maxTerminationMessage)

	if err := writeTerminationMessage(path, errors.New(message)); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) > maxTerminationMessage || !utf8.Valid(got) || !strings.HasPrefix(string(got), "failure-🙂") {
		t.Fatalf("termination message is not a bounded UTF-8 prefix: %d bytes, %q", len(got), got[:min(len(got), 32)])
	}
}
