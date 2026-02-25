/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package defaulting

import (
	"testing"
)

func TestDGDRDefaulter_defaultImageFor(t *testing.T) {
	tests := []struct {
		name            string
		operatorVersion string
		expectedImage   string
	}{
		{
			name:            "known version produces default image",
			operatorVersion: "1.0.0",
			expectedImage:   "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0",
		},
		{
			name:            "pre-release version is valid",
			operatorVersion: "1.0.0-rc1",
			expectedImage:   "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0-rc1",
		},
		{
			name:            "unknown operator version cannot be defaulted",
			operatorVersion: "unknown",
			expectedImage:   "",
		},
		{
			name:            "empty operator version cannot be defaulted",
			operatorVersion: "",
			expectedImage:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDGDRDefaulter(tt.operatorVersion)
			got := d.defaultImageFor()
			if got != tt.expectedImage {
				t.Errorf("defaultImageFor() = %q, want %q", got, tt.expectedImage)
			}
		})
	}
}
