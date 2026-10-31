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
package gpu

import (
	"reflect"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestCollectOtherModels(t *testing.T) {
	type node struct {
		sku   nvidiacomv1beta1.GPUSKUType
		model string
	}
	tests := []struct {
		name          string
		filter        nvidiacomv1beta1.GPUSKUType
		selectedSKU   nvidiacomv1beta1.GPUSKUType
		selectedModel string
		nodes         []node
		want          []string
	}{
		{"a cluster of one SKU", "", "h200_sxm", "H200", []node{{"h200_sxm", "H200"}, {"h200_sxm", "H200"}}, nil},
		{"a second SKU is reported once, sorted", "", "h200_sxm", "H200",
			[]node{{"h200_sxm", "H200"}, {"h100_sxm", "H100"}, {"b200_sxm", "B200"}, {"h100_sxm", "H100"}},
			[]string{"B200", "H100"}},
		{"a model the catalog does not know still counts", "", "h200_sxm", "H200",
			[]node{{"h200_sxm", "H200"}, {"", "Mystery GPU"}}, []string{"Mystery GPU"}},
		{"distinct models unknown to the catalog remain distinct", "", "", "Mystery Alpha",
			[]node{{"", "Mystery Alpha"}, {"", "Mystery Beta"}}, []string{"Mystery Beta"}},
		{"normalized spellings of an unknown selected model share its identity", "", "", "Mystery GPU",
			[]node{{"", "Mystery GPU"}, {"", "mystery-gpu"}}, nil},
		{"normalized spellings of another unknown model are reported once", "", "", "Mystery Alpha",
			[]node{{"", "Mystery Alpha"}, {"", "mystery-beta"}, {"", "Mystery Beta"}}, []string{"Mystery Beta"}},
		{"known models with the same SKU are reported once", "", "h200_sxm", "H200",
			[]node{{"h200_sxm", "H200"}, {"h100_sxm", "NVIDIA H100"}, {"h100_sxm", "H100 SXM"}}, []string{"H100 SXM"}},
		{"a filtered discovery never reports", "h200_sxm", "h200_sxm", "H200",
			[]node{{"h200_sxm", "H200"}, {"h100_sxm", "H100"}}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := collectOtherModels(tt.filter, tt.selectedSKU, tt.selectedModel, len(tt.nodes),
				func(i int) (nvidiacomv1beta1.GPUSKUType, string) { return tt.nodes[i].sku, tt.nodes[i].model })
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}
