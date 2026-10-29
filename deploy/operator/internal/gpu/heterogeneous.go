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
	"sort"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// collectOtherModels returns, sorted and without repeats, the models of nodes whose GPU identity
// differs from the selected one. It uses the same identity rule as capacity aggregation: known
// GPUs are grouped by SKU and unknown GPUs by normalized model name. A filtered discovery asked
// for one SKU and returns nothing because nodes of other SKUs were never candidates.
func collectOtherModels(
	filterSKU, selectedSKU nvidiacomv1beta1.GPUSKUType,
	selectedModel string,
	nodes int,
	at func(i int) (nvidiacomv1beta1.GPUSKUType, string),
) []string {
	if filterSKU != "" {
		return nil
	}
	seen := map[string]string{}
	for i := 0; i < nodes; i++ {
		sku, model := at(i)
		if matchesSelectedGPUType(sku, selectedSKU, model, selectedModel) {
			continue
		}
		identity := "sku:" + string(sku)
		if sku == "" {
			identity = "model:" + normalize(model)
		}
		if existing, exists := seen[identity]; !exists || model < existing {
			seen[identity] = model
		}
	}
	if len(seen) == 0 {
		return nil
	}
	models := make([]string, 0, len(seen))
	for _, model := range seen {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}
