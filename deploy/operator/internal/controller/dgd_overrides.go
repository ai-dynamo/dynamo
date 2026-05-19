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

package controller

import (
	"bytes"
	"encoding/json"
	"fmt"

	dgdv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	sigsyaml "sigs.k8s.io/yaml"
)

const dgdOverrideKind = "DynamoGraphDeployment"

func applyDGDOverrides(generated *dgdv1alpha1.DynamoGraphDeployment, override *runtime.RawExtension) error {
	if generated == nil || override == nil || len(override.Raw) == 0 {
		return nil
	}

	overrideJSON, err := sigsyaml.YAMLToJSON(override.Raw)
	if err != nil {
		return fmt.Errorf("convert DGD override to JSON: %w", err)
	}

	var overrideDGD dgdv1alpha1.DynamoGraphDeployment
	decoder := json.NewDecoder(bytes.NewReader(overrideJSON))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&overrideDGD); err != nil {
		return fmt.Errorf("unmarshal DGD override: %w", err)
	}
	if overrideDGD.APIVersion != "" && overrideDGD.APIVersion != dgdv1alpha1.GroupVersion.String() {
		return fmt.Errorf("DGD override apiVersion must be %q, got %q", dgdv1alpha1.GroupVersion.String(), overrideDGD.APIVersion)
	}
	if overrideDGD.Kind != "" && overrideDGD.Kind != dgdOverrideKind {
		return fmt.Errorf("DGD override kind must be %q, got %q", dgdOverrideKind, overrideDGD.Kind)
	}

	generatedJSON, err := json.Marshal(generated)
	if err != nil {
		return fmt.Errorf("marshal generated DGD: %w", err)
	}

	var generatedPatch map[string]any
	if err := json.Unmarshal(generatedJSON, &generatedPatch); err != nil {
		return fmt.Errorf("unmarshal generated DGD: %w", err)
	}

	var userPatch map[string]any
	if err := json.Unmarshal(overrideJSON, &userPatch); err != nil {
		return fmt.Errorf("unmarshal DGD override patch: %w", err)
	}

	mergedPatch := mergeJSONPatch(generatedPatch, userPatch)
	mergedJSON, err := json.Marshal(mergedPatch)
	if err != nil {
		return fmt.Errorf("marshal merged DGD: %w", err)
	}

	if err := json.Unmarshal(mergedJSON, generated); err != nil {
		return fmt.Errorf("unmarshal merged DGD: %w", err)
	}
	return nil
}

func mergeJSONPatch(base, patch map[string]any) map[string]any {
	for key, patchValue := range patch {
		if patchValue == nil {
			delete(base, key)
			continue
		}

		patchMap, patchIsMap := patchValue.(map[string]any)
		baseMap, baseIsMap := base[key].(map[string]any)
		if patchIsMap && baseIsMap {
			base[key] = mergeJSONPatch(baseMap, patchMap)
			continue
		}
		if patchIsMap {
			base[key] = mergeJSONPatch(map[string]any{}, patchMap)
			continue
		}

		base[key] = patchValue
	}
	return base
}
