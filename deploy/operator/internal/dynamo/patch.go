/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	"github.com/imdario/mergo"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
)

func mergeObjectOverride(obj any, patch any) error {
	if err := mergo.Merge(obj, patch, mergo.WithOverride); err != nil {
		return fmt.Errorf("failed to apply override merge: %w", err)
	}

	return nil
}

func patchObjectStrategic[T any](obj T, patch T) error {
	objMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return fmt.Errorf("failed to convert object to unstructured: %w", err)
	}

	patchMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(patch)
	if err != nil {
		return fmt.Errorf("failed to convert patch to unstructured: %w", err)
	}

	var dataStruct T
	patchedObj, err := strategicpatch.StrategicMergeMapPatch(objMap, patchMap, dataStruct)
	if err != nil {
		return fmt.Errorf("failed to apply strategic merge patch: %w", err)
	}

	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(patchedObj, obj); err != nil {
		return fmt.Errorf("failed to deserialize patched object: %w", err)
	}

	return nil
}
