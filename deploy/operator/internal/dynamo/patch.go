/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"slices"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/imdario/mergo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
)

func mergeObjectOverride(obj any, patch any) error {
	if err := mergo.Merge(obj, patch, mergo.WithOverride); err != nil {
		return fmt.Errorf("failed to apply override merge: %w", err)
	}
	return nil
}

// Strategic merge field-merges matching list items, which breaks union-style
// objects when a user needs to replace an existing entry.
func removeNamed[T any](items, overrides []T, nameOf func(T) string) []T {
	if len(items) == 0 || len(overrides) == 0 {
		return items
	}

	// Index every non-empty override name.
	overrideNames := make(map[string]struct{}, len(overrides))
	for _, override := range overrides {
		if name := nameOf(override); name != "" {
			overrideNames[name] = struct{}{}
		}
	}
	if len(overrideNames) == 0 {
		return items
	}

	// Remove every overridden base item while preserving survivor order.
	return slices.DeleteFunc(items, func(item T) bool {
		_, exists := overrideNames[nameOf(item)]
		return exists
	})
}

// mergeContainerByName merges override into base; both must be non-nil.
// mergeStrategy must be resolved by ResolveExtraPodSpecMergeStrategy.
// It preserves override by copying it once, then transferring owned fields to base.
func mergeContainerByName(base *corev1.Container, override *corev1.Container, mergeStrategy v1alpha1.ExtraPodSpecMergeStrategy) error {
	user := override.DeepCopy()
	user.Name = commonconsts.MainContainerName

	if mergeStrategy == v1alpha1.ExtraPodSpecMergeStrategyOverride {
		baseEnv := base.Env
		if err := mergeObjectOverride(base, *user); err != nil {
			return err
		}
		base.Env = MergeEnvs(baseEnv, user.Env)
		if user.Ports != nil {
			base.Ports = user.Ports
		}
	} else {
		clearPorts := user.Ports != nil && len(user.Ports) == 0
		base.Env = removeNamed(base.Env, user.Env, func(env corev1.EnvVar) string { return env.Name })
		if !clearPorts {
			base.Ports = removeNamed(base.Ports, user.Ports, func(port corev1.ContainerPort) string { return port.Name })
		}
		if err := patchObjectStrategic(base, user); err != nil {
			return err
		}
		if clearPorts {
			base.Ports = user.Ports
		}
	}
	if user.LivenessProbe != nil {
		base.LivenessProbe = user.LivenessProbe
	}
	if user.ReadinessProbe != nil {
		base.ReadinessProbe = user.ReadinessProbe
	}
	if user.StartupProbe != nil {
		base.StartupProbe = user.StartupProbe
	}
	base.Name = commonconsts.MainContainerName
	return nil
}

// mergePodSpecOverride applies the caller-owned override to podSpec, which must be non-nil.
// mergeStrategy must be resolved by ResolveExtraPodSpecMergeStrategy.
func mergePodSpecOverride(podSpec *corev1.PodSpec, podSpecOverride corev1.PodSpec, mergeStrategy v1alpha1.ExtraPodSpecMergeStrategy) error {
	if mergeStrategy == v1alpha1.ExtraPodSpecMergeStrategyOverride {
		return mergeObjectOverride(podSpec, &podSpecOverride)
	}

	// Replace named volume unions before strategic field merging.
	podSpec.Volumes = removeNamed(podSpec.Volumes, podSpecOverride.Volumes, func(volume corev1.Volume) string { return volume.Name })
	return patchObjectStrategic(podSpec, &podSpecOverride)
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

	err = runtime.DefaultUnstructuredConverter.FromUnstructured(patchedObj, obj)
	if err != nil {
		return fmt.Errorf("failed to deserialize patched object: %w", err)
	}

	return nil
}
