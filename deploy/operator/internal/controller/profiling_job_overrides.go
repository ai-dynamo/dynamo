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

package controller

import (
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// protectedLabelKeys are controller-managed label keys that user overrides
// must not overwrite. The controller relies on these for ownership tracking
// and watch predicates.
var protectedLabelKeys = map[string]struct{}{
	nvidiacomv1beta1.LabelApp:           {},
	nvidiacomv1beta1.LabelDGDR:          {},
	nvidiacomv1beta1.LabelDGDRName:      {},
	nvidiacomv1beta1.LabelDGDRNamespace: {},
	nvidiacomv1beta1.LabelManagedBy:     {},
}

// applyProfilingJobOverrides merges user-provided overrides from
// spec.overrides.profilingJob into the controller-generated Job.
// Uses a deterministic allowlist: only explicitly handled fields are merged.
func applyProfilingJobOverrides(job *batchv1.Job, overrides *batchv1.JobSpec) {
	if overrides == nil {
		return
	}
	applyJobSpecOverrides(&job.Spec, overrides)
	applyPodTemplateOverrides(&job.Spec.Template, &overrides.Template)
}

// applyJobSpecOverrides merges JobSpec-level scalar fields.
func applyJobSpecOverrides(spec *batchv1.JobSpec, overrides *batchv1.JobSpec) {
	if overrides.BackoffLimit != nil {
		spec.BackoffLimit = overrides.BackoffLimit
	}
	if overrides.ActiveDeadlineSeconds != nil {
		spec.ActiveDeadlineSeconds = overrides.ActiveDeadlineSeconds
	}
	if overrides.TTLSecondsAfterFinished != nil {
		spec.TTLSecondsAfterFinished = overrides.TTLSecondsAfterFinished
	}
	if overrides.Completions != nil {
		spec.Completions = overrides.Completions
	}
	if overrides.Parallelism != nil {
		spec.Parallelism = overrides.Parallelism
	}
	if overrides.Suspend != nil {
		spec.Suspend = overrides.Suspend
	}
}

// applyPodTemplateOverrides merges PodTemplateSpec metadata and PodSpec fields.
func applyPodTemplateOverrides(tmpl *corev1.PodTemplateSpec, overrides *corev1.PodTemplateSpec) {
	mergeLabels(tmpl, overrides.Labels)
	mergeAnnotations(tmpl, overrides.Annotations)
	applyPodSpecOverrides(&tmpl.Spec, &overrides.Spec)
}

// mergeLabels adds user labels to the template, skipping protected controller keys.
func mergeLabels(tmpl *corev1.PodTemplateSpec, userLabels map[string]string) {
	if len(userLabels) == 0 {
		return
	}
	if tmpl.Labels == nil {
		tmpl.Labels = make(map[string]string, len(userLabels))
	}
	for k, v := range userLabels {
		if _, protected := protectedLabelKeys[k]; protected {
			continue
		}
		tmpl.Labels[k] = v
	}
}

// mergeAnnotations adds user annotations to the template.
func mergeAnnotations(tmpl *corev1.PodTemplateSpec, userAnnotations map[string]string) {
	if len(userAnnotations) == 0 {
		return
	}
	if tmpl.Annotations == nil {
		tmpl.Annotations = make(map[string]string, len(userAnnotations))
	}
	for k, v := range userAnnotations {
		tmpl.Annotations[k] = v
	}
}

// mergeImagePullSecrets combines base and override secrets, deduplicating by name.
// Override secrets that already exist in base are skipped (base wins on conflict).
func mergeImagePullSecrets(base, overrides []corev1.LocalObjectReference) []corev1.LocalObjectReference {
	if len(overrides) == 0 {
		return base
	}
	seen := make(map[string]bool, len(base))
	result := make([]corev1.LocalObjectReference, len(base))
	copy(result, base)
	for _, s := range base {
		seen[s.Name] = true
	}
	for _, s := range overrides {
		if !seen[s.Name] {
			result = append(result, s)
			seen[s.Name] = true
		}
	}
	return result
}

// applyPodSpecOverrides merges PodSpec-level fields and the first container.
func applyPodSpecOverrides(spec *corev1.PodSpec, overrides *corev1.PodSpec) {
	if len(overrides.Tolerations) > 0 {
		spec.Tolerations = overrides.Tolerations
	}
	if len(overrides.NodeSelector) > 0 {
		spec.NodeSelector = overrides.NodeSelector
	}
	if overrides.Affinity != nil {
		spec.Affinity = overrides.Affinity
	}
	if overrides.PriorityClassName != "" {
		spec.PriorityClassName = overrides.PriorityClassName
	}
	if len(overrides.ImagePullSecrets) > 0 {
		spec.ImagePullSecrets = mergeImagePullSecrets(spec.ImagePullSecrets, overrides.ImagePullSecrets)
	}
	if overrides.ServiceAccountName != "" {
		spec.ServiceAccountName = overrides.ServiceAccountName
	}
	if overrides.RuntimeClassName != nil {
		spec.RuntimeClassName = overrides.RuntimeClassName
	}
	if overrides.DNSPolicy != "" {
		spec.DNSPolicy = overrides.DNSPolicy
	}
	if overrides.DNSConfig != nil {
		spec.DNSConfig = overrides.DNSConfig
	}

	spec.Volumes = mergeNamedSlice(spec.Volumes, overrides.Volumes, func(v corev1.Volume) string { return v.Name })
	spec.InitContainers = mergeNamedSlice(spec.InitContainers, overrides.InitContainers, func(c corev1.Container) string { return c.Name })

	if len(overrides.Containers) > 0 && len(spec.Containers) > 0 {
		applyContainerOverrides(&spec.Containers[0], &overrides.Containers[0])
	}
}

// applyContainerOverrides merges fields from the user's first container override
// into the controller-generated profiler container.
func applyContainerOverrides(container *corev1.Container, overrides *corev1.Container) {
	if overrides.Image != "" {
		container.Image = overrides.Image
	}
	if len(overrides.Resources.Requests) > 0 || len(overrides.Resources.Limits) > 0 || len(overrides.Resources.Claims) > 0 {
		container.Resources = overrides.Resources
	}
	if overrides.SecurityContext != nil {
		container.SecurityContext = overrides.SecurityContext
	}

	container.Env = mergeNamedSlice(container.Env, overrides.Env, func(e corev1.EnvVar) string { return e.Name })
	container.VolumeMounts = mergeNamedSlice(container.VolumeMounts, overrides.VolumeMounts, func(vm corev1.VolumeMount) string { return vm.Name })

	if len(overrides.EnvFrom) > 0 {
		container.EnvFrom = append(container.EnvFrom, overrides.EnvFrom...)
	}
}

// mergeNamedSlice merges two slices of named items. Items from overrides with
// the same name as a base item replace the base entry; new names are appended.
// Preserves ordering of base items.
func mergeNamedSlice[T any](base, overrides []T, nameFunc func(T) string) []T {
	if len(overrides) == 0 {
		return base
	}
	seen := make(map[string]int, len(base))
	result := make([]T, len(base))
	copy(result, base)
	for i, item := range result {
		seen[nameFunc(item)] = i
	}
	for _, item := range overrides {
		if idx, exists := seen[nameFunc(item)]; exists {
			result[idx] = item
		} else {
			result = append(result, item)
			seen[nameFunc(item)] = len(result) - 1
		}
	}
	return result
}
