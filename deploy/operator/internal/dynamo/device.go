/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	corev1 "k8s.io/api/core/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// applyDeviceSpec copies a vendor-agnostic DeviceSpec into the pod spec.
//
// When a component sets Spec.Device, the operator hands the four Kubernetes
// primitives (Resources, Tolerations, NodeSelector, SchedulerName) through to
// the pod without injecting any NVIDIA defaults. Caller must check
// component.Device == nil before calling; this helper assumes the caller has
// decided to honor the user's device choice.
//
// Pod-level fields (Tolerations, NodeSelector, SchedulerName) are only set when
// non-empty so existing podTemplate overrides stay intact. Container-level
// resources are merged into the main container's Resources.Limits and
// Resources.Requests; existing entries from podTemplate win on conflict so a
// user can still override individual keys.
func applyDeviceSpec(podSpec *corev1.PodSpec, device *v1beta1.DeviceSpec) {
	if podSpec == nil || device == nil {
		return
	}

	if len(device.Tolerations) > 0 {
		podSpec.Tolerations = appendTolerations(podSpec.Tolerations, device.Tolerations)
	}

	if len(device.NodeSelector) > 0 && podSpec.NodeSelector == nil {
		podSpec.NodeSelector = map[string]string{}
	}
	for k, v := range device.NodeSelector {
		podSpec.NodeSelector[k] = v
	}

	if device.SchedulerName != "" {
		podSpec.SchedulerName = device.SchedulerName
	}

	if len(device.Resources) == 0 || len(podSpec.Containers) == 0 {
		return
	}

	container := &podSpec.Containers[0]
	if container.Resources.Limits == nil {
		container.Resources.Limits = corev1.ResourceList{}
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = corev1.ResourceList{}
	}
	for name, qty := range device.Resources {
		if _, ok := container.Resources.Limits[name]; !ok {
			container.Resources.Limits[name] = qty
		}
		if _, ok := container.Resources.Requests[name]; !ok {
			container.Resources.Requests[name] = qty
		}
	}
}

// appendTolerations adds tolerations from src to dst, preserving order and
// skipping duplicates (matched on key+operator+effect+value).
func appendTolerations(dst, src []corev1.Toleration) []corev1.Toleration {
	for _, t := range src {
		duplicate := false
		for _, existing := range dst {
			if existing.Key == t.Key &&
				existing.Operator == t.Operator &&
				existing.Effect == t.Effect &&
				existing.Value == t.Value {
				duplicate = true
				break
			}
		}
		if !duplicate {
			dst = append(dst, t)
		}
	}
	return dst
}

// HasAnyGPUResource reports whether the resource requirements request at least
// one of any kind of GPU-shaped extended resource. Used by the LWS worker
// generator to accept vendor-agnostic device requests (nvidia.com/gpu,
// amd.com/gpu, gpu.intel.com/xe, ...).
func HasAnyGPUResource(resources corev1.ResourceRequirements) bool {
	limits := resources.Limits
	requests := resources.Requests
	for name, qty := range limits {
		if isExtendedGPUResourceName(name) && !qty.IsZero() {
			return true
		}
	}
	for name, qty := range requests {
		if isExtendedGPUResourceName(name) && !qty.IsZero() {
			return true
		}
	}
	return false
}

// isExtendedGPUResourceName matches common GPU extended resource keys across
// vendors. We intentionally do not match the bare "cpu"/"memory" or the
// generic "gpu.intel.com/..." MIG-style keys are not enumerated; any extended
// resource containing "/gpu" or starting with "gpu." is treated as GPU-shaped.
func isExtendedGPUResourceName(name corev1.ResourceName) bool {
	s := string(name)
	if s == "" {
		return false
	}
	// Standard and HAMi keys.
	switch s {
	case "nvidia.com/gpu",
		"amd.com/gpu",
		"gpu.intel.com/xe",
		"gpu.intel.com/i915":
		return true
	}
	// Generic fallthrough: any "<vendor>/gpu" or "gpu.<vendor>/..." form.
	for i := 0; i < len(s)-3; i++ {
		if s[i] == '/' && s[i+1] == 'g' && s[i+2] == 'p' && s[i+3] == 'u' {
			return true
		}
	}
	if len(s) >= 4 && s[:4] == "gpu." {
		return true
	}
	return false
}