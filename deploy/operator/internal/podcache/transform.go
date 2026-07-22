/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package podcache defines the Pod representation retained by the shared
// controller-runtime cache.
package podcache

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Configure installs Transform on the single typed Pod cache while preserving
// any namespace or selector configuration already attached to that cache.
func Configure(options *cache.Options) error {
	if options == nil {
		return nil
	}
	if options.ByObject == nil {
		options.ByObject = make(map[client.Object]cache.ByObject)
	}

	var podKey client.Object
	for obj := range options.ByObject {
		if _, ok := obj.(*corev1.Pod); !ok {
			continue
		}
		if podKey != nil {
			return fmt.Errorf("multiple Pod cache configurations are not supported")
		}
		podKey = obj
	}

	if podKey == nil {
		options.ByObject[&corev1.Pod{}] = cache.ByObject{Transform: Transform}
		return nil
	}

	podOptions := options.ByObject[podKey]
	podOptions.Transform = Transform
	options.ByObject[podKey] = podOptions
	return nil
}

// Transform projects Pods to the fields used by operator controllers before
// the objects enter the shared informer cache. It is intentionally idempotent.
func Transform(obj any) (any, error) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return obj, nil
	}
	return Project(pod), nil
}

// Project reduces pod in place to the shared cache contract.
//
// Pods read from the cached client are partial objects. Callers must use narrow
// Patch/Delete operations, or use the manager APIReader before an Update or
// whenever a field outside this projection is required.
func Project(pod *corev1.Pod) *corev1.Pod {
	if pod == nil {
		return nil
	}

	// All labels, annotations, owner references, identity, finalizers, and
	// deletion state are retained. ManagedFields is the large metadata field no
	// controller consumes.
	pod.ManagedFields = nil
	pod.Spec = projectSpec(pod.Spec)
	pod.Status = projectStatus(pod.Status)
	return pod
}

func projectSpec(in corev1.PodSpec) corev1.PodSpec {
	return corev1.PodSpec{
		// Topology and snapshot controllers require the assigned node.
		NodeName: in.NodeName,
		// Model endpoint classification requires the main container's command
		// and arguments.
		Containers: projectContainers(in.Containers),
		// Topology label discovery inspects DownwardAPI label field paths.
		Volumes: projectTopologyVolumes(in.Volumes),
	}
}

func projectContainers(in []corev1.Container) []corev1.Container {
	if len(in) == 0 {
		return nil
	}
	out := make([]corev1.Container, len(in))
	for i := range in {
		out[i] = corev1.Container{
			Name:    in[i].Name,
			Command: in[i].Command,
			Args:    in[i].Args,
		}
	}
	return out
}

func projectTopologyVolumes(in []corev1.Volume) []corev1.Volume {
	out := make([]corev1.Volume, 0, len(in))
	for i := range in {
		if in[i].DownwardAPI == nil {
			continue
		}

		var items []corev1.DownwardAPIVolumeFile
		for j := range in[i].DownwardAPI.Items {
			item := &in[i].DownwardAPI.Items[j]
			if item.FieldRef == nil {
				continue
			}
			items = append(items, corev1.DownwardAPIVolumeFile{
				Path: item.Path,
				FieldRef: &corev1.ObjectFieldSelector{
					APIVersion: item.FieldRef.APIVersion,
					FieldPath:  item.FieldRef.FieldPath,
				},
				Mode: item.Mode,
			})
		}
		if len(items) == 0 {
			continue
		}

		out = append(out, corev1.Volume{
			Name: in[i].Name,
			VolumeSource: corev1.VolumeSource{
				DownwardAPI: &corev1.DownwardAPIVolumeSource{
					Items:       items,
					DefaultMode: in[i].DownwardAPI.DefaultMode,
				},
			},
		})
	}
	return out
}

func projectStatus(in corev1.PodStatus) corev1.PodStatus {
	return corev1.PodStatus{
		// Failover, DGDR diagnostics, and Recreate drain barriers require the
		// lifecycle phase.
		Phase: in.Phase,
		// Model endpoint classification requires the Kubernetes Ready condition.
		Conditions: projectReadyConditions(in.Conditions),
		// DGDR diagnostics inspect waiting and terminated container failures.
		ContainerStatuses:     projectContainerStatuses(in.ContainerStatuses),
		InitContainerStatuses: projectContainerStatuses(in.InitContainerStatuses),
	}
}

func projectReadyConditions(in []corev1.PodCondition) []corev1.PodCondition {
	out := make([]corev1.PodCondition, 0, len(in))
	for i := range in {
		if in[i].Type != corev1.PodReady {
			continue
		}
		out = append(out, corev1.PodCondition{
			Type:   in[i].Type,
			Status: in[i].Status,
		})
	}
	return out
}

func projectContainerStatuses(in []corev1.ContainerStatus) []corev1.ContainerStatus {
	if len(in) == 0 {
		return nil
	}
	out := make([]corev1.ContainerStatus, len(in))
	for i := range in {
		out[i] = corev1.ContainerStatus{
			Name:  in[i].Name,
			State: projectContainerState(in[i].State),
		}
	}
	return out
}

func projectContainerState(in corev1.ContainerState) corev1.ContainerState {
	out := corev1.ContainerState{}
	if in.Waiting != nil {
		out.Waiting = &corev1.ContainerStateWaiting{
			Reason:  in.Waiting.Reason,
			Message: in.Waiting.Message,
		}
	}
	if in.Terminated != nil {
		out.Terminated = &corev1.ContainerStateTerminated{
			ExitCode: in.Terminated.ExitCode,
			Reason:   in.Terminated.Reason,
			Message:  in.Terminated.Message,
		}
	}
	return out
}
