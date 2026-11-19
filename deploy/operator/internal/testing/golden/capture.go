// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package golden

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// CaptureOwned reads the operator's ownership graph through its immediate
// workloads, including kinds absent from the golden. Kubernetes/Grove-generated
// descendants (for example Service EndpointSlices and workload Pods) are outside
// this lowering boundary.
func CaptureOwned(ctx context.Context, config *rest.Config, c client.Client, root client.Object) ([]unstructured.Unstructured, error) {
	d, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}
	resources, err := d.ServerPreferredNamespacedResources()
	if err != nil {
		return nil, err
	}
	var candidates []unstructured.Unstructured
	for _, group := range resources {
		gv, err := schema.ParseGroupVersion(group.GroupVersion)
		if err != nil {
			return nil, err
		}
		for _, resource := range group.APIResources {
			if strings.Contains(resource.Name, "/") || !slices.Contains(resource.Verbs, "list") {
				continue
			}
			list := &unstructured.UnstructuredList{}
			list.SetGroupVersionKind(gv.WithKind(resource.Kind + "List"))
			if err := c.List(ctx, list, client.InNamespace(root.GetNamespace())); err != nil {
				return nil, fmt.Errorf("capture %s: %w", resource.Name, err)
			}
			for i := range list.Items {
				list.Items[i].SetGroupVersionKind(gv.WithKind(resource.Kind))
			}
			candidates = append(candidates, list.Items...)
		}
	}
	selected := map[types.UID]bool{root.GetUID(): true}
	terminal := map[types.UID]bool{}
	for _, object := range candidates {
		switch object.GetKind() {
		case "Deployment", "LeaderWorkerSet", "DisaggregatedSet", "PodCliqueSet", "Service", "Job":
			terminal[object.GetUID()] = true
		}
	}
	for changed := true; changed; {
		changed = false
		for i := range candidates {
			if selected[candidates[i].GetUID()] {
				continue
			}
			for _, owner := range candidates[i].GetOwnerReferences() {
				if selected[owner.UID] && !terminal[owner.UID] {
					selected[candidates[i].GetUID()] = true
					changed = true
					break
				}
			}
		}
	}
	var result []unstructured.Unstructured
	for i := range candidates {
		if selected[candidates[i].GetUID()] {
			result = append(result, candidates[i])
		}
	}
	slices.SortFunc(result, func(a, b unstructured.Unstructured) int {
		return strings.Compare(a.GetAPIVersion()+"/"+a.GetKind()+"/"+a.GetName(), b.GetAPIVersion()+"/"+b.GetKind()+"/"+b.GetName())
	})
	if len(result) == 0 {
		return nil, fmt.Errorf("captured no resources for %s", root.GetName())
	}
	return result, nil
}
