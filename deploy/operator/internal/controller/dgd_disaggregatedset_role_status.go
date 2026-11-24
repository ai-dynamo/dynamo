// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/disaggregatedset"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// applyDisaggregatedSetRoleStatuses projects target templates from the persisted
// DisaggregatedSet, not from its possibly older child LeaderWorkerSets.
func applyDisaggregatedSetRoleStatuses(ds *unstructured.Unstructured, selection disaggregatedset.Selection, statuses map[string]v1beta1.ComponentReplicaStatus) error {
	var spec struct {
		Roles []struct {
			Name string                                `json:"name"`
			Spec leaderworkersetv1.LeaderWorkerSetSpec `json:"spec"`
		} `json:"roles"`
	}
	raw, _, err := unstructured.NestedMap(ds.Object, "spec")
	if err != nil {
		return fmt.Errorf("read DisaggregatedSet target spec: %w", err)
	}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(raw, &spec); err != nil {
		return fmt.Errorf("decode DisaggregatedSet target roles: %w", err)
	}

	// Bind each selected component to its lowered LWS role without re-rendering.
	for component, roleName := range selection.ComponentToRole {
		status := statuses[component]
		for _, role := range spec.Roles {
			if role.Name != roleName {
				continue
			}
			templates := role.Spec.LeaderWorkerTemplate
			if templates.LeaderTemplate == nil {
				break
			}
			status.Roles = []v1beta1.ComponentRoleStatus{
				dynamo.ResolveComponentRoleStatus(string(dynamo.RoleLeader), &templates.LeaderTemplate.Spec),
				dynamo.ResolveComponentRoleStatus(string(dynamo.RoleWorker), &templates.WorkerTemplate.Spec),
			}
			break
		}
		statuses[component] = status
	}
	return nil
}
