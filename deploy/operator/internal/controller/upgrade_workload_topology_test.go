// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Compare every multinode component's rendered LWS size against its DGD intent, not its golden.
func upgradeLWSSizeError(dgd *api.DynamoGraphDeployment, objects []unstructured.Unstructured) error {
	for _, component := range dgd.Spec.Components {
		if !component.IsMultinode() {
			continue
		}

		// Match the logical component even when workload names carry a worker hash.
		matches := 0
		for _, object := range objects {
			if object.GetKind() != "LeaderWorkerSet" || object.GetLabels()[consts.KubeLabelDynamoComponent] != component.ComponentName {
				continue
			}
			matches++
			size, found, err := unstructured.NestedInt64(object.Object, "spec", "leaderWorkerTemplate", "size")
			if err != nil {
				return fmt.Errorf("LeaderWorkerSet %s for component %s has invalid size: %w", object.GetName(), component.ComponentName, err)
			}
			if !found {
				return fmt.Errorf("LeaderWorkerSet %s for component %s has no leaderWorkerTemplate.size", object.GetName(), component.ComponentName)
			}
			if size != int64(component.Multinode.NodeCount) {
				return fmt.Errorf("LeaderWorkerSet %s for component %s has size %d, want %d", object.GetName(), component.ComponentName, size, component.Multinode.NodeCount)
			}
		}
		if matches != 1 {
			return fmt.Errorf("component %s has %d LeaderWorkerSets, want 1", component.ComponentName, matches)
		}
	}
	return nil
}

func TestUpgradeLWSSize(t *testing.T) {
	for _, test := range []struct {
		name      string
		sizes     []any
		duplicate bool
		want      string
	}{
		{name: "correct", sizes: []any{int64(2), int64(3)}},
		{name: "lowered-to-single-node", sizes: []any{int64(1), int64(3)}, want: "has size 1, want 2"},
		{name: "per-component-size", sizes: []any{int64(2), int64(2)}, want: "has size 2, want 3"},
		{name: "missing-size", sizes: []any{int64(2), nil}, want: "has no leaderWorkerTemplate.size"},
		{name: "invalid-size", sizes: []any{int64(2), "3"}, want: "has invalid size"},
		{name: "missing-component", sizes: []any{int64(2)}, want: "component decode has 0 LeaderWorkerSets"},
		{name: "duplicate-component", sizes: []any{int64(2), int64(3)}, duplicate: true, want: "component prefill has 2 LeaderWorkerSets"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Require one correctly sized LWS for each multinode component independently of goldens")
			dgd := &api.DynamoGraphDeployment{Spec: api.DynamoGraphDeploymentSpec{
				Components: []api.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "frontend", ComponentType: api.ComponentTypeFrontend},
					{ComponentName: "prefill", ComponentType: api.ComponentTypePrefill, Multinode: &api.MultinodeSpec{NodeCount: 2}},
					{ComponentName: "decode", ComponentType: api.ComponentTypeDecode, Multinode: &api.MultinodeSpec{NodeCount: 3}},
				},
			}}
			var objects []unstructured.Unstructured
			for i, size := range test.sizes {
				name := dgd.Spec.Components[i+1].ComponentName
				object := unstructured.Unstructured{Object: map[string]any{
					"kind": "LeaderWorkerSet", "metadata": map[string]any{
						"name": "workers-" + name, "labels": map[string]any{consts.KubeLabelDynamoComponent: name},
					},
				}}
				if size != nil {
					require.NoError(t, unstructured.SetNestedField(object.Object, size, "spec", "leaderWorkerTemplate", "size"))
				}
				objects = append(objects, object)
			}
			if test.duplicate {
				objects = append(objects, *objects[0].DeepCopy())
			}

			t.Log("A wrong rendered topology must fail even when that output is accepted as a golden")
			err := upgradeLWSSizeError(dgd, objects)
			if test.want == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, test.want)
			}
		})
	}
}

// Compare every DS role's rendered size and templates against its DGD intent, not its golden.
func upgradeDSRolesError(dgd *api.DynamoGraphDeployment, object unstructured.Unstructured) error {
	roles, found, err := unstructured.NestedSlice(object.Object, "spec", "roles")
	if err != nil {
		return fmt.Errorf("DisaggregatedSet %s has invalid roles: %w", object.GetName(), err)
	}
	if !found {
		return fmt.Errorf("DisaggregatedSet %s has no roles", object.GetName())
	}

	// Derive the expected role inventory and each size from the authored components.
	want := map[string]int64{}
	for _, component := range dgd.Spec.Components {
		if component.IsMultinode() {
			want[component.ComponentName] = int64(component.Multinode.NodeCount)
		}
	}
	seen := map[string]bool{}
	for i, value := range roles {
		role, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("DisaggregatedSet %s role %d is not an object", object.GetName(), i)
		}
		name, ok := role["name"].(string)
		if !ok {
			return fmt.Errorf("DisaggregatedSet %s role %d has no string name", object.GetName(), i)
		}
		nodes, ok := want[name]
		if !ok {
			return fmt.Errorf("DisaggregatedSet %s has unexpected role %s", object.GetName(), name)
		}
		if seen[name] {
			return fmt.Errorf("DisaggregatedSet %s has duplicate role %s", object.GetName(), name)
		}
		seen[name] = true

		// Check size and both immediate pod templates before accepting this role.
		size, found, err := unstructured.NestedInt64(role, "spec", "leaderWorkerTemplate", "size")
		if err != nil {
			return fmt.Errorf("DisaggregatedSet %s role %s has invalid size: %w", object.GetName(), name, err)
		}
		if !found {
			return fmt.Errorf("DisaggregatedSet %s role %s has no leaderWorkerTemplate.size", object.GetName(), name)
		}
		if size != nodes {
			return fmt.Errorf("DisaggregatedSet %s role %s has size %d, want %d", object.GetName(), name, size, nodes)
		}
		for _, template := range []string{"leaderTemplate", "workerTemplate"} {
			containers, _, err := unstructured.NestedSlice(role, "spec", "leaderWorkerTemplate", template, "spec", "containers")
			if err != nil || len(containers) == 0 {
				return fmt.Errorf("DisaggregatedSet %s role %s has no %s containers", object.GetName(), name, template)
			}
		}
	}

	// Every authored multinode component must have a distinct rendered role.
	for _, component := range dgd.Spec.Components {
		if component.IsMultinode() && !seen[component.ComponentName] {
			return fmt.Errorf("DisaggregatedSet %s has no role %s", object.GetName(), component.ComponentName)
		}
	}
	return nil
}

// upgradeDSRole constructs an independent role fixture with both pod templates.
func upgradeDSRole(name string, size any) map[string]any {
	template := map[string]any{}
	for _, role := range []string{"leaderTemplate", "workerTemplate"} {
		template[role] = map[string]any{"spec": map[string]any{
			"containers": []any{map[string]any{"name": "main"}},
		}}
	}
	if size != nil {
		template["size"] = size
	}
	return map[string]any{"name": name, "spec": map[string]any{"leaderWorkerTemplate": template}}
}

func TestUpgradeDSRoles(t *testing.T) {
	t.Log("Build separate fixtures for missing and empty leader and worker templates")
	noLeader, noWorker := upgradeDSRole("decode", int64(3)), upgradeDSRole("decode", int64(3))
	unstructured.RemoveNestedField(noLeader, "spec", "leaderWorkerTemplate", "leaderTemplate")
	unstructured.RemoveNestedField(noWorker, "spec", "leaderWorkerTemplate", "workerTemplate")
	emptyLeader, emptyWorker := upgradeDSRole("decode", int64(3)), upgradeDSRole("decode", int64(3))
	require.NoError(t, unstructured.SetNestedSlice(emptyLeader, []any{}, "spec", "leaderWorkerTemplate", "leaderTemplate", "spec", "containers"))
	require.NoError(t, unstructured.SetNestedSlice(emptyWorker, []any{}, "spec", "leaderWorkerTemplate", "workerTemplate", "spec", "containers"))

	for _, test := range []struct {
		name  string
		roles any
		want  string
	}{
		{name: "correct", roles: []any{upgradeDSRole("prefill", int64(2)), upgradeDSRole("decode", int64(3))}},
		{name: "reordered", roles: []any{upgradeDSRole("decode", int64(3)), upgradeDSRole("prefill", int64(2))}},
		{name: "lowered-to-single-node", roles: []any{upgradeDSRole("prefill", int64(1)), upgradeDSRole("decode", int64(3))}, want: "role prefill has size 1, want 2"},
		{name: "per-component-size", roles: []any{upgradeDSRole("prefill", int64(2)), upgradeDSRole("decode", int64(2))}, want: "role decode has size 2, want 3"},
		{name: "missing-size", roles: []any{upgradeDSRole("prefill", int64(2)), upgradeDSRole("decode", nil)}, want: "role decode has no leaderWorkerTemplate.size"},
		{name: "invalid-size", roles: []any{upgradeDSRole("prefill", int64(2)), upgradeDSRole("decode", "3")}, want: "role decode has invalid size"},
		{name: "missing-role", roles: []any{upgradeDSRole("prefill", int64(2))}, want: "has no role decode"},
		{name: "duplicate-role", roles: []any{upgradeDSRole("prefill", int64(2)), upgradeDSRole("prefill", int64(2))}, want: "duplicate role prefill"},
		{name: "unexpected-role", roles: []any{upgradeDSRole("prefill", int64(2)), upgradeDSRole("frontend", int64(3))}, want: "unexpected role frontend"},
		{name: "missing-leader-template", roles: []any{upgradeDSRole("prefill", int64(2)), noLeader}, want: "role decode has no leaderTemplate containers"},
		{name: "missing-worker-template", roles: []any{upgradeDSRole("prefill", int64(2)), noWorker}, want: "role decode has no workerTemplate containers"},
		{name: "empty-leader-containers", roles: []any{upgradeDSRole("prefill", int64(2)), emptyLeader}, want: "role decode has no leaderTemplate containers"},
		{name: "empty-worker-containers", roles: []any{upgradeDSRole("prefill", int64(2)), emptyWorker}, want: "role decode has no workerTemplate containers"},
		{name: "not-an-object", roles: []any{"invalid"}, want: "role 0 is not an object"},
		{name: "missing-name", roles: []any{map[string]any{}}, want: "role 0 has no string name"},
		{name: "invalid-name", roles: []any{map[string]any{"name": int64(3)}}, want: "role 0 has no string name"},
		{name: "missing-roles", want: "has no roles"},
		{name: "invalid-roles", roles: "invalid", want: "has invalid roles"},
		{name: "empty-roles", roles: []any{}, want: "has no role prefill"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Require one correctly sized DS role for each multinode component independently of goldens")
			dgd := &api.DynamoGraphDeployment{Spec: api.DynamoGraphDeploymentSpec{
				Components: []api.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "frontend", ComponentType: api.ComponentTypeFrontend},
					{ComponentName: "prefill", ComponentType: api.ComponentTypePrefill, Multinode: &api.MultinodeSpec{NodeCount: 2}},
					{ComponentName: "decode", ComponentType: api.ComponentTypeDecode, Multinode: &api.MultinodeSpec{NodeCount: 3}},
				},
			}}
			object := unstructured.Unstructured{Object: map[string]any{
				"kind": "DisaggregatedSet", "metadata": map[string]any{"name": "workers"}, "spec": map[string]any{},
			}}
			if test.roles != nil {
				require.NoError(t, unstructured.SetNestedField(object.Object, test.roles, "spec", "roles"))
			}

			t.Log("Reject an invalid role graph even when its output could be accepted as a golden")
			err := upgradeDSRolesError(dgd, object)
			if test.want == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, test.want)
			}
		})
	}
}
