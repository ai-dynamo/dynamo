// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Every edited component must reach its immediate workload, including both multinode roles.
func upgradeWorkerEditError(objects []unstructured.Unstructured, workers []string) error {
	edited := map[string]bool{}
	for _, object := range objects {
		component := object.GetLabels()["nvidia.com/dynamo-component"]
		switch object.GetKind() {
		case "Deployment":
			edited[component] = edited[component] || upgradeTemplateHasEdit(object.Object, "spec", "template")
		case "LeaderWorkerSet":
			edited[component] = edited[component] || upgradeLWSHasEdit(object.Object)
		case "DisaggregatedSet":
			roles, _, err := unstructured.NestedSlice(object.Object, "spec", "roles")
			if err != nil {
				return err
			}
			for i, value := range roles {
				role, ok := value.(map[string]any)
				if !ok {
					return fmt.Errorf("DisaggregatedSet %s role %d is not an object", object.GetName(), i)
				}
				name, ok := role["name"].(string)
				if !ok {
					return fmt.Errorf("DisaggregatedSet %s role %d has no string name", object.GetName(), i)
				}
				edited[name] = edited[name] || upgradeLWSHasEdit(role)
			}
		}
	}
	for _, worker := range workers {
		if !edited[worker] {
			return fmt.Errorf("worker edit has not reached all templates for %s", worker)
		}
	}
	return nil
}

func upgradeLWSHasEdit(object map[string]any) bool {
	return upgradeTemplateHasEdit(object, "spec", "leaderWorkerTemplate", "leaderTemplate") &&
		upgradeTemplateHasEdit(object, "spec", "leaderWorkerTemplate", "workerTemplate")
}

func upgradeTemplateHasEdit(object map[string]any, path ...string) bool {
	containers, _, err := unstructured.NestedSlice(object, append(path, "spec", "containers")...)
	if err != nil {
		return false
	}
	for _, value := range containers {
		container := value.(map[string]any)
		if container["name"] != "main" {
			continue
		}
		variables, _, err := unstructured.NestedSlice(container, "env")
		if err != nil {
			return false
		}
		for _, value := range variables {
			variable := value.(map[string]any)
			if variable["name"] == "WORKER_REVISION" && variable["value"] == "2" {
				return true
			}
		}
	}
	return false
}

func TestUpgradeMultinodeWorkerEdit(t *testing.T) {
	for _, kind := range []string{"LeaderWorkerSet", "DisaggregatedSet"} {
		for _, missing := range []string{"none", "leader", "worker", "decode"} {
			t.Run(kind+"/"+missing, func(t *testing.T) {
				t.Log("Require the edit in both templates of every worker component")
				var objects []unstructured.Unstructured
				var roles []any
				for _, name := range []string{"prefill", "decode"} {
					role := map[string]any{"name": name, "spec": map[string]any{}}
					for _, template := range []string{"leader", "worker"} {
						pod := map[string]any{"spec": map[string]any{"containers": []any{map[string]any{"name": "main"}}}}
						if missing != template && missing != name {
							require.NoError(t, unstructured.SetNestedSlice(pod, []any{map[string]any{
								"name": "main", "env": []any{map[string]any{"name": "WORKER_REVISION", "value": "2"}},
							}}, "spec", "containers"))
						}
						require.NoError(t, unstructured.SetNestedMap(role, pod, "spec", "leaderWorkerTemplate", template+"Template"))
					}
					roles = append(roles, role)
					objects = append(objects, unstructured.Unstructured{Object: map[string]any{
						"kind": kind, "metadata": map[string]any{"labels": map[string]any{"nvidia.com/dynamo-component": name}}, "spec": role["spec"],
					}})
				}
				if kind == "DisaggregatedSet" {
					objects = []unstructured.Unstructured{{Object: map[string]any{"kind": kind, "spec": map[string]any{"roles": roles}}}}
				}
				err := upgradeWorkerEditError(objects, []string{"prefill", "decode"})
				if missing == "none" {
					require.NoError(t, err)
				} else {
					require.Error(t, err)
				}
			})
		}
	}
}

func TestUpgradeWorkerEditMalformedRoles(t *testing.T) {
	for _, test := range []struct {
		name string
		role any
		want string
	}{
		{name: "not-an-object", role: "invalid", want: "role 0 is not an object"},
		{name: "missing-name", role: map[string]any{}, want: "role 0 has no string name"},
		{name: "invalid-name", role: map[string]any{"name": int64(1)}, want: "role 0 has no string name"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Malformed DS roles report context instead of panicking during polling")
			object := unstructured.Unstructured{Object: map[string]any{
				"kind": "DisaggregatedSet", "metadata": map[string]any{"name": "workers"},
				"spec": map[string]any{"roles": []any{test.role}},
			}}
			require.ErrorContains(t, upgradeWorkerEditError([]unstructured.Unstructured{object}, []string{"prefill"}), test.want)
		})
	}
}
