// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"strings"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Every selected worker needs the edit in its immediate workload, including both LWS roles.
func upgradeComponentWorkerEditError(dgd *api.DynamoGraphDeployment, objects []unstructured.Unstructured, desiredHash string) error {
	for _, component := range dgd.Spec.Components {
		if component.ComponentType == api.ComponentTypeFrontend {
			continue
		}
		kind := "Deployment"
		paths := [][]string{{"spec", "template", "spec", "containers"}}
		if component.Multinode != nil && component.Multinode.NodeCount > 1 {
			kind = "LeaderWorkerSet"
			paths = [][]string{
				{"spec", "leaderWorkerTemplate", "leaderTemplate", "spec", "containers"},
				{"spec", "leaderWorkerTemplate", "workerTemplate", "spec", "containers"},
			}
		}
		matched := 0
		for _, object := range objects {
			if object.GetKind() != kind || object.GetLabels()["nvidia.com/dynamo-component"] != component.ComponentName || object.GetLabels()["nvidia.com/dynamo-worker-hash"] != desiredHash {
				continue
			}
			matched++
			for _, path := range paths {
				containers, found, err := unstructured.NestedSlice(object.Object, path...)
				if err != nil || !found {
					return fmt.Errorf("%s/%s missing containers at %v: %v", kind, object.GetName(), path, err)
				}
				edited := false
				for containerIndex, container := range containers {
					c, ok := container.(map[string]any)
					if !ok {
						return fmt.Errorf("%s/%s has non-object container at %s[%d]: %T", kind, object.GetName(), strings.Join(path, "."), containerIndex, container)
					}
					if c["name"] != "main" {
						continue
					}
					variables, _, err := unstructured.NestedSlice(c, "env")
					if err != nil {
						return fmt.Errorf("%s/%s invalid env at %s[%d].env: %w", kind, object.GetName(), strings.Join(path, "."), containerIndex, err)
					}
					for variableIndex, variable := range variables {
						v, ok := variable.(map[string]any)
						if !ok {
							return fmt.Errorf("%s/%s has non-object env entry at %s[%d].env[%d]: %T", kind, object.GetName(), strings.Join(path, "."), containerIndex, variableIndex, variable)
						}
						edited = edited || (v["name"] == "WORKER_REVISION" && v["value"] == "2")
					}
				}
				if !edited {
					return fmt.Errorf("%s/%s has not applied worker edit at %v", kind, object.GetName(), path)
				}
			}
		}
		if matched != 1 {
			return fmt.Errorf("component %s expected one edited %s, found %d", component.ComponentName, kind, matched)
		}
	}
	return nil
}

func TestUpgradeComponentWorkerEdit(t *testing.T) {
	t.Log("A changed Deployment must not hide a stale LWS leader or worker template")
	dgd := &api.DynamoGraphDeployment{Spec: api.DynamoGraphDeploymentSpec{Components: []api.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "prefill", ComponentType: api.ComponentTypePrefill},
		{ComponentName: "decode", ComponentType: api.ComponentTypeDecode, Multinode: &api.MultinodeSpec{NodeCount: 2}},
	}}}
	container := []any{map[string]any{"name": "main", "env": []any{map[string]any{"name": "WORKER_REVISION", "value": "2"}}}}
	deployment := unstructured.Unstructured{Object: map[string]any{"kind": "Deployment"}}
	lws := unstructured.Unstructured{Object: map[string]any{"kind": "LeaderWorkerSet"}}
	require.NoError(t, unstructured.SetNestedSlice(deployment.Object, container, "spec", "template", "spec", "containers"))
	for _, role := range []string{"leaderTemplate", "workerTemplate"} {
		require.NoError(t, unstructured.SetNestedSlice(lws.Object, container, "spec", "leaderWorkerTemplate", role, "spec", "containers"))
	}
	deployment.SetName("prefill")
	deployment.SetLabels(map[string]string{"nvidia.com/dynamo-component": "prefill", "nvidia.com/dynamo-worker-hash": "new"})
	lws.SetLabels(map[string]string{"nvidia.com/dynamo-component": "decode", "nvidia.com/dynamo-worker-hash": "new"})
	require.NoError(t, upgradeComponentWorkerEditError(dgd, []unstructured.Unstructured{deployment, lws}, "new"))
	require.Error(t, upgradeComponentWorkerEditError(dgd, []unstructured.Unstructured{deployment, lws}, "old"))

	t.Log("Require both LWS roles and every worker component to carry the real edit")
	for _, role := range []string{"leaderTemplate", "workerTemplate"} {
		stale := lws.DeepCopy()
		oldContainers := []any{map[string]any{"name": "main", "env": []any{map[string]any{"name": "WORKER_REVISION", "value": "1"}}}}
		require.NoError(t, unstructured.SetNestedSlice(stale.Object, oldContainers, "spec", "leaderWorkerTemplate", role, "spec", "containers"))
		require.Error(t, upgradeComponentWorkerEditError(dgd, []unstructured.Unstructured{deployment, *stale}, "new"))
	}
	require.Error(t, upgradeComponentWorkerEditError(dgd, []unstructured.Unstructured{deployment}, "new"))

	t.Log("Malformed container and env entries return workload-specific errors instead of panicking")
	for _, tc := range []struct {
		name       string
		containers []any
		want       string
	}{
		{"container", []any{"invalid"}, "Deployment/prefill has non-object container at spec.template.spec.containers[0]: string"},
		{"env", []any{map[string]any{"name": "main", "env": []any{"invalid"}}}, "Deployment/prefill has non-object env entry at spec.template.spec.containers[0].env[0]: string"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			malformed := deployment.DeepCopy()
			require.NoError(t, unstructured.SetNestedSlice(malformed.Object, tc.containers, "spec", "template", "spec", "containers"))
			require.ErrorContains(t, upgradeComponentWorkerEditError(dgd, []unstructured.Unstructured{*malformed, lws}, "new"), tc.want)
		})
	}
}
