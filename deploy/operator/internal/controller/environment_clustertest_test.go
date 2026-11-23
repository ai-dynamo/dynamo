//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// These are authored semantic examples, not released-history snapshots. Seed
// only their inputs before webhooks can replace the selected origin on CREATE.
func environmentExamples() (map[string][]unstructured.Unstructured, error) {
	examples := map[string][]unstructured.Unstructured{}
	for _, origin := range []string{"1.7.0", "1.8.0"} {
		for _, provider := range []string{"grove", "component", "disaggregatedset", "deployment", "lws"} {
			objects, err := golden.ReadManifests("testdata/environment/origin-" + origin + ".yaml")
			if err != nil {
				return nil, err
			}
			namespace := "environment-" + strings.ReplaceAll(origin, ".", "-") + "-" + provider
			object := objects[0]
			if provider == "deployment" || provider == "lws" {
				object = objects[1]
			} else {
				annotations := object.GetAnnotations()
				annotations[consts.KubeAnnotationEnableGrove] = fmt.Sprint(provider == "grove")
				annotations[consts.KubeAnnotationEnableDisaggregatedSet] = fmt.Sprint(provider == "disaggregatedset")
				object.SetAnnotations(annotations)
			}
			if provider == "lws" {
				if err := unstructured.SetNestedField(object.Object, int64(2), "spec", "multinode", "nodeCount"); err != nil {
					return nil, err
				}
			}

			// The existing storage seeder requires an owner-remapping key, not a captured UID.
			object.SetUID(types.UID(namespace + "-input"))
			object.SetNamespace(namespace)
			examples[namespace] = []unstructured.Unstructured{object}
		}
	}
	return examples, nil
}

func testClusterEnvironment(t *testing.T) {
	examples, err := environmentExamples()
	require.NoError(t, err)
	for _, namespace := range slices.Sorted(maps.Keys(examples)) {
		t.Run(namespace, func(t *testing.T) {
			t.Parallel()
			t.Log("Reconcile the authored origin example through production controllers with Pods blocked")
			env := clusterTestEnv.RunTInNamespace(t, namespace)
			env.BlockWorkloads()
			root := examples[namespace][0].DeepCopy()
			origin := root.GetAnnotations()[consts.KubeAnnotationDynamoOperatorOriginVersion]
			require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(root), root))
			require.Equal(t, origin, root.GetAnnotations()[consts.KubeAnnotationDynamoOperatorOriginVersion])
			ordered := origin == "1.8.0"
			graph := root.GetKind() == "DynamoGraphDeployment"
			nodes, _, err := unstructured.NestedInt64(root.Object, "spec", "multinode", "nodeCount")
			require.NoError(t, err)
			workloads := 1
			if graph && root.GetAnnotations()[consts.KubeAnnotationEnableGrove] == "false" {
				workloads = 3 // Frontend Deployment and two worker LWS resources.
				if root.GetAnnotations()[consts.KubeAnnotationEnableDisaggregatedSet] == "true" {
					workloads = 2 // Frontend Deployment and the DisaggregatedSet.
				}
			}
			candidate := &upgradeDGDReconciles{}
			completed := &upgradeComponentReconciles{}
			env.StartManager(func(mgr ctrl.Manager) error {
				opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true, DisaggregatedSet: true}}}
				if graph {
					if err := setupUpgradeDGDController(mgr, opts, candidate); err != nil {
						return err
					}
				}
				return setupUpgradeComponentController(mgr, opts, completed)
			})

			t.Log("Check literal output order, whole-binding override and untouched reference strings")
			want := []corev1.EnvVar{{Name: "A", Value: "x"}, {Name: "B", Value: "$(A)"}, {Name: "GRAPH_REF", Value: "$(Z_BASE)"}, {Name: "Z_BASE", Value: "specific"}}
			if ordered {
				want = []corev1.EnvVar{{Name: "B", Value: "$(A)"}, {Name: "A", Value: "x"}, {Name: "Z_BASE", Value: "specific"}, {Name: "GRAPH_REF", Value: "$(Z_BASE)"}}
			}
			var before []unstructured.Unstructured
			require.Eventually(t, func() bool {
				require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(root), root))
				observed, _, _ := unstructured.NestedInt64(root.Object, "status", "observedGeneration")
				if observed != root.GetGeneration() || (graph && root.GetAnnotations()[consts.AnnotationCurrentWorkerHashV2] == "") {
					return false
				}
				var err error
				before, err = golden.CaptureOwned(t.Context(), env.RESTConfig(), env.Client(), root)
				return err == nil && environmentMatches(t, before, want, nodes > 1, workloads)
			}, 90*time.Second, 200*time.Millisecond)
			require.Equal(t, origin, root.GetAnnotations()[consts.KubeAnnotationDynamoOperatorOriginVersion])
			workerHash := root.GetAnnotations()[consts.AnnotationCurrentWorkerHashV2]
			if graph {
				require.NotEmpty(t, workerHash)
			}

			t.Log("A further successful reconcile must preserve complete workload specs, identities and generations")
			if graph {
				require.Eventually(t, func() bool {
					_, err := candidate.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(root)})
					return err == nil
				}, 90*time.Second, 200*time.Millisecond)
			}
			for _, object := range before {
				if object.GetKind() != "DynamoComponentDeployment" {
					continue
				}
				require.Eventually(t, func() bool {
					result, err := completed.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(&object)})
					return err == nil && result.IsZero()
				}, 90*time.Second, 200*time.Millisecond)
			}
			unchanged, err := golden.CaptureOwned(t.Context(), env.RESTConfig(), env.Client(), root)
			require.NoError(t, err)
			requireEnvironmentWorkloads(t, before, unchanged, false)

			t.Log("Swap only the authored frontend or standalone env order, without changing worker intent")
			require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(root), root))
			if graph {
				components, _, err := unstructured.NestedSlice(root.Object, "spec", "components")
				require.NoError(t, err)
				frontend := components[0].(map[string]any)
				containers, _, err := unstructured.NestedSlice(frontend, "podTemplate", "spec", "containers")
				require.NoError(t, err)
				swapEnvironmentEntries(t, containers)
				require.NoError(t, unstructured.SetNestedSlice(frontend, containers, "podTemplate", "spec", "containers"))
				require.NoError(t, unstructured.SetNestedSlice(root.Object, components, "spec", "components"))
			} else {
				containers, _, err := unstructured.NestedSlice(root.Object, "spec", "podTemplate", "spec", "containers")
				require.NoError(t, err)
				swapEnvironmentEntries(t, containers)
				require.NoError(t, unstructured.SetNestedSlice(root.Object, containers, "spec", "podTemplate", "spec", "containers"))
			}
			require.NoError(t, env.Client().Update(t.Context(), root))
			if ordered {
				want[0], want[1] = want[1], want[0]
			}
			var after []unstructured.Unstructured
			require.Eventually(t, func() bool {
				require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(root), root))
				observed, _, _ := unstructured.NestedInt64(root.Object, "status", "observedGeneration")
				if observed != root.GetGeneration() {
					return false
				}
				var err error
				after, err = golden.CaptureOwned(t.Context(), env.RESTConfig(), env.Client(), root)
				return err == nil && environmentMatches(t, after, want, nodes > 1, workloads)
			}, 90*time.Second, 200*time.Millisecond)
			require.Equal(t, workerHash, root.GetAnnotations()[consts.AnnotationCurrentWorkerHashV2])
			requireEnvironmentWorkloads(t, before, after, ordered)
		})
	}
}

func swapEnvironmentEntries(t *testing.T, containers []any) {
	t.Helper()
	container := containers[0].(map[string]any)
	require.Equal(t, "main", container["name"])
	variables, _, err := unstructured.NestedSlice(container, "env")
	require.NoError(t, err)
	first, second := -1, -1
	for i, variable := range variables {
		switch variable.(map[string]any)["name"] {
		case "A":
			first = i
		case "B":
			second = i
		}
	}
	require.NotEqual(t, -1, first, "missing A binding")
	require.NotEqual(t, -1, second, "missing B binding")
	variables[first], variables[second] = variables[second], variables[first]
	require.NoError(t, unstructured.SetNestedSlice(container, variables, "env"))
}

func environmentMatches(t *testing.T, objects []unstructured.Unstructured, want []corev1.EnvVar, multinode bool, workloads int) bool {
	t.Helper()
	count, workloadCount := 0, 0
	for _, object := range objects {
		if object.GetKind() == "DynamoGraphDeployment" && object.GetAnnotations()[consts.AnnotationCurrentWorkerHashV2] == "" {
			return false
		}
		if isUpgradeWorkload(object.GetKind()) {
			workloadCount++
		}
		if object.GetKind() == "DynamoComponentDeployment" {
			observed, _, _ := unstructured.NestedInt64(object.Object, "status", "observedGeneration")
			if observed != object.GetGeneration() {
				return false
			}
		}
		for _, variables := range environmentVariables(t, object) {
			if !reflect.DeepEqual(want, variables) {
				return false
			}
			count++
		}
	}
	wantCount := 1
	if multinode {
		wantCount = 2
	}
	return count == wantCount && workloadCount == workloads
}

// Select only the fixture's main-container bindings in immediate workloads.
func environmentVariables(t *testing.T, object unstructured.Unstructured) [][]corev1.EnvVar {
	t.Helper()
	var pods []map[string]any
	switch object.GetKind() {
	case "Deployment":
		pod, _, err := unstructured.NestedMap(object.Object, "spec", "template", "spec")
		require.NoError(t, err)
		pods = append(pods, pod)
	case "LeaderWorkerSet":
		for _, role := range []string{"leaderTemplate", "workerTemplate"} {
			pod, _, err := unstructured.NestedMap(object.Object, "spec", "leaderWorkerTemplate", role, "spec")
			require.NoError(t, err)
			pods = append(pods, pod)
		}
	case "PodCliqueSet":
		cliques, _, err := unstructured.NestedSlice(object.Object, "spec", "template", "cliques")
		require.NoError(t, err)
		for _, clique := range cliques {
			pod, _, err := unstructured.NestedMap(clique.(map[string]any), "spec", "podSpec")
			require.NoError(t, err)
			pods = append(pods, pod)
		}
	}
	var result [][]corev1.EnvVar
	for _, pod := range pods {
		var spec corev1.PodSpec
		require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(pod, &spec))
		for _, container := range spec.Containers {
			if container.Name != "main" || !slices.ContainsFunc(container.Env, func(e corev1.EnvVar) bool { return e.Name == "A" }) {
				continue
			}
			var variables []corev1.EnvVar
			for _, e := range container.Env {
				if e.Name == "A" || e.Name == "B" || e.Name == "GRAPH_REF" || e.Name == "Z_BASE" {
					variables = append(variables, e)
				}
			}
			result = append(result, variables)
		}
	}
	return result
}

func requireEnvironmentWorkloads(t *testing.T, before, after []unstructured.Unstructured, edited bool) {
	t.Helper()
	if !edited {
		require.NoError(t, upgradeSafetyError(false, before, after))
	}
	currentWorkloads := map[string]unstructured.Unstructured{}
	for _, object := range after {
		if isUpgradeWorkload(object.GetKind()) {
			currentWorkloads[object.GetKind()+"/"+object.GetName()] = object
		}
	}
	count := 0
	for _, old := range before {
		if !isUpgradeWorkload(old.GetKind()) {
			continue
		}
		count++
		current, exists := currentWorkloads[old.GetKind()+"/"+old.GetName()]
		require.True(t, exists, "missing workload %s/%s", old.GetKind(), old.GetName())
		require.Equal(t, old.GetUID(), current.GetUID())
		if edited && len(environmentVariables(t, old)) > 0 {
			require.Greater(t, current.GetGeneration(), old.GetGeneration())
			t.Log("Only the main-container env order may change; all other workload fields must remain identical")
			expectedSpec, _, err := unstructured.NestedMap(old.Object, "spec")
			require.NoError(t, err)
			var containerPaths [][]string
			switch old.GetKind() {
			case "Deployment":
				containerPaths = [][]string{{"template", "spec", "containers"}}
			case "LeaderWorkerSet":
				containerPaths = [][]string{
					{"leaderWorkerTemplate", "leaderTemplate", "spec", "containers"},
					{"leaderWorkerTemplate", "workerTemplate", "spec", "containers"},
				}
			case upgradePodCliqueSetKind:
				t.Log("Only the frontend env order may change; all worker cliques and other PCS fields must remain identical")
				cliques, _, err := unstructured.NestedSlice(expectedSpec, "template", "cliques")
				require.NoError(t, err)
				for _, value := range cliques {
					clique := value.(map[string]any)
					if clique["name"] != "frontend" {
						continue
					}
					containers, _, err := unstructured.NestedSlice(clique, "spec", "podSpec", "containers")
					require.NoError(t, err)
					swapEnvironmentEntries(t, containers)
					require.NoError(t, unstructured.SetNestedSlice(clique, containers, "spec", "podSpec", "containers"))
				}
				require.NoError(t, unstructured.SetNestedSlice(expectedSpec, cliques, "template", "cliques"))
			}
			for _, path := range containerPaths {
				containers, _, err := unstructured.NestedSlice(expectedSpec, path...)
				require.NoError(t, err)
				swapEnvironmentEntries(t, containers)
				require.NoError(t, unstructured.SetNestedSlice(expectedSpec, containers, path...))
			}
			actualSpec, _, err := unstructured.NestedMap(current.Object, "spec")
			require.NoError(t, err)
			require.Equal(t, expectedSpec, actualSpec, "env reorder must not change any other workload setting")
		} else {
			require.NoError(t, upgradeComponentWorkloadSafetyError(old, current))
		}
	}
	require.Equal(t, count, len(currentWorkloads), "workload inventory must not change")
}
