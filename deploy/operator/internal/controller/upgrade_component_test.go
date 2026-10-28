// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// A DGD reconcile can finish before its DCD controllers have rendered workloads.
//
//nolint:unparam // Tagged cluster tests pass the actual component count; unit tests use one.
func upgradeComponentGraphReady(objects []unstructured.Unstructured, components int) error {
	dcds := 0
	for _, object := range objects {
		if object.GetKind() == upgradePodCliqueSetKind {
			return fmt.Errorf("component scenario generated a PodCliqueSet")
		}
		if object.GetKind() != "DynamoComponentDeployment" {
			continue
		}
		dcds++
		observed, _, err := unstructured.NestedInt64(object.Object, "status", "observedGeneration")
		if err != nil {
			return err
		}
		if observed != object.GetGeneration() {
			return fmt.Errorf("DCD %s has not reconciled its generation", object.GetName())
		}
		kind := "Deployment"
		nodes, _, err := unstructured.NestedInt64(object.Object, "spec", "multinode", "nodeCount")
		if err != nil {
			return err
		}
		if nodes > 1 {
			kind = "LeaderWorkerSet"
		}
		found := false
		for _, workload := range objects {
			if workload.GetKind() != kind {
				continue
			}
			for _, owner := range workload.GetOwnerReferences() {
				if owner.UID == object.GetUID() && owner.Controller != nil && *owner.Controller && workload.GetAnnotations()["nvidia.com/last-applied-generation"] == strconv.FormatInt(object.GetGeneration(), 10) {
					found = true
				}
			}
		}
		if !found {
			return fmt.Errorf("DCD %s has no reconciled %s", object.GetName(), kind)
		}
	}
	if dcds != components {
		return fmt.Errorf("expected %d DCDs, observed %d", components, dcds)
	}
	return nil
}

func TestUpgradeComponentSafety(t *testing.T) {
	for _, kind := range []string{"Deployment", "LeaderWorkerSet", "DisaggregatedSet"} {
		t.Run(kind, func(t *testing.T) {
			t.Log("Preserve immediate workload identity and spec independently of golden updates")
			old := unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "apps/v1", "kind": kind,
				"metadata": map[string]any{"name": "worker", "uid": "original", "generation": int64(1)},
				"spec":     map[string]any{"replicas": int64(2)},
			}}
			before := []unstructured.Unstructured{old}
			metadata := old.DeepCopy()
			metadata.SetAnnotations(map[string]string{"reviewed-default": "true"})
			require.NoError(t, upgradeSafetyError(false, before, []unstructured.Unstructured{*metadata}))

			t.Log("Reject replacement, intermediate spec changes, and changed rollout intent")
			replaced := old.DeepCopy()
			replaced.SetUID("replacement")
			require.ErrorContains(t, upgradeSafetyError(false, before, []unstructured.Unstructured{*replaced}), "active identity")
			changed := old.DeepCopy()
			changed.SetGeneration(2)
			require.ErrorContains(t, upgradeSafetyError(false, before, []unstructured.Unstructured{*changed}), "generation")
			changed = old.DeepCopy()
			path := []string{"spec", "template", "spec", "hostname"}
			if kind == "LeaderWorkerSet" {
				path = []string{"spec", "leaderWorkerTemplate", "workerTemplate", "spec", "hostname"}
			}
			if kind == "DisaggregatedSet" {
				path = []string{"spec", "roles"}
				require.NoError(t, unstructured.SetNestedSlice(changed.Object, []any{map[string]any{"name": "changed-role"}}, path...))
			} else {
				require.NoError(t, unstructured.SetNestedField(changed.Object, "changed-worker", path...))
			}
			require.ErrorContains(t, upgradeSafetyError(false, before, []unstructured.Unstructured{*changed}), "rollout intent")
			require.ErrorContains(t, upgradeSafetyError(false, before, nil), "inventory")
		})
	}
}

func TestUpgradeComponentGraphReady(t *testing.T) {
	for _, kind := range []string{"Deployment", "LeaderWorkerSet"} {
		t.Run(kind, func(t *testing.T) {
			t.Log("A completed DGD reconcile alone cannot satisfy the child workload barrier")
			require.Error(t, upgradeComponentGraphReady(nil, 1))
			dcd := unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoComponentDeployment",
				"metadata": map[string]any{"name": "worker", "uid": "dcd", "generation": int64(2)},
				"spec":     map[string]any{}, "status": map[string]any{"observedGeneration": int64(1)},
			}}
			if kind == "LeaderWorkerSet" {
				require.NoError(t, unstructured.SetNestedField(dcd.Object, int64(2), "spec", "multinode", "nodeCount"))
			}
			workload := unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "apps/v1", "kind": kind,
				"metadata": map[string]any{"name": "worker", "annotations": map[string]any{"nvidia.com/last-applied-generation": "2"}, "ownerReferences": []any{map[string]any{"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoComponentDeployment", "name": "worker", "uid": "dcd", "controller": true}}},
			}}
			require.ErrorContains(t, upgradeComponentGraphReady([]unstructured.Unstructured{dcd, workload}, 1), "has not reconciled")

			t.Log("Require the DCD generation and its owned workload to agree without checking Pod readiness")
			require.NoError(t, unstructured.SetNestedField(dcd.Object, int64(2), "status", "observedGeneration"))
			require.NoError(t, upgradeComponentGraphReady([]unstructured.Unstructured{dcd, workload}, 1))
			workload.SetAnnotations(map[string]string{"nvidia.com/last-applied-generation": "1"})
			require.ErrorContains(t, upgradeComponentGraphReady([]unstructured.Unstructured{dcd, workload}, 1), "no reconciled")
			workload.SetAnnotations(map[string]string{"nvidia.com/last-applied-generation": "2"})
			owners := workload.GetOwnerReferences()
			owners[0].UID = "another-dcd"
			workload.SetOwnerReferences(owners)
			require.ErrorContains(t, upgradeComponentGraphReady([]unstructured.Unstructured{dcd, workload}, 1), "no reconciled")
		})
	}
}

func TestUpgradeComponentVersionBindings(t *testing.T) {
	for _, scenario := range []string{"aggregated/single/component", "aggregated/multi/component", "disaggregated/multi/disaggregatedset"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Keep generated component identities exact across an operator version bump")
			path := "testdata/upgrade/latest/sglang/" + scenario + "/output.yaml"
			previous := upgradeLatestVariables(t, path, "1.6.0")
			current := upgradeLatestVariables(t, path, "1.7.0")
			require.NotEqual(t, previous["worker-hash"], current["worker-hash"])
			bindings := 0
			for name, value := range current {
				if !strings.HasPrefix(name, "worker-string:") {
					continue
				}
				bindings++
				require.Contains(t, value, current["worker-hash"])
				require.NotEqual(t, previous[name], value)
			}
			require.Positive(t, bindings, "component names must follow the reviewed DGD hash")
			if strings.HasSuffix(scenario, "/disaggregatedset") {
				require.NotEqual(t, "pending", current["ds-revision"])
				require.NotEqual(t, previous["ds-revision"], current["ds-revision"])
			}
		})
	}
}
