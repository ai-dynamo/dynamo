//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"
	"time"

	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// Exercise a populated DS graph without fabricating released historical provenance.
func testUpgradeDisaggregatedSetReplay(t *testing.T) {
	var snapshot []unstructured.Unstructured
	var namespace string
	var seedClient client.Client
	t.Run("capture", func(t *testing.T) {
		t.Log("Capture production DS lowering and its auxiliary frontend without downstream controllers")
		env := clusterTestEnv.RunT(t)
		namespace, seedClient = env.Namespace(), env.Client()
		env.BlockWorkloads()
		objects, err := golden.ReadManifests("testdata/upgrade/latest/sglang/disaggregated/multi/disaggregatedset/input.yaml")
		require.NoError(t, err)
		for i := range objects {
			objects[i].SetNamespace(namespace)
			require.NoError(t, seedClient.Create(t.Context(), &objects[i]))
		}
		completed := &upgradeComponentReconciles{}
		env.StartManager(func(mgr ctrl.Manager) error {
			opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true, DisaggregatedSet: true}}}
			if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts}); err != nil {
				return err
			}
			return setupUpgradeComponentController(mgr, opts, completed)
		})
		require.Eventually(t, func() bool {
			var err error
			snapshot, err = captureUpgradeComponent(t, env, completed, 1)
			return err == nil
		}, 90*time.Second, 100*time.Millisecond)
	})
	require.NotEmpty(t, snapshot)

	t.Run("takeover", func(t *testing.T) {
		t.Log("Restore the concrete DS graph with the workload quota already active")
		require.NoError(t, seedClient.Create(t.Context(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}))
		env := clusterTestEnv.RunTInNamespace(t, namespace)
		env.BlockWorkloads()
		require.NoError(t, golden.ApplySnapshot(t.Context(), env.Client(), namespace, snapshot))
		before := captureUpgrade(t, env)
		completed := &upgradeComponentReconciles{}
		baseline := completedUpgradeReconciles(t)
		env.StartManager(func(mgr ctrl.Manager) error {
			opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true, DisaggregatedSet: true}}}
			if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts}); err != nil {
				return err
			}
			return setupUpgradeComponentController(mgr, opts, completed)
		})
		require.Eventually(t, func() bool { return completedUpgradeReconciles(t) > baseline }, 90*time.Second, 100*time.Millisecond)
		var after []unstructured.Unstructured
		require.Eventually(t, func() bool {
			var err error
			after, err = captureUpgradeComponent(t, env, completed, 1)
			return err == nil
		}, 90*time.Second, 100*time.Millisecond)
		require.NoError(t, upgradeSafetyError(false, before, after))
		checkUpgradeWorkloadMatrix(t, "latest/sglang/disaggregated/multi/disaggregatedset", getOnlyClusterDGD(t, t.Context(), env.Client(), namespace), after)

		t.Log("The upstream DS admission handler rejects a schema-valid rollout that cannot make progress")
		var workloads disaggregatedsetv1.DisaggregatedSetList
		require.NoError(t, env.Client().List(t.Context(), &workloads, client.InNamespace(namespace)))
		require.Len(t, workloads.Items, 1)
		invalid := workloads.Items[0].DeepCopy()
		invalid.Spec.Roles[0].Spec.RolloutStrategy.RollingUpdateConfiguration = &leaderworkersetv1.RollingUpdateConfiguration{
			MaxSurge: intstr.FromInt(0), MaxUnavailable: intstr.FromInt(0),
		}
		require.ErrorContains(t, env.Client().Update(t.Context(), invalid), "vdisaggregatedset.kb.io")

		t.Log("The independent assertion rejects changed DS worker templates without a golden comparison")
		changed := make([]unstructured.Unstructured, len(after))
		for i := range after {
			changed[i] = *after[i].DeepCopy()
			if changed[i].GetKind() != "DisaggregatedSet" {
				continue
			}
			roles, _, err := unstructured.NestedSlice(changed[i].Object, "spec", "roles")
			require.NoError(t, err)
			role := roles[0].(map[string]any)
			require.NoError(t, unstructured.SetNestedField(role, "incompatible", "spec", "leaderWorkerTemplate", "workerTemplate", "spec", "hostname"))
			require.NoError(t, unstructured.SetNestedSlice(changed[i].Object, roles, "spec", "roles"))
		}
		require.ErrorContains(t, upgradeSafetyError(false, before, changed), "rollout intent")
	})
}
