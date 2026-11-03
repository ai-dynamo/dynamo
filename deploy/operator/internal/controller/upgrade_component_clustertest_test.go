//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"
	"time"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
)

func setupUpgradeComponentController(mgr ctrl.Manager, opts SetupOptions, completed *upgradeComponentReconciles) error {
	// Keep the production reconciler and its complete watch/index wiring.
	r := &DynamoComponentDeploymentReconciler{
		Client: mgr.GetClient(), Recorder: mgr.GetEventRecorder("dynamocomponentdeployment"),
		Config: opts.Config, RuntimeConfig: opts.RuntimeConfig,
	}
	builder, err := r.controllerBuilder(mgr)
	if err != nil {
		return err
	}
	completed.next = r
	return builder.WithOptions(controller.Options{MaxConcurrentReconciles: 2}).Complete(completed)
}

// incompatibleUpgradeClient changes only the candidate's in-memory rendering input.
// Neither the persisted DCD spec nor the restored history is edited.
type incompatibleUpgradeClient struct{ client.Client }

func (c incompatibleUpgradeClient) Get(ctx context.Context, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
	if err := c.Client.Get(ctx, key, object, opts...); err != nil {
		return err
	}
	if dcd, ok := object.(*api.DynamoComponentDeployment); ok && dcd.Spec.ComponentType != api.ComponentTypeFrontend {
		container := &dcd.Spec.PodTemplate.Spec.Containers[0]
		container.Env = append(container.Env, corev1.EnvVar{Name: "INCOMPATIBLE_CANDIDATE", Value: "true"})
	}
	return nil
}

func testUpgradeComponentHistoryBarrier(t *testing.T) {
	for _, topology := range []string{"single", "multi"} {
		t.Run(topology, func(t *testing.T) {
			testUpgradeComponentHistory(t, "testdata/upgrade/latest/sglang/aggregated/"+topology+"/component/input.yaml")
		})
	}
}

func testUpgradeComponentHistory(t *testing.T, input string) {
	var snapshot []unstructured.Unstructured
	var namespace string
	var seedClient client.Client

	t.Run("capture-predecessor", func(t *testing.T) {
		t.Log("Capture a fully reconciled Component graph with production admission and no serving Pods")
		env := clusterTestEnv.RunT(t)
		namespace, seedClient = env.Namespace(), env.Client()
		env.BlockWorkloads()
		objects, err := golden.ReadManifests(input)
		require.NoError(t, err)
		for i := range objects {
			objects[i].SetNamespace(namespace)
			require.NoError(t, seedClient.Create(t.Context(), &objects[i]))
		}
		completed := &upgradeComponentReconciles{}
		env.StartManager(func(mgr ctrl.Manager) error {
			opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}}
			if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts}); err != nil {
				return err
			}
			return setupUpgradeComponentController(mgr, opts, completed)
		})
		require.Eventually(t, func() bool {
			var err error
			snapshot, err = captureUpgradeComponent(t, env, completed, 2)
			return err == nil
		}, 90*time.Second, 100*time.Millisecond)
	})
	require.NotEmpty(t, snapshot)

	for _, incompatible := range []bool{false, true} {
		name := "unchanged"
		if incompatible {
			name = "incompatible-candidate"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("Restore predecessor specs, annotations, status and reconstructed owners before starting controllers")
			require.NoError(t, seedClient.Create(t.Context(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}))
			env := clusterTestEnv.RunTInNamespace(t, namespace)
			env.BlockWorkloads()
			require.NoError(t, golden.ApplySnapshot(t.Context(), env.Client(), namespace, snapshot))
			before := captureUpgrade(t, env)
			require.NoError(t, upgradeComponentGraphReady(before, 2), "persisted predecessor evidence already matches")
			completed := &upgradeComponentReconciles{gates: map[types.NamespacedName]chan struct{}{}}
			var children []types.NamespacedName
			for _, object := range before {
				if object.GetKind() == "DynamoComponentDeployment" {
					key := types.NamespacedName{Namespace: namespace, Name: object.GetName()}
					completed.gates[key] = make(chan struct{})
					children = append(children, key)
				}
			}
			require.Len(t, children, 2)
			// Release all gates before namespace cleanup, including assertion failures.
			t.Cleanup(func() {
				for _, gate := range completed.gates {
					select {
					case <-gate:
					default:
						close(gate)
					}
				}
			})
			baseline := completedUpgradeReconciles(t)
			env.StartManager(func(mgr ctrl.Manager) error {
				opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}}
				if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts}); err != nil {
					return err
				}
				if err := setupUpgradeComponentController(mgr, opts, completed); err != nil {
					return err
				}
				if incompatible {
					r := completed.next.(*DynamoComponentDeploymentReconciler)
					r.Client = incompatibleUpgradeClient{Client: mgr.GetClient()}
				}
				return nil
			})

			t.Log("A completed candidate DGD cannot release capture while its DCDs are held")
			require.Eventually(t, func() bool { return completedUpgradeReconciles(t) > baseline }, 90*time.Second, 100*time.Millisecond)
			require.ErrorContains(t, completed.graphReady(captureUpgrade(t, env), 2), "no completed candidate")
			require.NoError(t, upgradeSafetyError(false, before, captureUpgrade(t, env)), "the premature comparison would incorrectly pass")

			t.Log("Releasing one DCD cannot substitute for the other child's successful candidate call")
			close(completed.gates[children[0]])
			require.Eventually(t, func() bool {
				completed.mu.Lock()
				defer completed.mu.Unlock()
				return completed.completed[children[0]]
			}, 90*time.Second, 100*time.Millisecond)
			require.Error(t, completed.graphReady(captureUpgrade(t, env), 2))

			t.Log("Release the remaining DCD and compare the candidate-rendered graph independently of goldens")
			close(completed.gates[children[1]])
			require.Eventually(t, func() bool {
				completed.mu.Lock()
				defer completed.mu.Unlock()
				return completed.completed[children[1]]
			}, 90*time.Second, 100*time.Millisecond)
			if incompatible {
				require.Error(t, upgradeSafetyError(false, before, captureUpgrade(t, env)))
			} else {
				var after []unstructured.Unstructured
				require.Eventually(t, func() bool {
					var err error
					after, err = captureUpgradeComponent(t, env, completed, 2)
					return err == nil
				}, 90*time.Second, 100*time.Millisecond)
				require.NoError(t, upgradeSafetyError(false, before, after))
			}
		})
	}
}
