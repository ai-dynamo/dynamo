//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/clusterenv"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
	"sigs.k8s.io/yaml"
)

const upgradeRoot = "testdata/upgrade"

func TestMain(m *testing.M) { os.Exit(runClusterTests(m, prepareUpgradeHistory)) }

func upgradeInputs() ([]string, error) {
	return filepath.Glob(filepath.Join(upgradeRoot, "history", "*", "*", "input.yaml"))
}

func prepareUpgradeHistory(ctx context.Context, c client.Client) (func(context.Context) error, error) {
	inputs, err := upgradeInputs()
	if err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no historical inputs")
	}
	snapshots := map[string][]unstructured.Unstructured{}
	for _, input := range inputs {
		objects, err := golden.ReadManifests(input)
		if err != nil {
			return nil, err
		}
		ns := objects[0].GetNamespace()
		if _, exists := snapshots[ns]; exists {
			return nil, fmt.Errorf("duplicate historical namespace %s", ns)
		}
		snapshots[ns] = objects
	}
	return clusterenv.PrepareHistory(ctx, c, snapshots)
}

// The controller-runtime completion counter is a barrier against comparing an
// already-matching historical object before the candidate has ever reconciled it.
func completedUpgradeReconciles(t *testing.T) float64 {
	t.Helper()
	families, err := metrics.Registry.Gather()
	require.NoError(t, err)
	var count float64
	for _, family := range families {
		if family.GetName() != "controller_runtime_reconcile_total" {
			continue
		}
		for _, metric := range family.Metric {
			controller, result := "", ""
			for _, label := range metric.Label {
				if label.GetName() == "controller" {
					controller = label.GetValue()
				}
				if label.GetName() == "result" {
					result = label.GetValue()
				}
			}
			if controller == consts.ResourceTypeDynamoGraphDeployment && result != "error" {
				count += metric.GetCounter().GetValue()
			}
		}
	}
	return count
}

func captureUpgrade(t *testing.T, env *clusterenv.TestEnv) []unstructured.Unstructured {
	t.Helper()
	dgd := getOnlyClusterDGD(t, t.Context(), env.Client(), env.Namespace())
	objects, err := golden.CaptureOwned(t.Context(), env.RESTConfig(), env.Client(), dgd)
	require.NoError(t, err)
	return objects
}

func TestClusterOperatorUpgrade(t *testing.T) {
	historical, err := upgradeInputs()
	require.NoError(t, err)
	latest, err := filepath.Glob(filepath.Join(upgradeRoot, "latest", "*", "input.yaml"))
	require.NoError(t, err)
	require.NotEmpty(t, historical)
	require.NotEmpty(t, latest)
	version, err := clusterTestOperatorVersion()
	require.NoError(t, err)
	for _, input := range append(historical, latest...) {
		relative, err := filepath.Rel(upgradeRoot, filepath.Dir(input))
		require.NoError(t, err)
		t.Run(relative, func(t *testing.T) {
			ctx := t.Context()
			isHistory := strings.HasPrefix(relative, "history/")
			objects, err := golden.ReadManifests(input)
			require.NoError(t, err)
			namespace := objects[0].GetNamespace()
			var env *clusterenv.TestEnv
			if isHistory {
				env = clusterTestEnv.RunTInNamespace(t, namespace)
			} else {
				// Trigger exclusive history preparation before creating new admitted inputs.
				env = clusterTestEnv.RunT(t)
				// Latest inputs have fixed names too, so their literal hashes are reviewable.
				require.NoError(t, env.Client().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}))
				env = clusterTestEnv.RunTInNamespace(t, namespace)
			}
			env.BlockWorkloads()
			if !isHistory {
				golden.ApplyManifests(t, input, env.Client(), namespace)
			}
			before := captureUpgrade(t, env)
			if isHistory {
				// Preserve historical fields for safety comparisons even if the
				// candidate API schema normalizes them on its first read.
				ids := map[string]unstructured.Unstructured{}
				for _, object := range before {
					ids[object.GetKind()+"/"+object.GetName()] = object
				}
				before = nil
				for _, object := range objects {
					old := object.DeepCopy()
					restored, ok := ids[old.GetKind()+"/"+old.GetName()]
					require.True(t, ok)
					old.SetUID(restored.GetUID())
					before = append(before, *old)
				}
			}
			dgd := getOnlyClusterDGD(t, ctx, env.Client(), namespace)
			rollout := newDGDWorkerRolloutReconciler(env.Client(), nil)
			transition, err := rollout.planUnsupportedWorkerHashTransition(dgd)
			require.NoError(t, err)
			// Report independently of the golden and keep observing to emit the full diff.
			if isHistory {
				require.NotEmpty(t, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
				if transition.hashChanged {
					t.Error("operator-only takeover requests a new worker generation")
				}
			}
			baseline := completedUpgradeReconciles(t)
			env.StartManager(func(mgr ctrl.Manager) error {
				opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}}
				if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts}); err != nil {
					return err
				}
				return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: opts})
			})
			t.Log("Wait for a completed production reconcile, then compare the full owned resource graph")
			require.Eventually(t, func() bool { return completedUpgradeReconciles(t) >= baseline+1 }, 90*time.Second, 200*time.Millisecond, "candidate did not complete a successful reconciliation")
			after := captureUpgrade(t, env)
			if isHistory {
				if err := upgradeSafetyError(transition.hashChanged, before, after); err != nil {
					t.Error(err)
				}
			} else {
				dgd = getOnlyClusterDGD(t, ctx, env.Client(), namespace)
				require.Equal(t, version, dgd.Annotations["nvidia.com/dynamo-operator-origin-version"])
			}
			golden.MatchCapturedManifests(t, after, filepath.Join(filepath.Dir(input), "output.yaml"))
			if root := os.Getenv("DYNAMO_UPGRADE_CAPTURE_DIR"); root != "" && !t.Failed() {
				path := filepath.Join(root, relative, "input.yaml")
				require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
				require.NoError(t, golden.WriteManifests(path, after))
			}
			if isHistory && strings.HasSuffix(relative, "/sglang-1.5") && !t.Failed() {
				t.Log("A real worker environment edit must still request and persist a new generation")
				current := getOnlyClusterDGD(t, ctx, env.Client(), namespace)
				oldHash := current.Annotations[consts.AnnotationCurrentWorkerHashV2]
				current.Spec.Components[0].PodTemplate.Spec.Containers[0].Env = append(current.Spec.Components[0].PodTemplate.Spec.Containers[0].Env, corev1.EnvVar{Name: "WORKER_REVISION", Value: "2"})
				edit, err := rollout.planUnsupportedWorkerHashTransition(current)
				require.NoError(t, err)
				require.True(t, edit.hashChanged)
				require.NoError(t, env.Client().Update(ctx, current))
				require.Eventually(t, func() bool {
					var live api.DynamoGraphDeployment
					if env.Client().Get(ctx, client.ObjectKeyFromObject(current), &live) != nil {
						return false
					}
					want, err := dynamo.ComputeDGDWorkersSpecHash(&live)
					if err != nil {
						return false
					}
					return want != oldHash && live.Annotations[consts.AnnotationCurrentWorkerHashV2] == want && live.Status.ObservedGeneration == live.Generation
				}, 60*time.Second, 200*time.Millisecond)
				changed := captureUpgrade(t, env)
				require.Error(t, upgradeSafetyError(false, after, changed))
				// Verify the workload actually carries the edit, not just the annotation.
				for _, object := range changed {
					if object.GetKind() == upgradePodCliqueSetKind {
						templates, err := upgradeWorkerTemplates(object)
						require.NoError(t, err)
						found := false
						for _, value := range templates {
							pod := value.(map[string]any)["podSpec"].(map[string]any)
							containers := pod["containers"].([]any)
							for _, container := range containers {
								for _, variable := range container.(map[string]any)["env"].([]any) {
									if variable.(map[string]any)["name"] == "WORKER_REVISION" {
										found = true
									}
								}
							}
						}
						require.True(t, found, "worker edit did not reach PCS")
					}
				}
			}
		})
	}
}

// Upgrade capture and replay use the same version source as production admission.
func clusterTestOperatorVersion() (string, error) {
	data, err := os.ReadFile("../../../helm/charts/platform/components/operator/Chart.yaml")
	if err != nil {
		return "", err
	}
	var chart struct{ AppVersion string }
	if err := yaml.Unmarshal(data, &chart); err != nil {
		return "", err
	}
	if chart.AppVersion == "" {
		return "", fmt.Errorf("operator chart has no appVersion")
	}
	return chart.AppVersion, nil
}
