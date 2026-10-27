//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	configvalidation "github.com/ai-dynamo/dynamo/deploy/operator/api/config/validation"
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

func upgradeInputs(role string) ([]string, error) {
	var inputs []string
	err := filepath.WalkDir(filepath.Join(upgradeRoot, role), func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && entry.Name() == "input.yaml" {
			inputs = append(inputs, path)
		}
		return nil
	})
	return inputs, err
}

func prepareUpgradeHistory(ctx context.Context, c client.Client) (func(context.Context) error, error) {
	inputs, err := upgradeInputs("history")
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

// Capture a graph only if no candidate child call overlapped the API reads.
func captureUpgradeComponent(t *testing.T, env *clusterenv.TestEnv, completed *upgradeComponentReconciles, components int) ([]unstructured.Unstructured, error) {
	t.Helper()

	// Record the candidate call revision before reading the resource graph.
	completed.mu.Lock()
	revision := completed.revision
	completed.mu.Unlock()

	// Validate the same graph that the caller will compare against the golden or predecessor.
	objects := captureUpgrade(t, env)
	if err := completed.graphReady(objects, components); err != nil {
		return nil, err
	}

	// A call that started or finished during capture invalidates these observations.
	completed.mu.Lock()
	defer completed.mu.Unlock()
	if revision != completed.revision {
		return nil, fmt.Errorf("candidate component reconciliation overlapped capture")
	}
	return objects, nil
}

func TestClusterOperatorUpgrade(t *testing.T) {
	historical, err := upgradeInputs("history")
	require.NoError(t, err)
	latest, err := upgradeInputs("latest")
	require.NoError(t, err)
	require.NotEmpty(t, historical)
	require.NotEmpty(t, latest)
	version, err := clusterTestOperatorVersion()
	require.NoError(t, err)

	// Capturing also admits each latest scenario in its final historical namespace.
	type scenario struct {
		input, relative string
		promote         bool
	}
	var scenarios []scenario
	for _, input := range append(historical, latest...) {
		relative, err := filepath.Rel(upgradeRoot, filepath.Dir(input))
		require.NoError(t, err)
		scenarios = append(scenarios, scenario{input: input, relative: relative})
	}
	if os.Getenv("DYNAMO_UPGRADE_CAPTURE_DIR") != "" {
		parts := strings.Split(version, ".")
		generation := parts[0] + "." + parts[1] + ".0"
		for _, input := range latest {
			path, err := filepath.Rel(filepath.Join(upgradeRoot, "latest"), filepath.Dir(input))
			require.NoError(t, err)
			scenarios = append(scenarios, scenario{input: input, relative: filepath.Join("history", generation, path), promote: true})
		}
	}

	// Reject collisions across historical, latest and promoted roles before creating any latest namespace.
	namespaces := map[string]string{}
	for _, item := range scenarios {
		objects, err := golden.ReadManifests(item.input)
		require.NoError(t, err)
		namespace := objects[0].GetNamespace()
		if item.promote {
			namespace = upgradeScenarioNamespace(item.relative)
		} else if strings.HasPrefix(item.relative, "latest/") {
			require.Equal(t, upgradeScenarioNamespace(item.relative), namespace)
		}
		require.NotContains(t, namespaces, namespace, "namespace collision: %s", item.relative)
		namespaces[namespace] = item.relative
	}
	for _, item := range scenarios {
		input, relative := item.input, item.relative
		t.Run(relative, func(t *testing.T) {
			ctx := t.Context()
			isHistory := strings.HasPrefix(relative, "history/") && !item.promote
			objects, err := golden.ReadManifests(input)
			require.NoError(t, err)
			namespace := objects[0].GetNamespace()
			if item.promote {
				namespace = upgradeScenarioNamespace(relative)
			}
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
				// Set the namespace on CREATE, never on a reconciled snapshot.
				for i := range objects {
					objects[i].SetNamespace(namespace)
					require.NoError(t, env.Client().Create(ctx, &objects[i]))
				}
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
					old.SetGeneration(restored.GetGeneration())
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
			completed := &upgradeComponentReconciles{}
			baseline := completedUpgradeReconciles(t)
			// Render valid MPI Secret references without starting model-serving Pods.
			operatorConfig := clusterTestRestrictedConfig(namespace)
			operatorConfig.MPI.SSHSecretName = upgradeMPISSHSecret
			operatorConfig.MPI.SSHSecretNamespace = namespace
			require.Empty(t, configvalidation.ValidateOperatorConfiguration(operatorConfig))
			env.StartManager(func(mgr ctrl.Manager) error {
				opts := SetupOptions{Config: operatorConfig, RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true, DisaggregatedSet: true}}}
				if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts}); err != nil {
					return err
				}
				return setupUpgradeComponentController(mgr, opts, completed)
			})
			t.Log("Wait for a completed production reconcile, then compare the full owned resource graph")
			require.Eventually(t, func() bool { return completedUpgradeReconciles(t) >= baseline+1 }, 90*time.Second, 200*time.Millisecond, "candidate did not complete a successful reconciliation")
			if !isHistory {
				t.Log("Wait for the admitted DGD and its worker hash to converge before capture")
				require.Eventually(t, func() bool {
					live := getOnlyClusterDGD(t, ctx, env.Client(), namespace)
					want, err := dynamo.ComputeDGDWorkersSpecHash(live)
					return err == nil && live.Status.ObservedGeneration == live.Generation && live.Annotations[consts.AnnotationCurrentWorkerHashV2] == want
				}, 90*time.Second, 200*time.Millisecond)
			}
			dgd = getOnlyClusterDGD(t, ctx, env.Client(), namespace)
			var after []unstructured.Unstructured
			if dgd.Annotations[consts.KubeAnnotationEnableGrove] == "false" {
				t.Log("Wait for each DCD and its immediate workload without requiring serving Pods")
				require.Eventually(t, func() bool {
					var err error
					after, err = captureUpgradeComponent(t, env, completed, upgradeExpectedDCDs(dgd))
					return err == nil
				}, 90*time.Second, 200*time.Millisecond, "component workload graph did not converge")
			} else {
				after = captureUpgrade(t, env)
			}
			checkUpgradeWorkloadMatrix(t, relative, dgd, after)
			checkUpgradeGroveMatrix(t, relative, getOnlyClusterDGD(t, ctx, env.Client(), namespace), after)
			if isHistory {
				if err := upgradeSafetyError(transition.hashChanged, before, after); err != nil {
					t.Error(err)
				}
			} else {
				dgd = getOnlyClusterDGD(t, ctx, env.Client(), namespace)
				require.Equal(t, version, dgd.Annotations["nvidia.com/dynamo-operator-origin-version"])
			}
			if !item.promote {
				path := filepath.Join(filepath.Dir(input), "output.yaml")
				var variables golden.Variables
				if !isHistory {
					variables = upgradeLatestVariables(t, path, version)
				}
				golden.MatchCapturedManifestsWithVariables(t, after, path, variables)
			}
			if root := os.Getenv("DYNAMO_UPGRADE_CAPTURE_DIR"); root != "" && !t.Failed() {
				path := filepath.Join(root, relative, "input.yaml")
				require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
				require.NoError(t, golden.WriteManifests(path, after))
			}
			if !isHistory && !item.promote && (strings.HasSuffix(relative, "/component") || strings.HasSuffix(relative, "/disaggregatedset")) && !t.Failed() {
				t.Log("An intentional worker edit must reach every immediate worker template while Pods are blocked")
				var workers []string
				current := getOnlyClusterDGD(t, ctx, env.Client(), namespace)
				for i := range current.Spec.Components {
					if current.Spec.Components[i].ComponentType == api.ComponentTypeFrontend {
						continue
					}
					workers = append(workers, current.Spec.Components[i].ComponentName)
					container := &current.Spec.Components[i].PodTemplate.Spec.Containers[0]
					container.Env = append(container.Env, corev1.EnvVar{Name: "WORKER_REVISION", Value: "2"})
				}
				require.NoError(t, env.Client().Update(ctx, current))
				require.Eventually(t, func() bool {
					return upgradeWorkerEditError(captureUpgrade(t, env), workers) == nil
				}, 60*time.Second, 200*time.Millisecond, "worker edit did not reach every immediate template")
				require.Error(t, upgradeSafetyError(false, after, captureUpgrade(t, env)))
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
	t.Run("component-history-barrier", testUpgradeComponentHistoryBarrier)
	t.Run("disaggregatedset-replay", testUpgradeDisaggregatedSetReplay)
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
