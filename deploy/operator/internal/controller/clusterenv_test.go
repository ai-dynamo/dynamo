//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	semver "github.com/Masterminds/semver/v3"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	dynamotesting "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/clusterenv"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	grovemock "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/mocks/grove"
	lwsmock "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/mocks/lws"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorchart"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

var clusterTestEnv *clusterenv.Env

func runClusterTests(m *testing.M, prepare func(context.Context, client.Client) (func(context.Context) error, error)) int {
	if prepare != nil {
		logf.SetLogger(zap.New(zap.UseDevMode(true)))
	} else {
		logf.SetLogger(logr.Discard())
	}
	operatorVersion, err := clusterTestOperatorVersion()
	if err != nil {
		fmt.Fprintf(os.Stderr, "derive cluster-test operator version: %v\n", err)
		return 1
	}
	operatorConfig := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(operatorConfig)
	operatorConfig.GPU.DiscoveryEnabled = ptr.To(false)
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true, DisaggregatedSet: true}}
	additionalAdmission := lwsmock.Configurations()
	groveAdmission := grovemock.Configurations()
	additionalAdmission.Mutating = append(additionalAdmission.Mutating, groveAdmission.Mutating...)
	additionalAdmission.Validating = append(additionalAdmission.Validating, groveAdmission.Validating...)

	var managerIdentity *rest.ImpersonationConfig
	if prepare != nil {
		managerIdentity = &rest.ImpersonationConfig{
			UserName: "system:serviceaccount:dynamo-system:dynamo-operator",
			Groups:   []string{"system:masters"},
		}
	}
	clusterTestEnv = clusterenv.New(clusterenv.Options{
		Scheme:                 newClusterTestScheme(),
		BeforeWebhooks:         prepare,
		ManagerImpersonation:   managerIdentity,
		AdditionalAdmission:    additionalAdmission,
		WebhookProxyImage:      os.Getenv("DYNAMO_CLUSTERTEST_WEBHOOK_PROXY_IMAGE"),
		EventuallyTimeout:      2 * time.Minute,
		NamespaceDeleteTimeout: 5 * time.Minute,
		SetupWebhooks: func(mgr ctrl.Manager) error {
			if err := webhooksetup.Setup(mgr, webhooksetup.Options{
				Config:            operatorConfig,
				RuntimeConfig:     runtimeConfig,
				OperatorVersion:   operatorVersion,
				OperatorPrincipal: "system:serviceaccount:dynamo-system:dynamo-operator",
			}); err != nil {
				return err
			}
			if err := lwsmock.Setup(mgr); err != nil {
				return err
			}
			return grovemock.Setup(mgr)
		},
	})
	return clusterTestEnv.RunM(m)
}

func TestClusterDynamoComponentDeploymentManifests(t *testing.T) {
	inputs, err := filepath.Glob(filepath.Join("testdata/dcd", "*", "input.yaml"))
	if err != nil {
		t.Fatalf("find DCD cluster-test scenarios: %v", err)
	}
	if len(inputs) == 0 {
		t.Fatal("no DCD cluster-test scenarios found")
	}
	for _, input := range inputs {
		scenarioDir := filepath.Dir(input)
		t.Run(filepath.Base(scenarioDir), func(t *testing.T) {
			t.Log("Create an isolated namespace in the explicitly unlocked cluster")
			env := clusterTestEnv.RunT(t)

			t.Log("Block Pods and ReplicaSets from actuating terminal workload manifests")
			env.BlockWorkloads()

			t.Log("Apply the scenario input manifests through Kubernetes admission")
			golden.ApplyManifests(t, filepath.Join(scenarioDir, "input.yaml"), env.Client(), env.Namespace())

			t.Log("Start the DynamoComponentDeployment controller")
			operatorConfig := clusterTestRestrictedConfig(env.Namespace())
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
			env.StartManager(func(mgr ctrl.Manager) error {
				setupOptions := SetupOptions{Config: operatorConfig, RuntimeConfig: runtimeConfig}
				return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: setupOptions})
			})

			t.Log("Wait for the complete generated manifest contract")
			golden.EventuallyMatchManifests(t, env.Client(), env.Namespace(), filepath.Join(scenarioDir, "output.yaml"))
		})
	}
}

func TestClusterDynamoGraphDeploymentManifests(t *testing.T) {
	inputs, err := filepath.Glob(filepath.Join("testdata/dgd", "*", "input.yaml"))
	if err != nil {
		t.Fatalf("find DGD cluster-test scenarios: %v", err)
	}
	if len(inputs) == 0 {
		t.Fatal("no DGD cluster-test scenarios found")
	}
	for _, input := range inputs {
		scenarioDir := filepath.Dir(input)
		t.Run(filepath.Base(scenarioDir), func(t *testing.T) {
			t.Log("Create an isolated namespace in the explicitly unlocked cluster")
			env := clusterTestEnv.RunT(t)

			t.Log("Block Pods and ReplicaSets from actuating terminal workload manifests")
			env.BlockWorkloads()

			t.Log("Apply the scenario input manifests through Kubernetes admission")
			golden.ApplyManifests(t, filepath.Join(scenarioDir, "input.yaml"), env.Client(), env.Namespace())

			t.Log("Start the DynamoGraphDeployment and DynamoComponentDeployment controllers")
			operatorConfig := clusterTestRestrictedConfig(env.Namespace())
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
			env.StartManager(func(mgr ctrl.Manager) error {
				setupOptions := SetupOptions{Config: operatorConfig, RuntimeConfig: runtimeConfig}
				if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: setupOptions}); err != nil {
					return err
				}
				return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: setupOptions})
			})

			liveDGD := getOnlyClusterDGD(t, t.Context(), env.Client(), env.Namespace())
			t.Log("Wait for the complete generated manifest contract with the live worker hash")
			golden.EventuallyMatchManifestsWithVariables(
				t,
				env.Client(),
				env.Namespace(),
				filepath.Join(scenarioDir, "output.yaml"),
				workerHashGoldenVariables(t, liveDGD),
			)
		})
	}
}

func TestClusterDynamoGraphDeploymentRequestProfilesAndCreatesWorkloadManifests(t *testing.T) {
	inputs, err := filepath.Glob(filepath.Join("testdata/dgdr", "*", "input.yaml"))
	if err != nil {
		t.Fatalf("find DGDR cluster-test scenarios: %v", err)
	}
	if len(inputs) == 0 {
		t.Fatal("no DGDR cluster-test scenarios found")
	}
	profilerImage := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_PROFILER_IMAGE")
	operatorImage := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_OPERATOR_IMAGE")
	for _, input := range inputs {
		scenarioDir := filepath.Dir(input)
		t.Run(filepath.Base(scenarioDir), func(t *testing.T) {
			ctx := t.Context()

			t.Log("Create an isolated namespace in the explicitly unlocked cluster")
			env := clusterTestEnv.RunT(t)

			t.Log("Block ReplicaSets while allowing the profiler Job Pod to run")
			env.BlockReplicaSets()

			t.Log("Install the profiling Job identity and permissions rendered from the production Helm chart")
			rbacObjects, err := operatorchart.Render("templates/profiling-job-rbac.yaml", operatorchart.Options{
				ReleaseName: "clustertest",
				Namespace:   env.Namespace(),
				Values: map[string]any{
					"namespaceRestriction": map[string]any{"enabled": true},
				},
			})
			if err != nil {
				t.Fatalf("render profiling Job RBAC: %v", err)
			}
			for i := range rbacObjects {
				if err := env.Client().Create(ctx, &rbacObjects[i]); err != nil {
					t.Fatalf("create profiling Job RBAC object %s %q: %v", rbacObjects[i].GetKind(), rbacObjects[i].GetName(), err)
				}
			}
			if err := env.Client().Create(ctx, &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "hf-token-secret", Namespace: env.Namespace()},
				StringData: map[string]string{"HF_TOKEN": ""},
			}); err != nil {
				t.Fatalf("create profiler token secret: %v", err)
			}

			t.Log("Apply the DGDR input manifest through Kubernetes admission and select the local profiler image with its runtime compatibility")
			objects := golden.ApplyManifests(t, filepath.Join(scenarioDir, "input.yaml"), env.Client(), env.Namespace())
			if len(objects) != 1 || objects[0].GetKind() != "DynamoGraphDeploymentRequest" {
				t.Fatalf("scenario input contains %d objects, want one DynamoGraphDeploymentRequest", len(objects))
			}
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
			if err := env.Client().Get(ctx, client.ObjectKeyFromObject(&objects[0]), dgdr); err != nil {
				t.Fatalf("get admitted DGDR: %v", err)
			}
			dgdr.Spec.Image = profilerImage
			dgdr.Spec.RuntimeVersionOverride = "1.4.0"
			if err := env.Client().Update(ctx, dgdr); err != nil {
				t.Fatalf("select profiler image: %v", err)
			}

			t.Log("Start the production DGDR-to-terminal-workload controller chain")
			operatorConfig := clusterTestRestrictedConfig(env.Namespace())
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
			env.StartManager(func(mgr ctrl.Manager) error {
				setupOptions := SetupOptions{Config: operatorConfig, RuntimeConfig: runtimeConfig}
				if err := SetupDynamoGraphDeploymentRequest(mgr, DynamoGraphDeploymentRequestSetupOptions{
					SetupOptions:            setupOptions,
					OperatorImage:           operatorImage,
					OperatorImagePullPolicy: corev1.PullIfNotPresent,
				}); err != nil {
					return err
				}
				if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
					SetupOptions: setupOptions,
				}); err != nil {
					return err
				}
				return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{
					SetupOptions: setupOptions,
				})
			})

			t.Log("Wait for the real profiling Job Pod to complete")
			var completedDGDR nvidiacomv1beta1.DynamoGraphDeploymentRequest
			dynamotesting.Eventually(t, func() (bool, string) {
				err := env.Client().Get(ctx, client.ObjectKeyFromObject(dgdr), &completedDGDR)
				require.NoError(t, err)
				if completedDGDR.Status.ProfilingJobName == "" {
					require.NotEqual(t, nvidiacomv1beta1.DGDRPhaseFailed, completedDGDR.Status.Phase,
						"DGDR failed before creating a profiling Job: %v", completedDGDR.Status.Conditions)
					return false, fmt.Sprintf("DGDR phase is %s and has no profiling Job", completedDGDR.Status.Phase)
				}
				var job batchv1.Job
				err = env.Client().Get(ctx, types.NamespacedName{
					Name: completedDGDR.Status.ProfilingJobName, Namespace: env.Namespace(),
				}, &job)
				if err != nil {
					if apierrors.IsNotFound(err) {
						return false, fmt.Sprintf("profiling Job %q does not exist yet", completedDGDR.Status.ProfilingJobName)
					}
					require.NoError(t, err)
				}
				require.Zero(t, job.Status.Failed, "profiling Job failed: %v", job.Status.Conditions)
				if job.Status.Succeeded != 1 {
					return false, fmt.Sprintf("profiling Job has %d successful Pods", job.Status.Succeeded)
				}
				return true, "profiling Job completed"
			}, 20*time.Minute, time.Second, "DGDR did not record a completed profiling Job")

			t.Log("Verify profiling output was consumed and autoApply created a DGD")
			var dgd nvidiacomv1beta1.DynamoGraphDeployment
			dynamotesting.Eventually(t, func() (bool, string) {
				err := env.Client().Get(ctx, types.NamespacedName{
					Name: dgdr.Name + "-dgd", Namespace: env.Namespace(),
				}, &dgd)
				if err != nil {
					if apierrors.IsNotFound(err) {
						return false, "generated DGD does not exist yet"
					}
					require.NoError(t, err)
				}
				return true, "generated DGD exists"
			}, 2*time.Minute, time.Second, "DGDR did not create a DGD")
			if err := env.Client().Get(ctx, client.ObjectKeyFromObject(dgdr), &completedDGDR); err != nil {
				t.Fatalf("get completed DGDR: %v", err)
			}
			if completedDGDR.Status.ProfilingResults == nil || completedDGDR.Status.ProfilingResults.SelectedConfig == nil {
				t.Fatal("DGDR has no selected profiling configuration")
			}

			var liveDGD nvidiacomv1beta1.DynamoGraphDeployment
			require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(&dgd), &liveDGD))
			t.Log("Wait for the generated DGD and terminal workload manifests with the live worker hash")
			golden.EventuallyMatchManifestsWithVariables(
				t,
				env.Client(),
				env.Namespace(),
				filepath.Join(scenarioDir, "output.yaml"),
				workerHashGoldenVariables(t, &liveDGD),
			)
		})
	}
}

func TestClusterDynamoGraphDeploymentRequestV1beta2ProducesCandidates(t *testing.T) {
	inputs, err := filepath.Glob(filepath.Join("testdata/dgdr-v1beta2", "*", "input.yaml"))
	if err != nil {
		t.Fatalf("find v1beta2 DGDR cluster-test scenarios: %v", err)
	}
	if len(inputs) == 0 {
		t.Fatal("no v1beta2 DGDR cluster-test scenarios found")
	}
	plannerImage := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_PROFILER_IMAGE")
	operatorImage := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_OPERATOR_IMAGE")
	operatorVersion, err := clusterTestOperatorVersion()
	require.NoError(t, err)
	parsedOperatorVersion, err := semver.NewVersion(operatorVersion)
	require.NoError(t, err)
	dynamoVersion := fmt.Sprintf("%d.%d.%d", parsedOperatorVersion.Major(), parsedOperatorVersion.Minor(), parsedOperatorVersion.Patch())
	for _, input := range inputs {
		scenarioDir := filepath.Dir(input)
		t.Run(filepath.Base(scenarioDir), func(t *testing.T) {
			ctx := t.Context()

			t.Log("Create an isolated namespace in the explicitly unlocked cluster")
			env := clusterTestEnv.RunT(t)

			t.Log("Allow the AISimulate Job Pod to run while preventing unrelated ReplicaSet workloads")
			env.BlockReplicaSets()

			t.Log("Apply the native v1beta2 Request through production admission")
			objects := golden.ApplyManifests(t, filepath.Join(scenarioDir, "input.yaml"), env.Client(), env.Namespace())
			if len(objects) != 1 || objects[0].GetKind() != "DynamoGraphDeploymentRequest" {
				t.Fatalf("scenario input contains %d objects, want one DynamoGraphDeploymentRequest", len(objects))
			}
			request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}
			if err := env.Client().Get(ctx, client.ObjectKeyFromObject(&objects[0]), request); err != nil {
				t.Fatalf("get admitted v1beta2 DGDR: %v", err)
			}

			t.Log("Watch every Run transition before starting the production controllers")
			watchClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: newClusterTestScheme()})
			require.NoError(t, err)
			runWatch, err := watchClient.Watch(ctx, &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}, client.InNamespace(env.Namespace()))
			require.NoError(t, err)
			defer runWatch.Stop()

			t.Log("Provide the Run controller the discovered node shape represented by the explicit Request hardware")
			hardwareCache := gpu.NewGPUDiscoveryCache()
			hardwareCache.Set(nvidiacomv1beta1.GPUSKUTypeH200SXM, &gpu.GPUInfo{
				System:        nvidiacomv1beta1.GPUSKUTypeH200SXM,
				GPUsPerNode:   8,
				NodesWithGPUs: 1,
				TotalGPUs:     8,
			}, time.Hour)

			t.Log("Start the production Request, Run, AISimulate Job, and publisher chain")
			operatorConfig := clusterTestRestrictedConfig(env.Namespace())
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{}}
			env.StartManager(func(mgr ctrl.Manager) error {
				setupOptions := SetupOptions{Config: operatorConfig, RuntimeConfig: runtimeConfig}
				if err := SetupDynamoGraphDeploymentRequestV1beta2(mgr, setupOptions); err != nil {
					return err
				}
				return SetupDynamoGraphDeploymentRunV1beta2(mgr, DynamoGraphDeploymentRunSetupOptions{
					SetupOptions:            setupOptions,
					GPUDiscoveryCache:       hardwareCache,
					PlannerImage:            plannerImage,
					DynamoVersion:           dynamoVersion,
					OperatorImage:           operatorImage,
					OperatorImagePullPolicy: corev1.PullIfNotPresent,
				})
			})

			t.Log("Observe the Run until the real AISimulate and publisher containers finish")
			var completedRun nvidiacomv1beta2.DynamoGraphDeploymentRun
			deadline := time.NewTimer(10 * time.Minute)
			defer deadline.Stop()
			for completedRun.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseSucceeded {
				select {
				case event, ok := <-runWatch.ResultChan():
					require.True(t, ok, "Run watch closed before the execution completed")
					if event.Type == watch.Error {
						t.Fatalf("Run watch failed: %v", apierrors.FromObject(event.Object))
					}
					run, ok := event.Object.(*nvidiacomv1beta2.DynamoGraphDeploymentRun)
					require.Truef(t, ok, "Run watch returned %T", event.Object)
					if !metav1.IsControlledBy(run, request) {
						continue
					}
					require.NotEqual(t, nvidiacomv1beta2.DGDRRunPhaseFailed, run.Status.Phase,
						"Run failed: %v", run.Status.Conditions)
					completedRun = *run.DeepCopy()
				case <-deadline.C:
					t.Fatal("v1beta2 DGDR did not complete with a candidate")
				}
			}
			require.NotNil(t, completedRun.Status.Progress, "completed Run has no progress")
			require.NotNil(t, completedRun.Status.Progress.Candidates, "completed Run has no candidate progress")
			require.Equal(t, int32(1), completedRun.Status.Progress.Candidates.Published)
			require.Len(t, completedRun.Status.CandidateRefs, 1)

			t.Log("Verify the Job consumed the immutable input recorded in Run status")
			require.NotNil(t, completedRun.Status.JobRef, "completed Run has no Job reference")
			require.NotNil(t, completedRun.Status.AISimulate, "completed Run has no AISimulate status")
			require.NotNil(t, completedRun.Status.AISimulate.Input, "completed Run has no effective AISimulate input")
			key := types.NamespacedName{Namespace: env.Namespace(), Name: completedRun.Status.JobRef.Name}
			var executionConfig corev1.ConfigMap
			require.NoError(t, env.Client().Get(ctx, key, &executionConfig))
			require.True(t, metav1.IsControlledBy(&executionConfig, &completedRun), "execution ConfigMap is not controlled by its Run")
			var job batchv1.Job
			require.NoError(t, env.Client().Get(ctx, key, &job))
			require.True(t, metav1.IsControlledBy(&job, &completedRun), "AISimulate Job is not controlled by its Run")
			var inputVolume *corev1.Volume
			for i := range job.Spec.Template.Spec.Volumes {
				if job.Spec.Template.Spec.Volumes[i].Name == dgdrRunInputVolume {
					inputVolume = &job.Spec.Template.Spec.Volumes[i]
				}
			}
			require.NotNil(t, inputVolume, "AISimulate Job has no input volume")
			require.NotNil(t, inputVolume.ConfigMap, "AISimulate input volume is not a ConfigMap")
			require.Equal(t, executionConfig.Name, inputVolume.ConfigMap.Name)
			sweeper, found := dgdrRunContainer(job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
			require.True(t, found, "AISimulate Job has no sweeper container")
			require.Contains(t, sweeper.VolumeMounts, corev1.VolumeMount{
				Name: dgdrRunInputVolume, MountPath: dgdrRunInputMountPath, ReadOnly: true,
			})
			configInput, err := dgdrRunStatusInput(executionConfig.Data[dgdrRunInputKey])
			require.NoError(t, err)
			require.JSONEq(t, string(completedRun.Status.AISimulate.Input.Raw), string(configInput.Raw))

			t.Log("Verify the Candidate is a real API object controlled by the immutable Run")
			require.True(t, metav1.IsControlledBy(&completedRun, request), "Run is not controlled by its Request")
			var candidates nvidiacomv1beta2.DynamoGraphDeploymentCandidateList
			require.NoError(t, env.Client().List(ctx, &candidates, client.InNamespace(env.Namespace())))
			require.Len(t, candidates.Items, 1)
			require.True(t, metav1.IsControlledBy(&candidates.Items[0], &completedRun), "Candidate is not controlled by its Run")
			require.Equal(t, completedRun.Status.CandidateRefs[0].Name, candidates.Items[0].Name)
			require.NotEmpty(t, candidates.Items[0].Spec.Components, "Candidate has no materialized DGD components")

			t.Log("Match the published Candidate API contract against the reviewed golden")
			golden.EventuallyMatchManifests(t, env.Client(), env.Namespace(), filepath.Join(scenarioDir, "output.yaml"))
		})
	}
}

func getOnlyClusterDGD(t *testing.T, ctx context.Context, c client.Client, namespace string) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()

	var dgds nvidiacomv1beta1.DynamoGraphDeploymentList
	require.NoError(t, c.List(ctx, &dgds, client.InNamespace(namespace)))
	require.Len(t, dgds.Items, 1)
	return &dgds.Items[0]
}

func workerHashGoldenVariables(t *testing.T, dgd *nvidiacomv1beta1.DynamoGraphDeployment) golden.Variables {
	t.Helper()

	workerHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	return golden.Variables{"worker-hash": workerHash}
}
