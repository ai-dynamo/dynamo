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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	dynamotesting "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/clusterenv"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	grovemock "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/mocks/grove"
	lwsmock "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/mocks/lws"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorchart"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

var clusterTestEnv *clusterenv.Env

func TestMain(m *testing.M) {
	logf.SetLogger(logr.Discard())
	operatorVersion, err := clusterTestOperatorVersion()
	if err != nil {
		fmt.Fprintf(os.Stderr, "derive cluster-test operator version: %v\n", err)
		os.Exit(1)
	}
	operatorConfig := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(operatorConfig)
	operatorConfig.GPU.DiscoveryEnabled = ptr.To(false)
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
	additionalAdmission := lwsmock.Configurations()
	groveAdmission := grovemock.Configurations()
	additionalAdmission.Mutating = append(additionalAdmission.Mutating, groveAdmission.Mutating...)
	additionalAdmission.Validating = append(additionalAdmission.Validating, groveAdmission.Validating...)

	clusterTestEnv = clusterenv.New(clusterenv.Options{
		Scheme:                 newClusterTestScheme(),
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
	os.Exit(clusterTestEnv.RunM(m))
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

			t.Log("Wait for the complete generated manifest contract")
			golden.EventuallyMatchManifests(t, env.Client(), env.Namespace(), filepath.Join(scenarioDir, "output.yaml"))

			t.Log("Verify every rendered worker hash and namespace suffix binds to the live DGD hash")
			assertClusterDGDWorkerHashBindings(t, t.Context(), env.Client(), getOnlyClusterDGD(t, t.Context(), env.Client(), env.Namespace()))
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

			t.Log("Create the scale client and start the production DGDR-to-terminal-workload controller chain")
			operatorConfig := clusterTestRestrictedConfig(env.Namespace())
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
			scaleClient, err := env.ScaleClient()
			if err != nil {
				t.Fatalf("create scale client: %v", err)
			}
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
					ScaleClient:  scaleClient,
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

			t.Log("Wait for the generated DGD and its terminal workload manifests")
			golden.EventuallyMatchManifests(t, env.Client(), env.Namespace(), filepath.Join(scenarioDir, "output.yaml"))

			t.Log("Verify every rendered worker hash and namespace suffix binds to the live DGD hash")
			var liveDGD nvidiacomv1beta1.DynamoGraphDeployment
			require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(&dgd), &liveDGD))
			assertClusterDGDWorkerHashBindings(t, ctx, env.Client(), &liveDGD)
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

func assertClusterDGDWorkerHashBindings(
	t *testing.T,
	ctx context.Context,
	c client.Client,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	t.Helper()

	want, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)

	var dcds nvidiacomv1beta1.DynamoComponentDeploymentList
	require.NoError(t, c.List(ctx, &dcds, client.InNamespace(dgd.Namespace)))
	for i := range dcds.Items {
		dcd := &dcds.Items[i]
		if dcd.GetParentGraphDeploymentName() != dgd.Name || !dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
			continue
		}
		require.Equal(t, want, dcd.Labels[consts.KubeLabelDynamoWorkerHash], "worker DCD %q metadata hash", dcd.Name)
		require.Equal(t, want, dynamo.GetPodTemplateLabels(&dcd.Spec.DynamoComponentDeploymentSharedSpec)[consts.KubeLabelDynamoWorkerHash], "worker DCD %q pod-template hash", dcd.Name)
	}

	workerPodTemplates := 0
	var deployments appsv1.DeploymentList
	require.NoError(t, c.List(ctx, &deployments, client.InNamespace(dgd.Namespace)))
	for i := range deployments.Items {
		deployment := &deployments.Items[i]
		if deployment.Spec.Template.Labels[consts.KubeLabelDynamoGraphDeploymentName] != dgd.Name {
			continue
		}
		if assertClusterWorkerPodTemplateHashBinding(t, "Deployment "+deployment.Name, &deployment.Spec.Template, want) {
			workerPodTemplates++
		}
	}

	var leaderWorkerSets lwsv1.LeaderWorkerSetList
	require.NoError(t, c.List(ctx, &leaderWorkerSets, client.InNamespace(dgd.Namespace)))
	for i := range leaderWorkerSets.Items {
		leaderWorkerSet := &leaderWorkerSets.Items[i]
		leaderTemplate := leaderWorkerSet.Spec.LeaderWorkerTemplate.LeaderTemplate
		if leaderTemplate != nil && leaderTemplate.Labels[consts.KubeLabelDynamoGraphDeploymentName] == dgd.Name {
			if assertClusterWorkerPodTemplateHashBinding(t, "LeaderWorkerSet "+leaderWorkerSet.Name+" leader template", leaderTemplate, want) {
				workerPodTemplates++
			}
		}
		workerTemplate := &leaderWorkerSet.Spec.LeaderWorkerTemplate.WorkerTemplate
		if workerTemplate.Labels[consts.KubeLabelDynamoGraphDeploymentName] == dgd.Name {
			if assertClusterWorkerPodTemplateHashBinding(t, "LeaderWorkerSet "+leaderWorkerSet.Name+" worker template", workerTemplate, want) {
				workerPodTemplates++
			}
		}
	}

	pcs := &grovev1alpha1.PodCliqueSet{}
	err = c.Get(ctx, types.NamespacedName{Name: dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components), Namespace: dgd.Namespace}, pcs)
	if err != nil && !apierrors.IsNotFound(err) {
		require.NoError(t, err)
	}
	if err == nil {
		for _, clique := range pcs.Spec.Template.Cliques {
			if clique == nil || clique.Labels[consts.KubeLabelDynamoGraphDeploymentName] != dgd.Name {
				continue
			}
			podTemplate := &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: clique.Labels},
				Spec:       clique.Spec.PodSpec,
			}
			if assertClusterWorkerPodTemplateHashBinding(t, "PodCliqueSet "+pcs.Name+" clique "+clique.Name, podTemplate, want) {
				workerPodTemplates++
			}
		}
	}

	require.Positive(t, workerPodTemplates, "DGD %q has no rendered worker pod templates", dgd.Name)
}

func assertClusterWorkerPodTemplateHashBinding(t *testing.T, source string, podTemplate *corev1.PodTemplateSpec, want string) bool {
	t.Helper()

	if !dynamo.IsWorkerComponent(podTemplate.Labels[consts.KubeLabelDynamoComponentType]) {
		return false
	}
	require.Equal(t, want, podTemplate.Labels[consts.KubeLabelDynamoWorkerHash], "%s worker hash", source)

	for i := range podTemplate.Spec.Containers {
		container := &podTemplate.Spec.Containers[i]
		if container.Name != consts.MainContainerName {
			continue
		}
		for _, env := range container.Env {
			if env.Name == consts.DynamoNamespaceWorkerSuffixEnvVar {
				require.Equal(t, want, env.Value, "%s worker namespace suffix", source)
				return true
			}
		}
		require.Failf(t, "%s main container has no worker namespace suffix", source, "%s is missing %s", source, consts.DynamoNamespaceWorkerSuffixEnvVar)
	}
	require.Failf(t, "%s has no main container", source, "%s is missing container %q", source, consts.MainContainerName)
	return false
}
