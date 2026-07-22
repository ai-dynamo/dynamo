//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
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
	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/require"
	istioclientsetscheme "istio.io/client-go/pkg/clientset/versioned/scheme"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	networkingv1 "k8s.io/api/networking/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	lwsscheme "sigs.k8s.io/lws/client-go/clientset/versioned/scheme"
	vcbatchv1alpha1 "volcano.sh/apis/pkg/apis/batch/v1alpha1"
	volcanov1beta1 "volcano.sh/apis/pkg/apis/scheduling/v1beta1"
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

func newClusterTestScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(configv1alpha1.AddToScheme(scheme))
	utilruntime.Must(nvidiacomv1alpha1.AddToScheme(scheme))
	utilruntime.Must(nvidiacomv1beta1.AddToScheme(scheme))
	utilruntime.Must(grovev1alpha1.AddToScheme(scheme))
	utilruntime.Must(monitoringv1.AddToScheme(scheme))
	utilruntime.Must(istioclientsetscheme.AddToScheme(scheme))
	utilruntime.Must(lwsscheme.AddToScheme(scheme))
	utilruntime.Must(gaiev1.Install(scheme))
	utilruntime.Must(appsv1.AddToScheme(scheme))
	utilruntime.Must(admissionregistrationv1.AddToScheme(scheme))
	utilruntime.Must(autoscalingv2.AddToScheme(scheme))
	utilruntime.Must(batchv1.AddToScheme(scheme))
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(discoveryv1.AddToScheme(scheme))
	utilruntime.Must(networkingv1.AddToScheme(scheme))
	utilruntime.Must(rbacv1.AddToScheme(scheme))
	utilruntime.Must(resourcev1.AddToScheme(scheme))
	utilruntime.Must(apiextensionsv1.AddToScheme(scheme))
	utilruntime.Must(volcanov1beta1.AddToScheme(scheme))
	utilruntime.Must(vcbatchv1alpha1.AddToScheme(scheme))
	return scheme
}

func TestClusterDynamoComponentDeploymentManifests(t *testing.T) {
	clusterTestForEachScenario(t, "testdata/dcd", func(t *testing.T, scenarioDir string) {
		clusterTestRunManifestScenario(t, scenarioDir, func(mgr ctrl.Manager, setupOptions SetupOptions) error {
			return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: setupOptions})
		})
	})
}

func TestClusterDynamoGraphDeploymentManifests(t *testing.T) {
	clusterTestForEachScenario(t, "testdata/dgd", func(t *testing.T, scenarioDir string) {
		clusterTestRunManifestScenario(t, scenarioDir, func(mgr ctrl.Manager, setupOptions SetupOptions) error {
			if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: setupOptions}); err != nil {
				return err
			}
			return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: setupOptions})
		})
	})
}

func clusterTestForEachScenario(t *testing.T, root string, run func(*testing.T, string)) {
	t.Helper()
	inputs, err := filepath.Glob(filepath.Join(root, "*", "input.yaml"))
	if err != nil {
		t.Fatalf("find cluster-test scenarios under %q: %v", root, err)
	}
	if len(inputs) == 0 {
		t.Fatalf("no cluster-test scenarios found under %q", root)
	}
	for _, input := range inputs {
		scenarioDir := filepath.Dir(input)
		t.Run(filepath.Base(scenarioDir), func(t *testing.T) {
			run(t, scenarioDir)
		})
	}
}

func clusterTestRunManifestScenario(
	t *testing.T,
	scenarioDir string,
	setupControllers func(ctrl.Manager, SetupOptions) error,
) {
	t.Helper()
	env := clusterTestEnv.RunT(t)

	t.Log("Block Pods and ReplicaSets from actuating terminal workload manifests")
	env.BlockWorkloads()

	t.Log("Apply the scenario input manifests through Kubernetes admission")
	golden.ApplyManifests(t, filepath.Join(scenarioDir, "input.yaml"), env.Client(), env.Namespace())

	t.Log("Start the controllers for the scenario group")
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
	env.StartManager(func(mgr ctrl.Manager) error {
		return setupControllers(mgr, SetupOptions{Config: operatorConfig, RuntimeConfig: runtimeConfig})
	})

	t.Log("Match the complete generated manifest contract")
	golden.MatchManifests(t, env.Client(), env.Namespace(), filepath.Join(scenarioDir, "output.yaml"))
}

func TestClusterDynamoGraphDeploymentRequestProfilesAndCreatesWorkloadManifests(t *testing.T) {
	scenarioDir := filepath.Join("testdata", "dgdr", "profiler")
	ctx := t.Context()
	profilerImage := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_PROFILER_IMAGE")

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

	t.Log("Apply the DGDR input manifest through Kubernetes admission and select the local profiler image")
	objects := golden.ApplyManifests(t, filepath.Join(scenarioDir, "input.yaml"), env.Client(), env.Namespace())
	if len(objects) != 1 || objects[0].GetKind() != "DynamoGraphDeploymentRequest" {
		t.Fatalf("scenario input contains %d objects, want one DynamoGraphDeploymentRequest", len(objects))
	}
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
	if err := env.Client().Get(ctx, client.ObjectKeyFromObject(&objects[0]), dgdr); err != nil {
		t.Fatalf("get admitted DGDR: %v", err)
	}
	dgdr.Spec.Image = profilerImage
	if err := env.Client().Update(ctx, dgdr); err != nil {
		t.Fatalf("select profiler image: %v", err)
	}

	t.Log("Create the scale client and start the production DGDR-to-DGD controller chain with Grove enabled")
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
	scaleClient, err := env.ScaleClient()
	if err != nil {
		t.Fatalf("create scale client: %v", err)
	}
	env.StartManager(func(mgr ctrl.Manager) error {
		if err := SetupDynamoGraphDeploymentRequest(mgr, DynamoGraphDeploymentRequestSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: runtimeConfig,
			},
		}); err != nil {
			return err
		}
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: runtimeConfig,
			},
			ScaleClient: scaleClient,
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

	t.Log("Match the generated DGD and its terminal Grove manifest")
	golden.MatchManifests(t, env.Client(), env.Namespace(), filepath.Join(scenarioDir, "output.yaml"))
}

func clusterTestRestrictedConfig(namespace string) *configv1alpha1.OperatorConfiguration {
	config := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(config)
	config.Namespace.Restricted = namespace
	config.GPU.DiscoveryEnabled = ptr.To(false)
	return config
}
