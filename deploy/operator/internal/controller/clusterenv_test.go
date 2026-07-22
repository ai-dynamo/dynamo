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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/clusterenv"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	grovemock "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/mocks/grove"
	lwsmock "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/mocks/lws"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/go-logr/logr"
	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
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
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	lwsscheme "sigs.k8s.io/lws/client-go/clientset/versioned/scheme"
	vcbatchv1alpha1 "volcano.sh/apis/pkg/apis/batch/v1alpha1"
	volcanov1beta1 "volcano.sh/apis/pkg/apis/scheduling/v1beta1"
)

const clusterTestOperatorVersion = "1.3.0"

var clusterTestEnv = newClusterTestEnv()

type clusterTestControllerChain string

const (
	clusterTestDCDChain clusterTestControllerChain = "dcd"
	clusterTestDGDChain clusterTestControllerChain = "dgd"
)

func runClusterTestEnv(runner interface{ Run() int }) int {
	logf.SetLogger(logr.Discard())
	return clusterTestEnv.RunM(runner)
}

func newClusterTestEnv() *clusterenv.Env {
	operatorConfig := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(operatorConfig)
	operatorConfig.GPU.DiscoveryEnabled = ptr.To(false)
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
	additionalAdmission := lwsmock.Configurations()
	groveAdmission := grovemock.Configurations()
	additionalAdmission.Mutating = append(additionalAdmission.Mutating, groveAdmission.Mutating...)
	additionalAdmission.Validating = append(additionalAdmission.Validating, groveAdmission.Validating...)

	return clusterenv.New(clusterenv.Options{
		Scheme:                 newClusterTestScheme(),
		AdditionalAdmission:    additionalAdmission,
		WebhookProxyImage:      os.Getenv("DYNAMO_CLUSTERTEST_WEBHOOK_PROXY_IMAGE"),
		EventuallyTimeout:      2 * time.Minute,
		NamespaceDeleteTimeout: 5 * time.Minute,
		SetupWebhooks: func(mgr ctrl.Manager) error {
			if err := webhooksetup.Setup(mgr, webhooksetup.Options{
				Config:            operatorConfig,
				RuntimeConfig:     runtimeConfig,
				OperatorVersion:   clusterTestOperatorVersion,
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

func clusterTestRunManifestScenarios(t *testing.T, root string, chain clusterTestControllerChain) {
	t.Helper()
	clusterTestForEachScenario(t, root, func(t *testing.T, scenarioDir string) {
		clusterTestRunManifestScenario(t, scenarioDir, chain)
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

func clusterTestRunManifestScenario(t *testing.T, scenarioDir string, chain clusterTestControllerChain) {
	t.Helper()
	env := clusterTestEnv.RunT(t)

	t.Log("Block Pods and ReplicaSets from actuating terminal workload manifests")
	env.BlockWorkloads()

	t.Log("Apply the scenario input manifests through Kubernetes admission")
	golden.ApplyManifests(t, filepath.Join(scenarioDir, "input.yaml"), env.Client(), env.Namespace())

	t.Log("Start the controller chain selected by the scenario group")
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
	env.StartManager(func(mgr ctrl.Manager) error {
		return clusterTestSetupControllerChain(mgr, chain, operatorConfig, runtimeConfig)
	})

	t.Log("Match the complete generated manifest contract")
	golden.MatchManifests(t, env.Client(), env.Namespace(), filepath.Join(scenarioDir, "output.yaml"))
}

func clusterTestSetupControllerChain(
	mgr ctrl.Manager,
	chain clusterTestControllerChain,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
) error {
	setupOptions := SetupOptions{Config: operatorConfig, RuntimeConfig: runtimeConfig}
	switch chain {
	case clusterTestDCDChain:
		return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: setupOptions})
	case clusterTestDGDChain:
		if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: setupOptions}); err != nil {
			return err
		}
		return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: setupOptions})
	default:
		return fmt.Errorf("unknown cluster-test controller chain %q", chain)
	}
}

func clusterTestRestrictedConfig(namespace string) *configv1alpha1.OperatorConfiguration {
	config := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(config)
	config.Namespace.Restricted = namespace
	config.GPU.DiscoveryEnabled = ptr.To(false)
	return config
}
