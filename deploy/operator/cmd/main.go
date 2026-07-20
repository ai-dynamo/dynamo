/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
 */

package main

import (
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	k8sCache "k8s.io/client-go/tools/cache"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlcontroller "sigs.k8s.io/controller-runtime/pkg/controller"

	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsfilters "sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	lwsscheme "sigs.k8s.io/lws/client-go/clientset/versioned/scheme"
	volcanoscheme "volcano.sh/apis/pkg/client/clientset/versioned/scheme"

	semver "github.com/Masterminds/semver/v3"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	configvalidation "github.com/ai-dynamo/dynamo/deploy/operator/api/config/validation"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	internalcert "github.com/ai-dynamo/dynamo/deploy/operator/internal/cert"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/crdmigrator"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/namespace_scope"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/rbac"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secrets"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	webhookmutation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/mutation"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/validation"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	istioclientsetscheme "istio.io/client-go/pkg/clientset/versioned/scheme"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	//+kubebuilder:scaffold:imports
)

var (
	crdScheme    = k8sruntime.NewScheme()
	setupLog     = ctrl.Log.WithName("setup")
	configScheme = k8sruntime.NewScheme()
)

// LoadAndValidateOperatorConfig loads the operator configuration from a file,
// applies defaults via the scheme, and validates it.
func LoadAndValidateOperatorConfig(path string) (*configv1alpha1.OperatorConfiguration, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	codecFactory := serializer.NewCodecFactory(configScheme)
	cfg := &configv1alpha1.OperatorConfiguration{}
	if err := k8sruntime.DecodeInto(codecFactory.UniversalDecoder(), data, cfg); err != nil {
		return nil, fmt.Errorf("failed to decode config file %s: %w", path, err)
	}

	// Validate the configuration
	if errs := configvalidation.ValidateOperatorConfiguration(cfg); len(errs) > 0 {
		return nil, fmt.Errorf("config validation failed: %s", errs.ToAggregate().Error())
	}

	return cfg, nil
}

func createScalesGetter(mgr ctrl.Manager) (scale.ScalesGetter, error) {
	config := mgr.GetConfig()

	// Create kubernetes client for discovery
	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	// Create cached discovery client
	cachedDiscovery := memory.NewMemCacheClient(kubeClient.Discovery())

	// Create REST mapper
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)

	scalesGetter, err := scale.NewForConfig(
		config,
		restMapper,
		dynamic.LegacyAPIPathResolverFunc,
		scale.NewDiscoveryScaleKindResolver(cachedDiscovery),
	)
	if err != nil {
		return nil, err
	}

	return scalesGetter, nil
}

func initCRDSchemes() {
	utilruntime.Must(clientgoscheme.AddToScheme(crdScheme))

	utilruntime.Must(nvidiacomv1alpha1.AddToScheme(crdScheme))

	utilruntime.Must(nvidiacomv1beta1.AddToScheme(crdScheme))

	utilruntime.Must(lwsscheme.AddToScheme(crdScheme))

	utilruntime.Must(volcanoscheme.AddToScheme(crdScheme))

	utilruntime.Must(grovev1alpha1.AddToScheme(crdScheme))

	utilruntime.Must(apiextensionsv1.AddToScheme(crdScheme))

	utilruntime.Must(admissionregistrationv1.AddToScheme(crdScheme))

	utilruntime.Must(istioclientsetscheme.AddToScheme(crdScheme))

	utilruntime.Must(gaiev1.Install(crdScheme))
	//+kubebuilder:scaffold:scheme
}

func initConfigScheme() {
	utilruntime.Must(configv1alpha1.AddToScheme(configScheme))
}

// +kubebuilder:rbac:groups=authentication.k8s.io,resources=tokenreviews,verbs=create
// +kubebuilder:rbac:groups=authorization.k8s.io,resources=subjectaccessreviews,verbs=create
// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch;create;update

//nolint:gocyclo
func main() {
	initCRDSchemes()
	initConfigScheme()

	var configFile string
	var operatorVersion string
	var operatorImage string
	var operatorImagePullPolicy string
	flag.StringVar(&configFile, "config", "", "Path to operator configuration file (required)")
	flag.StringVar(&operatorVersion, "operator-version", "unknown",
		"Version of the operator (used in lease holder identity)")
	flag.StringVar(
		&operatorImage,
		"operator-image",
		"",
		"Operator image used to deliver version-matched helper binaries for DGD overrides",
	)
	flag.StringVar(&operatorImagePullPolicy, "operator-image-pull-policy", string(corev1.PullIfNotPresent),
		"Image pull policy for operator helper init containers")
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	if configFile == "" {
		setupLog.Error(nil, "--config flag is required")
		os.Exit(1)
	}
	// Load, default, and validate operator configuration
	operatorCfg, err := LoadAndValidateOperatorConfig(configFile)
	if err != nil {
		setupLog.Error(err, "failed to load operator configuration", "configFile", configFile)
		os.Exit(1)
	}
	setupLog.Info("Operator configuration loaded successfully", "configFile", configFile)

	// Validate and normalize operator version to semver
	if _, err := semver.NewVersion(operatorVersion); err != nil {
		setupLog.Error(err, "operator-version is not valid semver",
			"provided", operatorVersion, "error", err.Error())
		os.Exit(1)
	}
	setupLog.Info("Operator version configured", "version", operatorVersion)

	pullPolicy := corev1.PullPolicy(operatorImagePullPolicy)
	switch pullPolicy {
	case corev1.PullAlways, corev1.PullIfNotPresent, corev1.PullNever:
	default:
		setupLog.Error(nil, "operator-image-pull-policy is invalid", "provided", operatorImagePullPolicy)
		os.Exit(1)
	}

	// Initialize runtime config (will be populated after detection)
	runtimeConfig := &commonController.RuntimeConfig{}

	mainCtx := ctrl.SetupSignalHandler()

	// if the enable-http2 flag is false (the default), http/2 should be disabled
	// due to its vulnerabilities. More specifically, disabling http/2 will
	// prevent from being vulnerable to the HTTP/2 Stream Cancellation and
	// Rapid Reset CVEs. For more information see:
	// - https://github.com/advisories/GHSA-qppj-fm5r-hxr3
	// - https://github.com/advisories/GHSA-4374-p667-p6c8
	disableHTTP2 := func(c *tls.Config) {
		setupLog.Info("disabling http/2")
		c.NextProtos = []string{"http/1.1"}
	}

	tlsOpts := []func(*tls.Config){}
	if !operatorCfg.Security.EnableHTTP2 {
		tlsOpts = append(tlsOpts, disableHTTP2)
	}

	webhookServer := webhook.NewServer(webhook.Options{
		Host:    operatorCfg.Server.Webhook.Host,
		Port:    operatorCfg.Server.Webhook.Port,
		CertDir: operatorCfg.Server.Webhook.CertDir,
		TLSOpts: tlsOpts,
	})

	metricsBindAddr := fmt.Sprintf("%s:%d", operatorCfg.Server.Metrics.BindAddress, operatorCfg.Server.Metrics.Port)
	healthProbeAddr := fmt.Sprintf(
		"%s:%d", operatorCfg.Server.HealthProbe.BindAddress, operatorCfg.Server.HealthProbe.Port,
	)

	mgrOpts := ctrl.Options{
		Scheme: crdScheme,
		Metrics: metricsserver.Options{
			BindAddress:    metricsBindAddr,
			SecureServing:  ptr.Deref(operatorCfg.Server.Metrics.Secure, true),
			FilterProvider: metricsfilters.WithAuthenticationAndAuthorization,
			TLSOpts:        tlsOpts,
		},
		WebhookServer:           webhookServer,
		HealthProbeBindAddress:  healthProbeAddr,
		LeaderElection:          operatorCfg.LeaderElection.Enabled,
		LeaderElectionID:        operatorCfg.LeaderElection.ID,
		LeaderElectionNamespace: operatorCfg.LeaderElection.Namespace,
	}

	restrictedNamespace := operatorCfg.Namespace.Restricted
	isClusterWide := restrictedNamespace == ""
	if restrictedNamespace != "" {
		mgrOpts.Cache.DefaultNamespaces = map[string]cache.Config{
			restrictedNamespace: {},
		}
		// PodSnapshotContent is cluster-scoped, so DefaultNamespaces does not cover it.
		// Register it cluster-wide explicitly so the PodSnapshotReconciler can watch it.
		mgrOpts.Cache.ByObject = map[client.Object]cache.ByObject{
			&nvidiacomv1alpha1.PodSnapshotContent{}: {},
		}
		setupLog.Info("Restricted namespace configured, launching in restricted mode", "namespace", restrictedNamespace)

		banner := strings.Repeat("=", 80)
		setupLog.Error(nil, banner)
		setupLog.Error(nil, "DEVELOPMENT AND TESTING ONLY: Namespace-restricted mode is not supported for production.")
		setupLog.Error(nil, "The operator is running in namespace-restricted mode",
			"namespace", restrictedNamespace)
		setupLog.Error(nil, "Use cluster-wide mode for production deployments.")
		setupLog.Error(nil, banner)
	} else {
		setupLog.Info("No restricted namespace configured, launching in cluster-wide mode")
	}
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), mgrOpts)
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	// Initialize observability metrics
	setupLog.Info("Initializing observability metrics")
	observability.InitMetrics()

	// Set up webhook certificate management.
	// A direct (non-cached) client is needed because the manager's cache isn't started yet.
	directClient, err := client.New(mgr.GetConfig(), client.Options{Scheme: crdScheme})
	if err != nil {
		setupLog.Error(err, "unable to create direct client for cert management")
		os.Exit(1)
	}
	certMgr, err := internalcert.NewCertManager(directClient, &operatorCfg.Server.Webhook)
	if err != nil {
		setupLog.Error(err, "unable to create cert manager")
		os.Exit(1)
	}
	// Auto mode runs one synchronous certificate refresh with the direct client,
	// then registers the cert-controller with the not-yet-started manager.
	if err = certMgr.SetupAndRunOnce(mainCtx, mgr); err != nil {
		setupLog.Error(err, "failed to setup webhook certificate management")
		os.Exit(1)
	}

	// Initialize namespace scope mechanism
	var leaseManager *namespace_scope.LeaseManager
	var leaseWatcher *namespace_scope.LeaseWatcher

	if restrictedNamespace != "" {
		// Namespace-restricted mode: Create and maintain namespace scope marker lease
		setupLog.Info("Creating namespace scope marker lease manager",
			"namespace", restrictedNamespace,
			"leaseDuration", operatorCfg.Namespace.Scope.LeaseDuration.Duration,
			"renewInterval", operatorCfg.Namespace.Scope.LeaseRenewInterval.Duration)

		leaseManager, err = namespace_scope.NewLeaseManager(
			mgr.GetConfig(),
			restrictedNamespace,
			operatorVersion,
			operatorCfg.Namespace.Scope.LeaseDuration.Duration,
			operatorCfg.Namespace.Scope.LeaseRenewInterval.Duration,
		)
		if err != nil {
			setupLog.Error(err, "unable to create namespace scope marker lease manager")
			os.Exit(1)
		}

		// Start the lease manager
		if err = leaseManager.Start(mainCtx); err != nil {
			setupLog.Error(err, "unable to start namespace scope marker lease manager")
			os.Exit(1)
		}

		// Monitor for fatal lease errors
		// If lease renewal fails repeatedly, we must exit to prevent split-brain
		go func() {
			select {
			case err := <-leaseManager.Errors():
				setupLog.Error(err, "FATAL: Lease manager encountered unrecoverable error, shutting down to prevent split-brain")
				os.Exit(1)
			case <-mainCtx.Done():
				// Normal shutdown, error channel monitoring no longer needed
				return
			}
		}()

		// Ensure lease is released on shutdown
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := leaseManager.Stop(shutdownCtx); err != nil {
				setupLog.Error(err, "failed to stop lease manager cleanly")
			}
		}()

		setupLog.Info("Namespace scope marker lease manager started successfully")
	} else {
		// Cluster-wide mode: Watch for namespace scope marker leases
		setupLog.Info("Setting up namespace scope marker lease watcher for cluster-wide mode")

		leaseWatcher, err = namespace_scope.NewLeaseWatcher(mgr.GetConfig())
		if err != nil {
			setupLog.Error(err, "unable to create namespace scope marker lease watcher")
			os.Exit(1)
		}

		// Start the lease watcher
		if err = leaseWatcher.Start(mainCtx); err != nil {
			setupLog.Error(err, "unable to start namespace scope marker lease watcher")
			os.Exit(1)
		}

		setupLog.Info("Namespace scope marker lease watcher started successfully")

		// Pass leaseWatcher to runtime config for namespace exclusion filtering
		runtimeConfig.ExcludedNamespaces = leaseWatcher
	}

	// Register after ExcludedNamespaces is set so cluster-wide metrics skip restricted namespaces.
	setupLog.Info("Registering resource counter")
	if err := mgr.Add(observability.NewResourceCounter(
		mgr.GetClient(),
		runtimeConfig.ExcludedNamespaces,
	)); err != nil {
		setupLog.Error(err, "unable to register resource counter")
		os.Exit(1)
	}

	gates, err := features.New(mainCtx, mgr, operatorCfg)
	if err != nil {
		setupLog.Error(err, "unable to resolve operator feature gates")
		os.Exit(1)
	}
	runtimeConfig.Gate = gates

	dockerSecretRetriever := secrets.NewDockerSecretIndexer(mgr.GetAPIReader(), restrictedNamespace)
	// refresh whenever a secret is created/deleted/updated
	// Set up informer
	var factory informers.SharedInformerFactory
	if restrictedNamespace == "" {
		factory = informers.NewSharedInformerFactory(kubernetes.NewForConfigOrDie(mgr.GetConfig()), time.Hour*24)
	} else {
		factory = informers.NewSharedInformerFactoryWithOptions(
			kubernetes.NewForConfigOrDie(mgr.GetConfig()),
			time.Hour*24,
			informers.WithNamespace(restrictedNamespace),
		)
	}
	secretInformer := factory.Core().V1().Secrets().Informer()
	// Start the informer factory
	go factory.Start(mainCtx.Done())
	// Wait for the initial sync
	if !k8sCache.WaitForCacheSync(mainCtx.Done(), secretInformer.HasSynced) {
		setupLog.Error(nil, "Failed to sync informer cache")
		os.Exit(1)
	}
	setupLog.Info("Secret informer cache synced and ready")
	_, err = secretInformer.AddEventHandler(k8sCache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			secret := obj.(*corev1.Secret)
			if secret.Type == corev1.SecretTypeDockerConfigJson {
				setupLog.Info("refreshing docker secrets index after secret creation...")
				err := dockerSecretRetriever.RefreshIndex(context.Background())
				if err != nil {
					setupLog.Error(err, "unable to refresh docker secrets index after secret creation")
				} else {
					setupLog.Info("docker secrets index refreshed after secret creation")
				}
			}
		},
		UpdateFunc: func(old, new interface{}) {
			newSecret := new.(*corev1.Secret)
			if newSecret.Type == corev1.SecretTypeDockerConfigJson {
				setupLog.Info("refreshing docker secrets index after secret update...")
				err := dockerSecretRetriever.RefreshIndex(context.Background())
				if err != nil {
					setupLog.Error(err, "unable to refresh docker secrets index after secret update")
				} else {
					setupLog.Info("docker secrets index refreshed after secret update")
				}
			}
		},
		DeleteFunc: func(obj interface{}) {
			secret := obj.(*corev1.Secret)
			if secret.Type == corev1.SecretTypeDockerConfigJson {
				setupLog.Info("refreshing docker secrets index after secret deletion...")
				err := dockerSecretRetriever.RefreshIndex(context.Background())
				if err != nil {
					setupLog.Error(err, "unable to refresh docker secrets index after secret deletion")
				} else {
					setupLog.Info("docker secrets index refreshed after secret deletion")
				}
			}
		},
	})
	if err != nil {
		setupLog.Error(err, "unable to add event handler to secret informer")
		os.Exit(1)
	}
	if err := dockerSecretRetriever.RefreshIndex(mainCtx); err != nil {
		setupLog.Error(err, "initial docker secrets index refresh completed with errors; continuing startup")
	} else {
		setupLog.Info("initial docker secrets index refreshed")
	}
	// launch a goroutine to refresh the docker secret indexer in any case every minute
	go func() {
		ticker := time.NewTicker(60 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-mainCtx.Done():
				return
			case <-ticker.C:
				if err := dockerSecretRetriever.RefreshIndex(mainCtx); err != nil {
					setupLog.Error(err, "failed to refresh docker secrets index")
				}
			}
		}
	}()

	sshKeyManager := secret.NewSSHKeyManager(mgr.GetClient(), operatorCfg.MPI)

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	// Register controllers synchronously before mgr.Start().
	// Controllers don't depend on TLS certificates.
	if err := registerControllers(
		mgr, operatorCfg, runtimeConfig,
		dockerSecretRetriever, sshKeyManager,
		operatorImage, pullPolicy,
	); err != nil {
		setupLog.Error(err, "failed to register controllers")
		os.Exit(1)
	}

	if err := registerWebhookHandlers(mgr, operatorCfg, runtimeConfig, operatorVersion, gates); err != nil {
		setupLog.Error(err, "failed to register webhooks")
		os.Exit(1)
	}

	// CertManager.SetupAndRunOnce has already bootstrapped auto-mode TLS secrets.
	// Auto mode patches admission and, for cluster-wide operators, conversion CAs.
	// Manual mode patches only cluster-wide conversion CAs; admission stays out-of-band.
	caInjector, err := internalcert.NewCABundleInjector(directClient, operatorCfg)
	if err != nil {
		setupLog.Error(err, "unable to create CA bundle injector")
		os.Exit(1)
	}
	if operatorCfg.Server.Webhook.CertProvisionMode == configv1alpha1.CertProvisionModeAuto {
		if isClusterWide {
			err = caInjector.InjectAll(mainCtx)
		} else {
			err = caInjector.InjectAdmission(mainCtx)
		}
		if err != nil {
			setupLog.Error(err, "failed to inject CA bundles into webhook configurations")
			os.Exit(1)
		}
	} else if isClusterWide {
		// Manual mode gets webhook CA material out-of-band. Missing ca.crt
		// blocks startup instead of running with unauthenticated conversion.
		if err := caInjector.InjectCRDConversionCA(mainCtx); err != nil {
			setupLog.Error(err, "failed to inject CRD conversion CA bundle")
			os.Exit(1)
		}
	}

	// mgr.Start reads tls.crt and tls.key from the projected Secret volume
	// synchronously. Secret API updates are not enough because kubelet projects
	// them into already-running pods asynchronously.
	if err := certMgr.WaitForMountedCertificate(mainCtx); err != nil {
		setupLog.Error(err, "failed waiting for mounted webhook TLS certificate")
		os.Exit(1)
	}

	// Kubernetes propagates webhook configuration asynchronously, especially
	// with HA apiservers. A missing or stale CA must fail closed during manager
	// cache startup rather than allowing the operator to run without conversion
	// or admission.
	setupLog.Info("starting manager")
	if err := mgr.Start(mainCtx); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func registerControllers(
	mgr ctrl.Manager,
	operatorCfg *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commonController.RuntimeConfig,
	dockerSecretRetriever *secrets.DockerSecretIndexer,
	sshKeyManager *secret.SSHKeyManager,
	operatorImage string,
	operatorPullPolicy corev1.PullPolicy,
) error {
	if operatorCfg.Namespace.Restricted == "" {
		migrator := &crdmigrator.CRDMigrator{
			Client:    mgr.GetClient(),
			APIReader: mgr.GetAPIReader(),
			Config: map[client.Object]crdmigrator.ByObjectConfig{
				&nvidiacomv1beta1.DynamoComponentDeployment{}: {
					UseCache:                            false,
					UseStatusForStorageVersionMigration: true,
				},
				&nvidiacomv1beta1.DynamoGraphDeployment{}: {
					UseCache:                            false,
					UseStatusForStorageVersionMigration: true,
				},
				&nvidiacomv1beta1.DynamoGraphDeploymentRequest{}: {
					UseCache:                            false,
					UseStatusForStorageVersionMigration: true,
				},
				&nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}: {
					UseCache:                            false,
					UseStatusForStorageVersionMigration: true,
				},
			},
		}
		if err := migrator.SetupWithManager(mgr, ctrlcontroller.Options{MaxConcurrentReconciles: 1}); err != nil {
			return fmt.Errorf("unable to create CRD migrator controller: %w", err)
		}
	}

	setupOptions := controller.SetupOptions{
		Config:        operatorCfg,
		RuntimeConfig: runtimeConfig,
	}

	if err := controller.SetupDynamoComponentDeployment(mgr, controller.DynamoComponentDeploymentSetupOptions{
		SetupOptions:          setupOptions,
		DockerSecretRetriever: dockerSecretRetriever,
	}); err != nil {
		return err
	}

	scaleClient, err := createScalesGetter(mgr)
	if err != nil {
		return fmt.Errorf("unable to create scale client: %w", err)
	}

	rbacManager := rbac.NewManager(mgr.GetClient())

	if err := controller.SetupDynamoGraphDeployment(mgr, controller.DynamoGraphDeploymentSetupOptions{
		SetupOptions:          setupOptions,
		DockerSecretRetriever: dockerSecretRetriever,
		ScaleClient:           scaleClient,
		RBACManager:           rbacManager,
		SSHKeyManager:         sshKeyManager,
	}); err != nil {
		return err
	}
	if err := controller.SetupDynamoGraphDeploymentScalingAdapter(mgr, setupOptions); err != nil {
		return err
	}
	if err := controller.SetupDynamoGraphDeploymentRequest(mgr, controller.DynamoGraphDeploymentRequestSetupOptions{
		SetupOptions:            setupOptions,
		RBACManager:             rbacManager,
		OperatorImage:           operatorImage,
		OperatorImagePullPolicy: operatorPullPolicy,
	}); err != nil {
		return err
	}
	if err := controller.SetupDynamoModel(mgr, controller.DynamoModelSetupOptions{
		SetupOptions: setupOptions,
	}); err != nil {
		return err
	}
	if err := controller.SetupDynamoCheckpoint(mgr, setupOptions); err != nil {
		return err
	}
	if err := controller.SetupPodSnapshot(mgr, setupOptions); err != nil {
		return err
	}

	if runtimeConfig.Gate.Enabled(features.Grove) {
		if err := controller.SetupFailoverCascade(mgr); err != nil {
			return err
		}
	}
	if err := controller.SetupTopologyLabel(mgr, setupOptions); err != nil {
		return err
	}

	setupLog.Info("Controllers registered successfully")
	return nil
}

func registerWebhookHandlers(
	mgr ctrl.Manager,
	operatorCfg *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commonController.RuntimeConfig,
	operatorVersion string,
	gate features.Gate,
) error {
	isClusterWide := operatorCfg.Namespace.Restricted == ""
	if isClusterWide {
		setupLog.Info("Configuring webhooks with lease-based namespace exclusion for cluster-wide mode")
		internalwebhook.SetExcludedNamespaces(runtimeConfig.ExcludedNamespaces)
	} else {
		setupLog.Info("Configuring webhooks for namespace-restricted mode (no lease checking)",
			"restrictedNamespace", operatorCfg.Namespace.Restricted)
		internalwebhook.SetExcludedNamespaces(nil)
	}

	var operatorPrincipal string
	if sa, ns := os.Getenv("POD_SERVICE_ACCOUNT"), os.Getenv("POD_NAMESPACE"); sa != "" && ns != "" {
		operatorPrincipal = fmt.Sprintf("system:serviceaccount:%s:%s", ns, sa)
		setupLog.Info("Detected operator principal from downward API", "principal", operatorPrincipal)
	} else {
		setupLog.Info("POD_SERVICE_ACCOUNT/POD_NAMESPACE not set; operator SA self-identification disabled")
	}

	// Temporary internal gate for GMS + Snapshot.
	if gate.Enabled(features.GMSSnapshot) {
		setupLog.Info(
			"INTERNAL OVERRIDE: GMS + Snapshot admission rule disabled via env var; do NOT enable in production",
			"envVar", features.GMSSnapshotEnvVar,
		)
	}

	setupLog.Info("Registering validation webhooks")

	dcdHandler := webhookvalidation.NewDynamoComponentDeploymentHandler()
	if err := dcdHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoComponentDeployment webhook: %w", err)
	}

	dgdHandler := webhookvalidation.NewDynamoGraphDeploymentHandler(mgr, operatorPrincipal)
	if err := dgdHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeployment webhook: %w", err)
	}

	dckptHandler := webhookvalidation.NewDynamoCheckpointHandler()
	if err := dckptHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoCheckpoint webhook: %w", err)
	}

	dmHandler := webhookvalidation.NewDynamoModelHandler()
	if err := dmHandler.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoModel webhook: %w", err)
	}

	dgdrHandler := webhookvalidation.NewDynamoGraphDeploymentRequestHandler()
	if err := dgdrHandler.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentRequest webhook: %w", err)
	}

	if isClusterWide {
		if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}).
			Complete(); err != nil {
			return fmt.Errorf("unable to register DynamoGraphDeploymentRequest conversion webhook: %w", err)
		}

		if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeployment{}).
			Complete(); err != nil {
			return fmt.Errorf("unable to register DynamoGraphDeployment conversion webhook: %w", err)
		}

		if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoComponentDeployment{}).
			Complete(); err != nil {
			return fmt.Errorf("unable to register DynamoComponentDeployment conversion webhook: %w", err)
		}

		if err := ctrl.NewWebhookManagedBy(mgr, &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}).
			Complete(); err != nil {
			return fmt.Errorf("unable to register DynamoGraphDeploymentScalingAdapter conversion webhook: %w", err)
		}
	}

	setupLog.Info("Registering defaulting webhooks")

	dcdDefaulter := webhookdefaulting.NewDCDDefaulter()
	if err := dcdDefaulter.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoComponentDeployment defaulting webhook: %w", err)
	}

	dgdDefaulter := webhookdefaulting.NewDGDDefaulter(operatorVersion)
	if err := dgdDefaulter.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeployment defaulting webhook: %w", err)
	}

	dgdrDefaulter := webhookdefaulting.NewDGDRDefaulter(operatorVersion)
	if err := dgdrDefaulter.RegisterWithManager(mgr); err != nil {
		return fmt.Errorf("unable to register DynamoGraphDeploymentRequest defaulting webhook: %w", err)
	}

	setupLog.Info("Registering mutation webhooks")

	podCheckpointRestoreMutator := webhookmutation.NewPodCheckpointRestoreMutator(mgr.GetClient(), operatorCfg)
	if err := podCheckpointRestoreMutator.RegisterWithManager(mgr, gate); err != nil {
		return fmt.Errorf("unable to register Pod checkpoint restore mutating webhook: %w", err)
	}

	setupLog.Info("Webhooks registered successfully")
	return nil
}
