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
	"time"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.

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

	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	lwsscheme "sigs.k8s.io/lws/client-go/clientset/versioned/scheme"
	volcanoscheme "volcano.sh/apis/pkg/client/clientset/versioned/scheme"

	semver "github.com/Masterminds/semver/v3"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	configvalidation "github.com/ai-dynamo/dynamo/deploy/operator/api/config/validation"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/namespace_scope"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/rbac"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secrets"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
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

	utilruntime.Must(istioclientsetscheme.AddToScheme(crdScheme))

	utilruntime.Must(gaiev1.Install(crdScheme))
	//+kubebuilder:scaffold:scheme
}

func initConfigScheme() {
	utilruntime.Must(configv1alpha1.AddToScheme(configScheme))
}

//nolint:gocyclo
func main() {
	initCRDSchemes()
	initConfigScheme()

	var configFile string
	var operatorVersion string
	flag.StringVar(&configFile, "config", "", "Path to operator configuration file (required)")
	flag.StringVar(&operatorVersion, "operator-version", "unknown",
		"Version of the operator (used in lease holder identity)")
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
			BindAddress:   metricsBindAddr,
			SecureServing: operatorCfg.Server.Metrics.Secure,
			TLSOpts:       tlsOpts,
		},
		WebhookServer:           webhookServer,
		HealthProbeBindAddress:  healthProbeAddr,
		LeaderElection:          operatorCfg.LeaderElection.Enabled,
		LeaderElectionID:        operatorCfg.LeaderElection.ID,
		LeaderElectionNamespace: operatorCfg.LeaderElection.Namespace,
	}

	restrictedNamespace := operatorCfg.Namespace.Restricted
	if restrictedNamespace != "" {
		mgrOpts.Cache.DefaultNamespaces = map[string]cache.Config{
			restrictedNamespace: {},
		}
		setupLog.Info("Restricted namespace configured, launching in restricted mode", "namespace", restrictedNamespace)
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

	// Start resource counter background goroutine (after ExcludedNamespaces is set)
	setupLog.Info("Starting resource counter")
	go observability.StartResourceCounter(mainCtx, mgr.GetClient(), runtimeConfig.ExcludedNamespaces)

	// Detect orchestrators availability using discovery client.
	// Config overrides (*bool) take precedence over auto-detection:
	//   nil   = auto-detect (backward compatible default)
	//   false = forcibly disabled regardless of API availability
	//   true  = forcibly enabled; hard exit if API is not available (misconfiguration)
	setupLog.Info("Detecting Grove availability...")
	groveDetected := commonController.DetectGroveAvailability(mainCtx, mgr)
	switch {
	case operatorCfg.Orchestrators.Grove.Enabled == nil:
		runtimeConfig.GroveEnabled = groveDetected
	case *operatorCfg.Orchestrators.Grove.Enabled:
		if !groveDetected {
			setupLog.Error(nil, "Grove is explicitly enabled in config but the Grove API group was not detected in the cluster")
			os.Exit(1)
		}
		runtimeConfig.GroveEnabled = true
	default:
		setupLog.Info("Grove is explicitly disabled via config override")
		runtimeConfig.GroveEnabled = false
	}

	setupLog.Info("Detecting LWS availability...")
	lwsDetected := commonController.DetectLWSAvailability(mainCtx, mgr)
	setupLog.Info("Detecting Volcano availability...")
	volcanoDetected := commonController.DetectVolcanoAvailability(mainCtx, mgr)
	// LWS for multinode deployment usage depends on both LWS and Volcano availability
	switch {
	case operatorCfg.Orchestrators.LWS.Enabled == nil:
		runtimeConfig.LWSEnabled = lwsDetected && volcanoDetected
	case *operatorCfg.Orchestrators.LWS.Enabled:
		if !lwsDetected {
			setupLog.Error(nil, "LWS is explicitly enabled in config but the LWS API group was not detected in the cluster")
			os.Exit(1)
		}
		if !volcanoDetected {
			setupLog.Error(nil, "LWS is explicitly enabled in config but the Volcano API group was not detected in the cluster")
			os.Exit(1)
		}
		runtimeConfig.LWSEnabled = true
	default:
		setupLog.Info("LWS is explicitly disabled via config override")
		runtimeConfig.LWSEnabled = false
	}

	// Detect Kai-scheduler availability using discovery client
	setupLog.Info("Detecting Kai-scheduler availability...")
	kaiSchedulerDetected := commonController.DetectKaiSchedulerAvailability(mainCtx, mgr)
	switch {
	case operatorCfg.Orchestrators.KaiScheduler.Enabled == nil:
		runtimeConfig.KaiSchedulerEnabled = kaiSchedulerDetected
	case *operatorCfg.Orchestrators.KaiScheduler.Enabled:
		if !kaiSchedulerDetected {
			setupLog.Error(nil,
				"Kai-scheduler is explicitly enabled in config but the scheduling.run.ai API group was not detected in the cluster",
			)
			os.Exit(1)
		}
		runtimeConfig.KaiSchedulerEnabled = true
	default:
		setupLog.Info("Kai-scheduler is explicitly disabled via config override")
		runtimeConfig.KaiSchedulerEnabled = false
	}

	setupLog.Info("Detected orchestrators availability",
		"grove", runtimeConfig.GroveEnabled,
		"lws", runtimeConfig.LWSEnabled,
		"volcano", volcanoDetected,
		"kai-scheduler", runtimeConfig.KaiSchedulerEnabled,
	)

	dockerSecretRetriever := secrets.NewDockerSecretIndexer(mgr.GetClient())
	// refresh whenever a secret is created/deleted/updated
	// Set up informer
	var factory informers.SharedInformerFactory
	if restrictedNamespace == "" {
		factory = informers.NewSharedInformerFactory(kubernetes.NewForConfigOrDie(mgr.GetConfig()), time.Hour*24)
	} else {
		factory = informers.NewFilteredSharedInformerFactory(
			kubernetes.NewForConfigOrDie(mgr.GetConfig()),
			time.Hour*24,
			restrictedNamespace,
			nil,
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
	// launch a goroutine to refresh the docker secret indexer in any case every minute
	go func() {
		// Initial refresh
		if err := dockerSecretRetriever.RefreshIndex(context.Background()); err != nil {
			setupLog.Error(err, "initial docker secrets index refresh failed")
		}
		ticker := time.NewTicker(60 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-mainCtx.Done():
				return
			case <-ticker.C:
				setupLog.Info("refreshing docker secrets index...")
				if err := dockerSecretRetriever.RefreshIndex(mainCtx); err != nil {
					setupLog.Error(err, "unable to refresh docker secrets index")
				}
				setupLog.Info("docker secrets index refreshed")
			}
		}
	}()

	// Create MPI SSH SecretReplicator for cross-namespace secret replication
	mpiSecretReplicator := secret.NewSecretReplicator(
		mgr.GetClient(),
		operatorCfg.MPI.SSHSecretNamespace,
		operatorCfg.MPI.SSHSecretName,
	)

	if err = (&controller.DynamoComponentDeploymentReconciler{
		Client:                mgr.GetClient(),
		Recorder:              mgr.GetEventRecorderFor("dynamocomponentdeployment"),
		Config:                operatorCfg,
		RuntimeConfig:         runtimeConfig,
		DockerSecretRetriever: dockerSecretRetriever,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoComponentDeployment")
		os.Exit(1)
	}
	// Create scale client for Grove resource scaling
	scaleClient, err := createScalesGetter(mgr)
	if err != nil {
		setupLog.Error(err, "unable to create scale client")
		os.Exit(1)
	}

	// Initialize RBAC manager for cross-namespace resource management
	rbacManager := rbac.NewManager(mgr.GetClient())

	if err = (&controller.DynamoGraphDeploymentReconciler{
		Client:                mgr.GetClient(),
		Recorder:              mgr.GetEventRecorderFor("dynamographdeployment"),
		Config:                operatorCfg,
		RuntimeConfig:         runtimeConfig,
		DockerSecretRetriever: dockerSecretRetriever,
		ScaleClient:           scaleClient,
		MPISecretReplicator:   mpiSecretReplicator,
		RBACManager:           rbacManager,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoGraphDeployment")
		os.Exit(1)
	}

	if err = (&controller.DynamoGraphDeploymentScalingAdapterReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		Recorder:      mgr.GetEventRecorderFor("dgdscalingadapter"),
		Config:        operatorCfg,
		RuntimeConfig: runtimeConfig,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DGDScalingAdapter")
		os.Exit(1)
	}

	if err = (&controller.DynamoGraphDeploymentRequestReconciler{
		Client:        mgr.GetClient(),
		Recorder:      mgr.GetEventRecorderFor("dynamographdeploymentrequest"),
		Config:        operatorCfg,
		RuntimeConfig: runtimeConfig,
		RBACManager:   rbacManager,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoGraphDeploymentRequest")
		os.Exit(1)
	}

	if err = (&controller.DynamoModelReconciler{
		Client:         mgr.GetClient(),
		Recorder:       mgr.GetEventRecorderFor("dynamomodel"),
		EndpointClient: modelendpoint.NewClient(),
		Config:         operatorCfg,
		RuntimeConfig:  runtimeConfig,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoModel")
		os.Exit(1)
	}

	if err = (&controller.CheckpointReconciler{
		Client:        mgr.GetClient(),
		Config:        operatorCfg,
		RuntimeConfig: runtimeConfig,
		Recorder:      mgr.GetEventRecorderFor("checkpoint"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoCheckpoint")
		os.Exit(1)
	}

	// Configure webhooks with lease-based namespace exclusion
	isClusterWide := operatorCfg.Namespace.Restricted == ""
	if isClusterWide {
		setupLog.Info("Configuring webhooks with lease-based namespace exclusion for cluster-wide mode")
		internalwebhook.SetExcludedNamespaces(runtimeConfig.ExcludedNamespaces)
	} else {
		setupLog.Info("Configuring webhooks for namespace-restricted mode (no lease checking)",
			"restrictedNamespace", operatorCfg.Namespace.Restricted)
		internalwebhook.SetExcludedNamespaces(nil)
	}

	// Register validation webhook handlers
	setupLog.Info("Registering validation webhooks")

	dcdHandler := webhookvalidation.NewDynamoComponentDeploymentHandler()
	if err = dcdHandler.RegisterWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to register webhook", "webhook", "DynamoComponentDeployment")
		os.Exit(1)
	}

	dgdHandler := webhookvalidation.NewDynamoGraphDeploymentHandler(mgr)
	if err = dgdHandler.RegisterWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to register webhook", "webhook", "DynamoGraphDeployment")
		os.Exit(1)
	}

	dmHandler := webhookvalidation.NewDynamoModelHandler()
	if err = dmHandler.RegisterWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to register webhook", "webhook", "DynamoModel")
		os.Exit(1)
	}

	dgdrHandler := webhookvalidation.NewDynamoGraphDeploymentRequestHandler(
		isClusterWide, ptr.Deref(operatorCfg.GPU.DiscoveryEnabled, true),
	)
	if err = dgdrHandler.RegisterWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to register webhook", "webhook", "DynamoGraphDeploymentRequest")
		os.Exit(1)
	}

	if err = ctrl.NewWebhookManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoGraphDeploymentRequest{}).
		Complete(); err != nil {
		setupLog.Error(err, "unable to register conversion webhook", "webhook", "DynamoGraphDeploymentRequest-conversion")
		os.Exit(1)
	}

	setupLog.Info("Validation webhooks registered successfully")

	// Register defaulting (mutating) webhook handlers
	setupLog.Info("Registering defaulting webhooks")

	dgdDefaulter := webhookdefaulting.NewDGDDefaulter(operatorVersion)
	if err = dgdDefaulter.RegisterWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to register webhook", "webhook", "DynamoGraphDeployment-defaulting")
		os.Exit(1)
	}

	dgdrDefaulter := webhookdefaulting.NewDGDRDefaulter(operatorVersion)
	if err = dgdrDefaulter.RegisterWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to register webhook", "webhook", "DynamoGraphDeploymentRequest-defaulting")
		os.Exit(1)
	}

	setupLog.Info("Defaulting webhooks registered successfully")
	//+kubebuilder:scaffold:builder

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(mainCtx); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
