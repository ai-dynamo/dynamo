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
	"flag"
	"fmt"
	"os"
	"time"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	semver "github.com/Masterminds/semver/v3"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	istioclientsetscheme "istio.io/client-go/pkg/clientset/versioned/scheme"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	k8sCache "k8s.io/client-go/tools/cache"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	lwsscheme "sigs.k8s.io/lws/client-go/clientset/versioned/scheme"
	volcanoscheme "volcano.sh/apis/pkg/client/clientset/versioned/scheme"

	configapi "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	internalcert "github.com/ai-dynamo/dynamo/deploy/operator/internal/cert"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/namespace_scope"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/rbac"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secrets"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/webhooks"
	//+kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(nvidiacomv1alpha1.AddToScheme(scheme))

	utilruntime.Must(nvidiacomv1beta1.AddToScheme(scheme))

	utilruntime.Must(lwsscheme.AddToScheme(scheme))

	utilruntime.Must(volcanoscheme.AddToScheme(scheme))

	utilruntime.Must(grovev1alpha1.AddToScheme(scheme))

	utilruntime.Must(apiextensionsv1.AddToScheme(scheme))

	utilruntime.Must(admissionregistrationv1.AddToScheme(scheme))

	utilruntime.Must(istioclientsetscheme.AddToScheme(scheme))

	utilruntime.Must(gaiev1.Install(scheme))

	utilruntime.Must(configapi.AddToScheme(scheme))
}

// +kubebuilder:rbac:groups=authentication.k8s.io,resources=tokenreviews,verbs=create
// +kubebuilder:rbac:groups=authorization.k8s.io,resources=subjectaccessreviews,verbs=create

//nolint:gocyclo
func main() {
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

	mgrOpts, cfg, err := apply(configFile)
	if err != nil {
		setupLog.Error(err, "Unable to load the configuration")
		os.Exit(1)
	}

	// Validates the configuration after it has been loaded.
	if err := config.Validate(&cfg).ToAggregate(); err != nil {
		setupLog.Error(err, "Unable to validate the configuration")
		os.Exit(1)
	}

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

	ctx := ctrl.SetupSignalHandler()

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
	directClient, err := client.New(mgr.GetConfig(), client.Options{Scheme: scheme})
	if err != nil {
		setupLog.Error(err, "unable to create direct client for cert management")
		os.Exit(1)
	}
	certMgr, err := internalcert.NewCertManager(directClient, &cfg.Server.Webhook)
	if err != nil {
		setupLog.Error(err, "unable to create cert manager")
		os.Exit(1)
	}
	// Auto mode runs one synchronous certificate refresh with the direct client,
	// then registers the cert-controller with the not-yet-started manager.
	if err = certMgr.SetupAndRunOnce(ctx, mgr); err != nil {
		setupLog.Error(err, "failed to setup webhook certificate management")
		os.Exit(1)
	}

	// Initialize namespace scope mechanism
	restrictedNamespace := cfg.Namespace.Restricted
	leaseWatcher, err := setupLease(ctx, mgr, &cfg, operatorVersion, restrictedNamespace)
	if err != nil {
		setupLog.Error(err, "unable to setup namespace scope lease mechanism")
		os.Exit(1)
	}

	// Initialize runtime config (auto-detection of orchestrators and service mesh)
	runtimeConfig, err := setupRuntimeConfig(ctx, mgr, &cfg, leaseWatcher)
	if err != nil {
		setupLog.Error(err, "unable to setup runtime config")
		os.Exit(1)
	}

	dockerSecretRetriever, err := setupDockerSecretRetriever(ctx, mgr, restrictedNamespace)
	if err != nil {
		setupLog.Error(err, "unable to setup docker secret retriever")
		os.Exit(1)
	}

	scaleClient, err := setupScalesGetter(mgr)
	if err != nil {
		setupLog.Error(err, "unable to create scale client")
		os.Exit(1)
	}

	if err := setupProbeEndpoints(mgr); err != nil {
		setupLog.Error(err, "Unable to setup probe endpoints")
		os.Exit(1)
	}

	// Setup controllers synchronously before mgr.Start().
	// Controllers don't depend on TLS certificates.
	controllerOpts := controller.SetupControllersOpts{
		RuntimeConfig:         runtimeConfig,
		DockerSecretRetriever: dockerSecretRetriever,
		SSHKeyManager:         secret.NewSSHKeyManager(mgr.GetClient(), cfg.MPI),
		RBACManager:           rbac.NewManager(mgr.GetClient()),
		ScaleClient:           scaleClient,
		OperatorImage:         operatorImage,
		OperatorPullPolicy:    pullPolicy,
	}
	if err := setupControllers(ctx, mgr, &cfg, controllerOpts); err != nil {
		setupLog.Error(err, "Unable to setup controllers")
		os.Exit(1)
	}

	if failedWebhook, err := webhooks.Setup(mgr, &cfg, runtimeConfig, operatorVersion); err != nil {
		setupLog.Error(err, "Unable to create webhook", "webhook", failedWebhook)
		os.Exit(1)
	}

	// CertManager.SetupAndRunOnce has already bootstrapped auto-mode TLS
	// secrets before this point. Auto mode can therefore patch admission and
	// conversion CAs immediately; manual mode waits for externally provided
	// ca.crt and only patches conversion, leaving admission CA management
	// out-of-band.
	caInjector, err := internalcert.NewCABundleInjector(directClient, &cfg)
	if err != nil {
		setupLog.Error(err, "unable to create CA bundle injector")
		os.Exit(1)
	}
	if cfg.Server.Webhook.CertProvisionMode == configapi.CertProvisionModeAuto {
		if err := caInjector.InjectAll(ctx); err != nil {
			setupLog.Error(err, "failed to inject CA bundles into webhook configurations")
			os.Exit(1)
		}
	} else {
		// Manual mode gets webhook CA material out-of-band. Missing ca.crt
		// blocks startup instead of running with unauthenticated conversion.
		if err := caInjector.InjectCRDConversionCA(ctx); err != nil {
			setupLog.Error(err, "failed to inject CRD conversion CA bundle")
			os.Exit(1)
		}
	}

	// mgr.Start reads tls.crt and tls.key from the projected Secret volume
	// synchronously. Secret API updates are not enough because kubelet projects
	// them into already-running pods asynchronously.
	if err := certMgr.WaitForMountedCertificate(ctx); err != nil {
		setupLog.Error(err, "failed waiting for mounted webhook TLS certificate")
		os.Exit(1)
	}

	// Kubernetes propagates webhook configuration asynchronously, especially
	// with HA apiservers. A missing or stale CA must fail closed during manager
	// cache startup rather than allowing the operator to run without conversion
	// or admission.
	setupLog.Info("starting manager")
	if err := mgr.Start(ctx); err != nil {
		setupLog.Error(err, "Could not run manager")
		os.Exit(1)
	}
}

func setupControllers(
	_ context.Context,
	mgr ctrl.Manager,
	cfg *configapi.OperatorConfiguration,
	opts controller.SetupControllersOpts,
) error {
	if failedCtrl, err := controller.SetupControllers(mgr, cfg, opts); err != nil {
		return fmt.Errorf("unable to create controller %s: %w", failedCtrl, err)
	}

	// Register after ExcludedNamespaces is set so cluster-wide metrics skip restricted namespaces.
	setupLog.Info("Registering resource counter")
	if err := mgr.Add(observability.NewResourceCounter(
		mgr.GetClient(),
		opts.RuntimeConfig.ExcludedNamespaces,
	)); err != nil {
		setupLog.Error(err, "unable to register resource counter")
		os.Exit(1)
	}

	return nil
}

// setupLease initializes the namespace scope lease mechanism based on the operator configuration.
// It returns a LeaseWatcher if the operator is running in cluster-wide mode, or nil if running
// in namespace-restricted mode.
func setupLease(
	ctx context.Context,
	mgr ctrl.Manager,
	cfg *configapi.OperatorConfiguration,
	operatorVersion string,
	restrictedNamespace string,
) (*namespace_scope.LeaseWatcher, error) {
	if restrictedNamespace != "" {
		// Namespace-restricted mode: Create and maintain namespace scope marker lease
		setupLog.Info("Creating namespace scope marker lease manager",
			"namespace", restrictedNamespace,
			"leaseDuration", cfg.Namespace.Scope.LeaseDuration.Duration,
			"renewInterval", cfg.Namespace.Scope.LeaseRenewInterval.Duration)

		leaseManager, err := namespace_scope.NewLeaseManager(
			mgr.GetConfig(),
			restrictedNamespace,
			operatorVersion,
			cfg.Namespace.Scope.LeaseDuration.Duration,
			cfg.Namespace.Scope.LeaseRenewInterval.Duration,
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create namespace scope marker lease manager: %w", err)
		}

		// Start the lease manager
		if err = leaseManager.Start(ctx); err != nil {
			return nil, fmt.Errorf("unable to start namespace scope marker lease manager: %w", err)
		}

		// Monitor for fatal lease errors
		// If lease renewal fails repeatedly, we must exit to prevent split-brain
		go func() {
			select {
			case err := <-leaseManager.Errors():
				setupLog.Error(err, "FATAL: Lease manager encountered unrecoverable error, shutting down to prevent split-brain")
				os.Exit(1)
			case <-ctx.Done():
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
		return nil, nil
	}

	// Cluster-wide mode: Watch for namespace scope marker leases
	setupLog.Info("Setting up namespace scope marker lease watcher for cluster-wide mode")

	leaseWatcher, err := namespace_scope.NewLeaseWatcher(mgr.GetConfig())
	if err != nil {
		return nil, fmt.Errorf("unable to create namespace scope marker lease watcher: %w", err)
	}

	// Start the lease watcher
	if err = leaseWatcher.Start(ctx); err != nil {
		return nil, fmt.Errorf("unable to start namespace scope marker lease watcher: %w", err)
	}

	setupLog.Info("Namespace scope marker lease watcher started successfully")
	return leaseWatcher, nil
}

func setupRuntimeConfig(
	ctx context.Context,
	mgr ctrl.Manager,
	cfg *configapi.OperatorConfiguration,
	excludedNamespaces commoncontroller.ExcludedNamespacesInterface,
) (*commoncontroller.RuntimeConfig, error) {
	runtimeConfig := &commoncontroller.RuntimeConfig{
		ExcludedNamespaces: excludedNamespaces,
	}

	// Detect orchestrators availability using discovery client.
	// Config overrides (*bool) take precedence over auto-detection:
	//   nil   = auto-detect (backward compatible default)
	//   false = forcibly disabled regardless of API availability
	//   true  = forcibly enabled; hard exit if API is not available (misconfiguration)
	setupLog.Info("Detecting Grove availability...")
	groveDetected := commoncontroller.DetectGroveAvailability(ctx, mgr)
	switch {
	case cfg.Orchestrators.Grove.Enabled == nil:
		runtimeConfig.GroveEnabled = groveDetected
	case *cfg.Orchestrators.Grove.Enabled:
		if !groveDetected {
			return nil, fmt.Errorf(
				"Grove is explicitly enabled in config but the Grove API group was not detected in the cluster",
			)
		}
		runtimeConfig.GroveEnabled = true
	default:
		setupLog.Info("Grove is explicitly disabled via config override")
		runtimeConfig.GroveEnabled = false
	}

	setupLog.Info("Detecting LWS availability...")
	lwsDetected := commoncontroller.DetectLWSAvailability(ctx, mgr)
	setupLog.Info("Detecting Volcano availability...")
	volcanoDetected := commoncontroller.DetectVolcanoAvailability(ctx, mgr)
	// LWS for multinode deployment usage depends on both LWS and Volcano availability
	switch {
	case cfg.Orchestrators.LWS.Enabled == nil:
		runtimeConfig.LWSEnabled = lwsDetected && volcanoDetected
	case *cfg.Orchestrators.LWS.Enabled:
		if !lwsDetected {
			return nil, fmt.Errorf("LWS is explicitly enabled in config but the LWS API group was not detected in the cluster")
		}
		if !volcanoDetected {
			return nil, fmt.Errorf(
				"LWS is explicitly enabled in config but the Volcano API group was not detected in the cluster",
			)
		}
		runtimeConfig.LWSEnabled = true
	default:
		setupLog.Info("LWS is explicitly disabled via config override")
		runtimeConfig.LWSEnabled = false
	}

	switch {
	case cfg.Orchestrators.VolcanoScheduler.Enabled == nil:
		runtimeConfig.VolcanoSchedulerEnabled = false
	case *cfg.Orchestrators.VolcanoScheduler.Enabled:
		if !volcanoDetected {
			return nil, fmt.Errorf(
				"Volcano scheduler integration is explicitly enabled in config but the Volcano API group " +
					"was not detected in the cluster",
			)
		}
		runtimeConfig.VolcanoSchedulerEnabled = true
	default:
		setupLog.Info("Volcano scheduler integration is explicitly disabled via config override")
		runtimeConfig.VolcanoSchedulerEnabled = false
	}

	// Detect Kai-scheduler availability using discovery client
	setupLog.Info("Detecting Kai-scheduler availability...")
	kaiSchedulerDetected := commoncontroller.DetectKaiSchedulerAvailability(ctx, mgr)
	switch {
	case cfg.Orchestrators.KaiScheduler.Enabled == nil:
		runtimeConfig.KaiSchedulerEnabled = kaiSchedulerDetected
	case *cfg.Orchestrators.KaiScheduler.Enabled:
		if !kaiSchedulerDetected {
			return nil, fmt.Errorf(
				"Kai-scheduler is explicitly enabled in config but the scheduling.run.ai API group was not detected in the cluster",
			)
		}
		runtimeConfig.KaiSchedulerEnabled = true
	default:
		setupLog.Info("Kai-scheduler is explicitly disabled via config override")
		runtimeConfig.KaiSchedulerEnabled = false
	}

	setupLog.Info("Detecting DRA (Dynamic Resource Allocation) availability...")
	draDetected := commoncontroller.DetectDRAAvailability(ctx, mgr)
	switch {
	case cfg.DRA.Enabled == nil:
		runtimeConfig.DRAEnabled = draDetected
	case *cfg.DRA.Enabled:
		if !draDetected {
			return nil, fmt.Errorf("DRA is explicitly enabled in config but the resource.k8s.io/v1 API" +
				" was not detected in the cluster (requires Kubernetes 1.34+)")
		}
		runtimeConfig.DRAEnabled = true
	default:
		setupLog.Info("DRA is explicitly disabled via config override")
		runtimeConfig.DRAEnabled = false
	}

	setupLog.Info("Detecting Istio availability...")
	switch {
	case cfg.ServiceMesh.Enabled == nil:
		setupLog.Info("Auto-detecting Istio availability")
		runtimeConfig.IstioEnabled = commoncontroller.DetectIstioDestinationRuleAvailability(ctx, mgr)
	case *cfg.ServiceMesh.Enabled:
		setupLog.Info("Istio service mesh is explicitly enabled; verifying availability")
		istioDetected := commoncontroller.DetectIstioDestinationRuleAvailability(ctx, mgr)
		if !istioDetected {
			return nil, fmt.Errorf("Service mesh is explicitly enabled but the networking.istio.io" +
				" DestinationRule API group was not detected in the cluster")
		}
		runtimeConfig.IstioEnabled = true
	default:
		setupLog.Info("Istio service mesh is explicitly disabled via config override")
		runtimeConfig.IstioEnabled = false
	}

	setupLog.Info("Detected orchestrators availability",
		"grove", runtimeConfig.GroveEnabled,
		"lws", runtimeConfig.LWSEnabled,
		"volcano", volcanoDetected,
		"volcano-scheduler", runtimeConfig.VolcanoSchedulerEnabled,
		"kai-scheduler", runtimeConfig.KaiSchedulerEnabled,
		"dra", runtimeConfig.DRAEnabled,
		"istio", runtimeConfig.IstioEnabled,
	)

	return runtimeConfig, nil
}

func setupScalesGetter(mgr ctrl.Manager) (scale.ScalesGetter, error) {
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

// TODO: Consider moving this secret lookup into controller-runtime cache/indexing where possible.
// Avoid periodic full-list refreshes.
func setupDockerSecretRetriever(
	ctx context.Context,
	mgr ctrl.Manager,
	restrictedNamespace string,
) (*secrets.DockerSecretIndexer, error) {
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
	go factory.Start(ctx.Done())
	// Wait for the initial sync
	if !k8sCache.WaitForCacheSync(ctx.Done(), secretInformer.HasSynced) {
		return nil, fmt.Errorf("failed to sync informer cache")
	}
	setupLog.Info("Secret informer cache synced and ready")
	_, err := secretInformer.AddEventHandler(k8sCache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			secret := obj.(*corev1.Secret)
			if secret.Type == corev1.SecretTypeDockerConfigJson {
				setupLog.Info("refreshing docker secrets index after secret creation...")
				err := dockerSecretRetriever.RefreshIndex(ctx)
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
				err := dockerSecretRetriever.RefreshIndex(ctx)
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
				err := dockerSecretRetriever.RefreshIndex(ctx)
				if err != nil {
					setupLog.Error(err, "unable to refresh docker secrets index after secret deletion")
				} else {
					setupLog.Info("docker secrets index refreshed after secret deletion")
				}
			}
		},
	})
	if err != nil {
		return nil, fmt.Errorf("unable to add event handler to secret informer: %w", err)
	}
	if err := dockerSecretRetriever.RefreshIndex(ctx); err != nil {
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
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := dockerSecretRetriever.RefreshIndex(ctx); err != nil {
					setupLog.Error(err, "failed to refresh docker secrets index")
				}
			}
		}
	}()
	return dockerSecretRetriever, nil
}

// setupProbeEndpoints registers the health endpoints
func setupProbeEndpoints(mgr ctrl.Manager) error {
	defer setupLog.Info("Probe endpoints are configured on healthz and readyz")

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		return fmt.Errorf("unable to set up health check: %w", err)
	}

	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		return fmt.Errorf("unable to set up ready check: %w", err)
	}

	return nil
}

func apply(configFile string) (ctrl.Options, configapi.OperatorConfiguration, error) {
	options, cfg, err := config.Load(scheme, configFile)
	if err != nil {
		return options, cfg, err
	}
	cfgStr, err := config.Encode(scheme, &cfg)
	if err != nil {
		return options, cfg, err
	}
	setupLog.Info("Configuration loaded", "config", cfgStr)
	return options, cfg, nil
}
