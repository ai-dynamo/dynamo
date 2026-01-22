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
	"net/url"
	"os"
	"time"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	clientv3 "go.etcd.io/etcd/client/v3"
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
	"sigs.k8s.io/controller-runtime/pkg/cache"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	lwsscheme "sigs.k8s.io/lws/client-go/clientset/versioned/scheme"
	volcanoscheme "volcano.sh/apis/pkg/client/clientset/versioned/scheme"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/etcd"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/namespace_scope"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/rbac"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secrets"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/validation"
	istioclientsetscheme "istio.io/client-go/pkg/clientset/versioned/scheme"
	//+kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

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

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(nvidiacomv1alpha1.AddToScheme(scheme))

	utilruntime.Must(lwsscheme.AddToScheme(scheme))

	utilruntime.Must(volcanoscheme.AddToScheme(scheme))

	utilruntime.Must(grovev1alpha1.AddToScheme(scheme))

	utilruntime.Must(apiextensionsv1.AddToScheme(scheme))

	utilruntime.Must(istioclientsetscheme.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

//nolint:gocyclo
func main() {
	var metricsAddr string
	var enableLeaderElection bool
	var probeAddr string
	var secureMetrics bool
	var enableHTTP2 bool
	var restrictedNamespace string
	var leaderElectionID string
	var leaderElectionNamespace string
	var natsAddr string
	var etcdAddr string
	var istioVirtualServiceGateway string
	var virtualServiceSupportsHTTPS bool
	var ingressControllerClassName string
	var ingressControllerTLSSecretName string
	var ingressHostSuffix string
	var groveTerminationDelay time.Duration
	var modelExpressURL string
	var prometheusEndpoint string
	var mpiRunSecretName string
	var mpiRunSecretNamespace string
	var plannerClusterRoleName string
	var dgdrProfilingClusterRoleName string
	var namespaceScopeLeaseDuration time.Duration
	var namespaceScopeLeaseRenewInterval time.Duration
	var operatorVersion string
	var discoveryBackend string
	var enableWebhooks bool
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.BoolVar(&secureMetrics, "metrics-secure", false,
		"If set the metrics endpoint is served securely")
	flag.BoolVar(&enableHTTP2, "enable-http2", false,
		"If set, HTTP/2 will be enabled for the metrics and webhook servers")
	flag.BoolVar(&enableWebhooks, "enable-webhooks", false,
		"Enable admission webhooks for validation. When enabled, controllers skip validation "+
			"(webhooks handle it). When disabled, controllers perform validation.")
	flag.StringVar(&restrictedNamespace, "restrictedNamespace", "",
		"Enable resources filtering, only the resources belonging to the given namespace will be handled.")
	flag.StringVar(&leaderElectionID, "leader-election-id", "", "Leader election id"+
		"Id to use for the leader election.")
	flag.StringVar(&leaderElectionNamespace,
		"leader-election-namespace", "",
		"Namespace where the leader election resource will be created (default: same as operator namespace)")
	flag.StringVar(&natsAddr, "natsAddr", "", "address of the NATS server")
	flag.StringVar(&etcdAddr, "etcdAddr", "", "address of the etcd server")
	flag.StringVar(&istioVirtualServiceGateway, "istio-virtual-service-gateway", "",
		"The name of the istio virtual service gateway to use")
	flag.BoolVar(&virtualServiceSupportsHTTPS, "virtual-service-supports-https", false,
		"If set, assume VirtualService endpoints are HTTPS")
	flag.StringVar(&ingressControllerClassName, "ingress-controller-class-name", "",
		"The name of the ingress controller class to use")
	flag.StringVar(&ingressControllerTLSSecretName, "ingress-controller-tls-secret-name", "",
		"The name of the ingress controller TLS secret to use")
	flag.StringVar(&ingressHostSuffix, "ingress-host-suffix", "",
		"The suffix to use for the ingress host")
	flag.DurationVar(&groveTerminationDelay, "grove-termination-delay", consts.DefaultGroveTerminationDelay,
		"The termination delay for Grove PodCliqueSets")
	flag.StringVar(&modelExpressURL, "model-express-url", "",
		"URL of the Model Express server to inject into all pods")
	flag.StringVar(&prometheusEndpoint, "prometheus-endpoint", "",
		"URL of the Prometheus endpoint to use for metrics")
	flag.StringVar(&mpiRunSecretName, "mpi-run-ssh-secret-name", "",
		"Name of the secret containing the SSH key for MPI Run (required)")
	flag.StringVar(&mpiRunSecretNamespace, "mpi-run-ssh-secret-namespace", "",
		"Namespace where the MPI SSH secret is located (required)")
	flag.StringVar(&plannerClusterRoleName, "planner-cluster-role-name", "",
		"Name of the ClusterRole for planner (cluster-wide mode only)")
	flag.StringVar(&dgdrProfilingClusterRoleName, "dgdr-profiling-cluster-role-name", "",
		"Name of the ClusterRole for DGDR profiling jobs (cluster-wide mode only)")
	flag.DurationVar(&namespaceScopeLeaseDuration, "namespace-scope-lease-duration", 30*time.Second,
		"Duration of namespace scope marker lease before expiration (namespace-restricted mode only)")
	flag.DurationVar(&namespaceScopeLeaseRenewInterval, "namespace-scope-lease-renew-interval", 10*time.Second,
		"Interval for renewing namespace scope marker lease (namespace-restricted mode only)")
	flag.StringVar(&operatorVersion, "operator-version", "unknown",
		"Version of the operator (used in lease holder identity)")
	flag.StringVar(&discoveryBackend, "discovery-backend", "kubernetes",
		"Discovery backend to use: 'kubernetes' (default, uses Kubernetes API) or 'etcd' (uses ETCD)")
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	if restrictedNamespace == "" && plannerClusterRoleName == "" {
		setupLog.Error(nil, "planner-cluster-role-name is required in cluster-wide mode")
		os.Exit(1)
	}

	// Validate discoveryBackend value
	if discoveryBackend != "kubernetes" && discoveryBackend != "etcd" {
		setupLog.Error(nil, "invalid discovery-backend value, must be 'kubernetes' or 'etcd'", "value", discoveryBackend)
		os.Exit(1)
	}
	setupLog.Info("Discovery backend configured", "backend", discoveryBackend)

	// Validate modelExpressURL if provided
	if modelExpressURL != "" {
		if _, err := url.Parse(modelExpressURL); err != nil {
			setupLog.Error(err, "invalid model-express-url provided", "url", modelExpressURL)
			os.Exit(1)
		}
		setupLog.Info("Model Express URL configured", "url", modelExpressURL)
	}

	if mpiRunSecretName == "" {
		setupLog.Error(nil, "mpi-run-ssh-secret-name is required")
		os.Exit(1)
	}

	if mpiRunSecretNamespace == "" {
		setupLog.Error(nil, "mpi-run-ssh-secret-namespace is required")
		os.Exit(1)
	}

	ctrlConfig := commonController.Config{
		RestrictedNamespace: restrictedNamespace,
		Grove: commonController.GroveConfig{
			Enabled:          false, // Will be set after Grove discovery
			TerminationDelay: groveTerminationDelay,
		},
		LWS: commonController.LWSConfig{
			Enabled: false, // Will be set after LWS discovery
		},
		KaiScheduler: commonController.KaiSchedulerConfig{
			Enabled: false, // Will be set after Kai-scheduler discovery
		},
		EtcdAddress: etcdAddr,
		NatsAddress: natsAddr,
		IngressConfig: commonController.IngressConfig{
			VirtualServiceGateway:      istioVirtualServiceGateway,
			IngressControllerClassName: ingressControllerClassName,
			IngressControllerTLSSecret: ingressControllerTLSSecretName,
			IngressHostSuffix:          ingressHostSuffix,
		},
		ModelExpressURL:    modelExpressURL,
		PrometheusEndpoint: prometheusEndpoint,
		MpiRun: commonController.MpiRunConfig{
			SecretName: mpiRunSecretName,
		},
		RBAC: commonController.RBACConfig{
			PlannerClusterRoleName:       plannerClusterRoleName,
			DGDRProfilingClusterRoleName: dgdrProfilingClusterRoleName,
		},
		DiscoveryBackend: discoveryBackend,
	}

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
	if !enableHTTP2 {
		tlsOpts = append(tlsOpts, disableHTTP2)
	}

	webhookServer := webhook.NewServer(webhook.Options{
		// Bind to all interfaces so the Service can reach the webhook server
		Host: "0.0.0.0",
		// Must match the port exposed by the manager container and targeted by the Service.
		Port: 9443,
		// Must match the mountPath of the webhook certificate secret in the Deployment.
		CertDir: "/tmp/k8s-webhook-server/serving-certs",
		TLSOpts: tlsOpts,
	})

	mgrOpts := ctrl.Options{
		Scheme: scheme,
		Metrics: metricsserver.Options{
			BindAddress:   metricsAddr,
			SecureServing: secureMetrics,
			TLSOpts:       tlsOpts,
		},
		WebhookServer:           webhookServer,
		HealthProbeBindAddress:  probeAddr,
		LeaderElection:          enableLeaderElection,
		LeaderElectionID:        leaderElectionID,
		LeaderElectionNamespace: leaderElectionNamespace,
		// LeaderElectionReleaseOnCancel defines if the leader should step down voluntarily
		// when the Manager ends. This requires the binary to immediately end when the
		// Manager is stopped, otherwise, this setting is unsafe. Setting this significantly
		// speeds up voluntary leader transitions as the new leader don't have to wait
		// LeaseDuration time first.
		//
		// In the default scaffold provided, the program ends immediately after
		// the manager stops, so would be fine to enable this option. However,
		// if you are doing or is intended to do any operation such as perform cleanups
		// after the manager stops then its usage might be unsafe.
		// LeaderElectionReleaseOnCancel: true,
	}
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
			"leaseDuration", namespaceScopeLeaseDuration,
			"renewInterval", namespaceScopeLeaseRenewInterval)

		leaseManager, err = namespace_scope.NewLeaseManager(
			mgr.GetConfig(),
			restrictedNamespace,
			operatorVersion,
			namespaceScopeLeaseDuration,
			namespaceScopeLeaseRenewInterval,
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

		// Pass leaseWatcher to controller config for namespace exclusion filtering
		ctrlConfig.ExcludedNamespaces = leaseWatcher
	}

	// Start resource counter background goroutine (after ExcludedNamespaces is set)
	setupLog.Info("Starting resource counter")
	go observability.StartResourceCounter(mainCtx, mgr.GetClient(), ctrlConfig.ExcludedNamespaces)

	// Detect orchestrators availability using discovery client
	setupLog.Info("Detecting Grove availability...")
	groveEnabled := commonController.DetectGroveAvailability(mainCtx, mgr)
	ctrlConfig.Grove.Enabled = groveEnabled
	setupLog.Info("Detecting LWS availability...")
	lwsEnabled := commonController.DetectLWSAvailability(mainCtx, mgr)
	setupLog.Info("Detecting Volcano availability...")
	volcanoEnabled := commonController.DetectVolcanoAvailability(mainCtx, mgr)
	// LWS for multinode deployment usage depends on both LWS and Volcano availability
	ctrlConfig.LWS.Enabled = lwsEnabled && volcanoEnabled
	// Detect Kai-scheduler availability using discovery client
	setupLog.Info("Detecting Kai-scheduler availability...")
	kaiSchedulerEnabled := commonController.DetectKaiSchedulerAvailability(mainCtx, mgr)
	ctrlConfig.KaiScheduler.Enabled = kaiSchedulerEnabled

	setupLog.Info("Detected orchestrators availability",
		"grove", groveEnabled,
		"lws", lwsEnabled,
		"volcano", volcanoEnabled,
		"kai-scheduler", kaiSchedulerEnabled,
	)

	// Create etcd client
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:            []string{etcdAddr},
		DialTimeout:          5 * time.Second,
		DialKeepAliveTime:    10 * time.Second,
		DialKeepAliveTimeout: 3 * time.Second,
	})
	if err != nil {
		setupLog.Error(err, "unable to create etcd client")
		os.Exit(1)
	}

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
		mpiRunSecretNamespace,
		mpiRunSecretName,
	)

	if err = (&controller.DynamoComponentDeploymentReconciler{
		Client:                mgr.GetClient(),
		Recorder:              mgr.GetEventRecorderFor("dynamocomponentdeployment"),
		Config:                ctrlConfig,
		EtcdStorage:           etcd.NewStorage(cli),
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
		Config:                ctrlConfig,
		DockerSecretRetriever: dockerSecretRetriever,
		ScaleClient:           scaleClient,
		MPISecretReplicator:   mpiSecretReplicator,
		RBACManager:           rbacManager,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoGraphDeployment")
		os.Exit(1)
	}

	if err = (&controller.DynamoGraphDeploymentScalingAdapterReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("dgdscalingadapter"),
		Config:   ctrlConfig,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DGDScalingAdapter")
		os.Exit(1)
	}

	if err = (&controller.DynamoGraphDeploymentRequestReconciler{
		Client:      mgr.GetClient(),
		Recorder:    mgr.GetEventRecorderFor("dynamographdeploymentrequest"),
		Config:      ctrlConfig,
		RBACManager: rbacManager,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoGraphDeploymentRequest")
		os.Exit(1)
	}

	if err = (&controller.DynamoModelReconciler{
		Client:         mgr.GetClient(),
		Recorder:       mgr.GetEventRecorderFor("dynamomodel"),
		EndpointClient: modelendpoint.NewClient(),
		Config:         ctrlConfig,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoModel")
		os.Exit(1)
	}

	// Set webhooks enabled flag in config
	ctrlConfig.WebhooksEnabled = enableWebhooks

	if enableWebhooks {
		setupLog.Info("Webhooks are enabled - webhooks will validate, controllers will skip validation")
	} else {
		setupLog.Info("Webhooks are disabled - controllers will validate (defense in depth)")
	}

	// Configure webhooks with lease-based namespace exclusion (only if enabled)
	// In cluster-wide mode, inject ctrlConfig.ExcludedNamespaces (leaseWatcher) so webhooks can defer
	// to namespace-restricted operators. In namespace-restricted mode, webhooks validate without checking
	// leases (ExcludedNamespaces is nil). The webhooks use LeaseAwareValidator wrapper to add coordination.
	if enableWebhooks {
		if ctrlConfig.RestrictedNamespace == "" {
			// Cluster-wide mode: inject the same ExcludedNamespaces used by controllers
			setupLog.Info("Configuring webhooks with lease-based namespace exclusion for cluster-wide mode")
			internalwebhook.SetExcludedNamespaces(ctrlConfig.ExcludedNamespaces)
		} else {
			// Namespace-restricted mode: no exclusion checking needed (validators not wrapped)
			setupLog.Info("Configuring webhooks for namespace-restricted mode (no lease checking)",
				"restrictedNamespace", ctrlConfig.RestrictedNamespace)
			internalwebhook.SetExcludedNamespaces(nil)
		}

		// Register validation webhook handlers
		setupLog.Info("Registering validation webhooks")

		dcdHandler := webhookvalidation.NewDynamoComponentDeploymentHandler()
		if err = dcdHandler.RegisterWithManager(mgr); err != nil {
			setupLog.Error(err, "unable to register webhook", "webhook", "DynamoComponentDeployment")
			os.Exit(1)
		}

		dgdHandler := webhookvalidation.NewDynamoGraphDeploymentHandler()
		if err = dgdHandler.RegisterWithManager(mgr); err != nil {
			setupLog.Error(err, "unable to register webhook", "webhook", "DynamoGraphDeployment")
			os.Exit(1)
		}

		dmHandler := webhookvalidation.NewDynamoModelHandler()
		if err = dmHandler.RegisterWithManager(mgr); err != nil {
			setupLog.Error(err, "unable to register webhook", "webhook", "DynamoModel")
			os.Exit(1)
		}

		isClusterWide := ctrlConfig.RestrictedNamespace == ""
		dgdrHandler := webhookvalidation.NewDynamoGraphDeploymentRequestHandler(isClusterWide)
		if err = dgdrHandler.RegisterWithManager(mgr); err != nil {
			setupLog.Error(err, "unable to register webhook", "webhook", "DynamoGraphDeploymentRequest")
			os.Exit(1)
		}

		setupLog.Info("Validation webhooks registered successfully")
	}
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
