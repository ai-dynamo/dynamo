/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
 */

package main

import (
	"crypto/tls"
	"flag"
	"os"
	"time"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	clientv3 "go.etcd.io/etcd/client/v3"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"sigs.k8s.io/controller-runtime/pkg/cache"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/controller"
	commonController "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/etcd"
	istioclientsetscheme "istio.io/client-go/pkg/clientset/versioned/scheme"
	//+kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(nvidiacomv1alpha1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func main() {
	var metricsAddr string
	var enableLeaderElection bool
	var probeAddr string
	var secureMetrics bool
	var enableHTTP2 bool
	var restrictedNamespace string
	var leaderElectionID string
	var natsAddr string
	var etcdAddr string
	var istioVirtualServiceGateway string
	var ingressControllerClassName string
	var ingressControllerTLSSecretName string
	var ingressHostSuffix string
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.BoolVar(&secureMetrics, "metrics-secure", false,
		"If set the metrics endpoint is served securely")
	flag.BoolVar(&enableHTTP2, "enable-http2", false,
		"If set, HTTP/2 will be enabled for the metrics and webhook servers")
	flag.StringVar(&restrictedNamespace, "restrictedNamespace", "",
		"Enable resources filtering, only the resources belonging to the given namespace will be handled.")
	flag.StringVar(&leaderElectionID, "leader-election-id", "", "Leader election id"+
		"Id to use for the leader election.")
	flag.StringVar(&natsAddr, "natsAddr", "", "address of the NATS server")
	flag.StringVar(&etcdAddr, "etcdAddr", "", "address of the etcd server")
	flag.StringVar(&istioVirtualServiceGateway, "istio-virtual-service-gateway", "",
		"The name of the istio virtual service gateway to use")
	flag.StringVar(&ingressControllerClassName, "ingress-controller-class-name", "",
		"The name of the ingress controller class to use")
	flag.StringVar(&ingressControllerTLSSecretName, "ingress-controller-tls-secret-name", "",
		"The name of the ingress controller TLS secret to use")
	flag.StringVar(&ingressHostSuffix, "ingress-host-suffix", "",
		"The suffix to use for the ingress host")
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	utilruntime.Must(istioclientsetscheme.AddToScheme(scheme))

	ctrlConfig := commonController.Config{
		RestrictedNamespace: restrictedNamespace,
	}

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

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
		TLSOpts: tlsOpts,
	})

	mgrOpts := ctrl.Options{
		Scheme: scheme,
		Metrics: metricsserver.Options{
			BindAddress:   metricsAddr,
			SecureServing: secureMetrics,
			TLSOpts:       tlsOpts,
		},
		WebhookServer:          webhookServer,
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       leaderElectionID,
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
	}
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), mgrOpts)
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

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
	if err = (&controller.DynamoComponentDeploymentReconciler{
		Client:            mgr.GetClient(),
		Scheme:            mgr.GetScheme(),
		Recorder:          mgr.GetEventRecorderFor("dynamocomponentdeployment"),
		Config:            ctrlConfig,
		NatsAddr:          natsAddr,
		EtcdAddr:          etcdAddr,
		EtcdStorage:       etcd.NewStorage(cli),
		UseVirtualService: istioVirtualServiceGateway != "",
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoComponentDeployment")
		os.Exit(1)
	}
	if err = (&controller.DynamoComponentReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("dynamocomponent"),
		Config:   ctrlConfig,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoComponent")
		os.Exit(1)
	}
	if err = (&controller.DynamoGraphDeploymentReconciler{
		Client:                     mgr.GetClient(),
		Scheme:                     mgr.GetScheme(),
		Recorder:                   mgr.GetEventRecorderFor("dynamographdeployment"),
		Config:                     ctrlConfig,
		VirtualServiceGateway:      istioVirtualServiceGateway,
		IngressControllerClassName: ingressControllerClassName,
		IngressControllerTLSSecret: ingressControllerTLSSecretName,
		IngressHostSuffix:          ingressHostSuffix,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "DynamoGraphDeployment")
		os.Exit(1)
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
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
