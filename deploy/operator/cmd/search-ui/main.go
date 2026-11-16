/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Command search-ui serves the Dynamo Search UI: a browser view of DynamoGraphDeploymentRequest,
// Run and Candidate resources with UI-local history. Reach it with
// kubectl port-forward on its ClusterIP Service.
package main

import (
	"context"
	"errors"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	nvv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui/history"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

var scheme = k8sruntime.NewScheme()

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(nvv1beta2.AddToScheme(scheme))
}

func main() {
	os.Exit(run())
}

func run() int {
	var (
		listenAddr      string
		namespace       string
		historyBackend  string
		historyDir      string
		azureAccountURL string
		azureContainer  string
		disableDGD      bool
	)
	logOptions := zap.Options{}
	logOptions.BindFlags(flag.CommandLine)
	flag.StringVar(&listenAddr, "listen-addr", ":8080", "HTTP listen address")
	flag.StringVar(&namespace, "namespace", "", "namespace to watch; empty watches all namespaces")
	flag.StringVar(&historyBackend, "history-backend", "none", "durable history backend: none, local or azblob")
	flag.StringVar(&historyDir, "history-dir", "/var/lib/dynamo-search-ui", "directory for the local history backend")
	flag.StringVar(&azureAccountURL, "azure-blob-account-url", os.Getenv("AZURE_BLOB_ACCOUNT_URL"),
		"blob service URL for the azblob backend, e.g. https://account.blob.core.windows.net")
	flag.StringVar(&azureContainer, "azure-blob-container", os.Getenv("AZURE_BLOB_CONTAINER"),
		"container name for the azblob backend")
	flag.BoolVar(&disableDGD, "disable-dgd-create", false, "serve read-only; refuse POST /api/v1/dgd")
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&logOptions)))
	log := ctrl.Log.WithName("search-ui")
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	restConfig, err := ctrl.GetConfig()
	if err != nil {
		log.Error(err, "kubeconfig")
		return 1
	}

	var store history.Store
	switch historyBackend {
	case "none":
	case "local":
		store, err = history.NewLocal(historyDir)
	case "azblob":
		store, err = history.NewAzureBlob(ctx, azureAccountURL, azureContainer)
	default:
		err = errors.New("unknown history backend " + historyBackend)
	}
	if err != nil {
		log.Error(err, "history backend", "backend", historyBackend)
		return 1
	}
	log.Info("history backend ready", "backend", historyBackend)

	index := searchui.NewIndex(store)
	if err := index.Load(ctx); err != nil {
		log.Error(err, "load history")
		return 1
	}
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		log.Error(err, "kubernetes clientset")
		return 1
	}
	logs := searchui.NewLogManager(clientset, store, log.WithName("logs"))

	var writer client.Client
	if !disableDGD {
		writer, err = client.New(restConfig, client.Options{Scheme: scheme})
		if err != nil {
			log.Error(err, "kubernetes client")
			return 1
		}
	}
	server := searchui.NewServer(index, logs, writer)
	watcher, err := searchui.NewWatcher(restConfig, namespace, index, logs, server.Notify, log.WithName("watch"))
	if err != nil {
		log.Error(err, "watcher")
		return 1
	}

	httpServer := &http.Server{
		Addr:              listenAddr,
		Handler:           server,
		ReadHeaderTimeout: 10 * time.Second,
	}
	errCh := make(chan error, 2)
	go func() {
		log.Info("listening", "addr", listenAddr, "namespace", namespace)
		errCh <- httpServer.ListenAndServe()
	}()
	go func() { errCh <- watcher.Start(ctx) }()

	select {
	case <-ctx.Done():
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) && !errors.Is(err, context.Canceled) {
			log.Error(err, "exiting")
			cancel()
			_ = httpServer.Close()
			return 1
		}
	}
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	_ = httpServer.Shutdown(shutdownCtx)
	return 0
}
