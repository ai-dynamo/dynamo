/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package clusterenv supports controller tests against a real Kubernetes
// cluster. Cluster and image lifecycle remain external.
package clusterenv

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/webhookconfig"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	controllerconfig "sigs.k8s.io/controller-runtime/pkg/config"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

const (
	// ContextEnvVar names the kubeconfig context explicitly unlocked for cluster tests.
	ContextEnvVar = "DYNAMO_CLUSTERTEST_CONTEXT"

	workloadBlockQuotaName = "clusterenv-block-workloads"
)

// Options configures an Env.
type Options struct {
	Scheme                 *runtime.Scheme
	SetupWebhooks          func(ctrl.Manager) error
	AdditionalAdmission    webhookconfig.Configurations
	WebhookProxyImage      string
	EventuallyTimeout      time.Duration
	NamespaceDeleteTimeout time.Duration
}

// Env lazily connects to one externally managed Kubernetes cluster.
type Env struct {
	opts Options

	mu   sync.Mutex
	runM bool
	once sync.Once
	rt   *runtimeEnv
	err  error
}

// New returns an Env configured with opts.
func New(opts Options) *Env {
	return &Env{opts: normalizeOptions(opts)}
}

// RunM owns the shared cluster connection and webhook runtime for one package.
// Its runner is normally *testing.M; the interface allows multiple test
// environments to compose their package lifecycles without running tests twice.
func (e *Env) RunM(m interface{ Run() int }) int {
	e.mu.Lock()
	e.runM = true
	e.mu.Unlock()

	code := m.Run()
	if e.rt != nil {
		if err := e.rt.stop(); err != nil {
			fmt.Fprintf(os.Stderr, "clusterenv: stop shared runtime: %v\n", err)
			if code == 0 {
				code = 1
			}
		}
	}
	return code
}

// RunT creates a namespace and returns a test environment backed by the
// explicitly unlocked cluster context.
func (e *Env) RunT(tb testing.TB) *TestEnv {
	tb.Helper()
	e.mu.Lock()
	runM := e.runM
	e.mu.Unlock()
	if !runM {
		tb.Fatal("clusterenv.RunT requires RunM from TestMain")
	}
	e.once.Do(func() {
		e.rt, e.err = startRuntime(e.opts)
	})
	if e.err != nil {
		tb.Fatalf("start clusterenv: %v", e.err)
	}

	testEnv := newTestEnv(tb, e.rt, e.opts)
	tb.Cleanup(testEnv.stop)
	return testEnv
}

type runtimeEnv struct {
	opts       Options
	config     *rest.Config
	scheme     *runtime.Scheme
	client     client.Client
	kubeClient kubernetes.Interface
	webhooks   *webhookRuntime
}

func startRuntime(opts Options) (*runtimeEnv, error) {
	if opts.Scheme == nil {
		return nil, errors.New("clusterenv: Scheme is required")
	}
	config, err := loadRESTConfig()
	if err != nil {
		return nil, err
	}
	k8sClient, err := client.New(config, client.Options{Scheme: opts.Scheme})
	if err != nil {
		return nil, fmt.Errorf("clusterenv: create client: %w", err)
	}
	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("clusterenv: create Kubernetes client: %w", err)
	}
	rt := &runtimeEnv{
		opts: opts, config: config, scheme: opts.Scheme, client: k8sClient, kubeClient: kubeClient,
	}
	if opts.SetupWebhooks != nil {
		rt.webhooks, err = startWebhookRuntime(opts, config, kubeClient)
		if err != nil {
			return nil, err
		}
	}
	return rt, nil
}

func (e *runtimeEnv) stop() error {
	if e.webhooks == nil {
		return nil
	}
	return e.webhooks.stop()
}

func loadRESTConfig() (*rest.Config, error) {
	contextName := os.Getenv(ContextEnvVar)
	if contextName == "" {
		return nil, fmt.Errorf("clusterenv: %s must be set to the allowed kubeconfig context", ContextEnvVar)
	}

	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	rawConfig, err := loadingRules.Load()
	if err != nil {
		return nil, fmt.Errorf("clusterenv: load kubeconfig: %w", err)
	}
	if _, exists := rawConfig.Contexts[contextName]; !exists {
		return nil, fmt.Errorf("clusterenv: context %q from %s does not exist in kubeconfig", contextName, ContextEnvVar)
	}

	config, err := clientcmd.NewNonInteractiveClientConfig(
		*rawConfig,
		contextName,
		&clientcmd.ConfigOverrides{},
		loadingRules,
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("clusterenv: load context %q: %w", contextName, err)
	}
	return config, nil
}

// TestEnv provides a dedicated namespace and a controller manager for one test.
type TestEnv struct {
	tb        testing.TB
	rt        *runtimeEnv
	opts      Options
	namespace string

	mu             sync.Mutex
	managerStarted bool
	managerCancel  context.CancelFunc
	managerDone    chan error
}

func newTestEnv(tb testing.TB, rt *runtimeEnv, opts Options) *TestEnv {
	tb.Helper()
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: "clusterenv-"}}
	if err := rt.client.Create(context.Background(), ns); err != nil {
		tb.Fatalf("create cluster-test namespace: %v", err)
	}
	return &TestEnv{tb: tb, rt: rt, opts: opts, namespace: ns.Name}
}

// Namespace returns the namespace dedicated to this test.
func (e *TestEnv) Namespace() string {
	return e.namespace
}

// Client returns an unrestricted client for the selected cluster. Tests must
// use Namespace for namespaced fixtures.
func (e *TestEnv) Client() client.Client {
	return e.rt.client
}

// RESTConfig returns a copy of the selected cluster REST configuration.
func (e *TestEnv) RESTConfig() *rest.Config {
	return rest.CopyConfig(e.rt.config)
}

// BlockWorkloads prevents controllers from creating ReplicaSets and Pods in
// the test namespace without changing the workload manifests under test.
func (e *TestEnv) BlockWorkloads() {
	e.tb.Helper()
	quota := workloadBlockQuota(e.namespace)
	if err := e.rt.client.Create(context.Background(), quota); err != nil && !apierrors.IsAlreadyExists(err) {
		e.tb.Fatalf("create cluster-test workload quota: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), e.opts.EventuallyTimeout)
	defer cancel()
	if err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		current := &corev1.ResourceQuota{}
		if err := e.rt.client.Get(ctx, client.ObjectKeyFromObject(quota), current); err != nil {
			return false, err
		}
		return quotaInitialized(current, quota.Spec.Hard), nil
	}); err != nil {
		e.tb.Fatalf("wait for cluster-test workload quota: %v", err)
	}
}

func workloadBlockQuota(namespace string) *corev1.ResourceQuota {
	return &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: workloadBlockQuotaName, Namespace: namespace},
		Spec: corev1.ResourceQuotaSpec{Hard: corev1.ResourceList{
			corev1.ResourceName("count/replicasets.apps"): resource.MustParse("0"),
			corev1.ResourcePods:                           resource.MustParse("0"),
		}},
	}
}

func quotaInitialized(quota *corev1.ResourceQuota, expected corev1.ResourceList) bool {
	for name, expectedQuantity := range expected {
		actualQuantity, found := quota.Status.Hard[name]
		if !found || actualQuantity.Cmp(expectedQuantity) != 0 {
			return false
		}
	}
	return true
}

// StartManager starts one namespace-scoped controller manager configured by setup.
func (e *TestEnv) StartManager(setup func(ctrl.Manager) error) {
	e.tb.Helper()
	e.mu.Lock()
	if e.managerStarted {
		e.mu.Unlock()
		e.tb.Fatal("clusterenv: StartManager may only be called once per test")
	}
	e.managerStarted = true
	e.mu.Unlock()

	skipNameValidation := true
	mgr, err := ctrl.NewManager(e.rt.config, ctrl.Options{
		Scheme:                 e.rt.scheme,
		Metrics:                metricsserver.Options{BindAddress: "0"},
		HealthProbeBindAddress: "0",
		Controller:             controllerconfig.Controller{SkipNameValidation: &skipNameValidation},
		Cache: cache.Options{DefaultNamespaces: map[string]cache.Config{
			e.namespace: {},
		}},
	})
	if err != nil {
		e.tb.Fatalf("create cluster-test manager: %v", err)
	}
	if err := setup(mgr); err != nil {
		e.tb.Fatalf("setup cluster-test manager: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- mgr.Start(ctx)
	}()

	syncCtx, syncCancel := context.WithTimeout(context.Background(), e.opts.EventuallyTimeout)
	defer syncCancel()
	if !mgr.GetCache().WaitForCacheSync(syncCtx) {
		cancel()
		if managerErr := <-done; managerErr != nil && !errors.Is(managerErr, context.Canceled) {
			e.tb.Fatalf("cluster-test manager exited before cache sync: %v", managerErr)
		}
		e.tb.Fatal("cluster-test manager cache did not sync")
	}

	e.mu.Lock()
	e.managerCancel = cancel
	e.managerDone = done
	e.mu.Unlock()
}

func (e *TestEnv) stop() {
	e.tb.Helper()
	if e.tb.Failed() {
		e.logDiagnostics()
	}
	if err := e.deleteNamespace(); err != nil {
		e.tb.Errorf("clean cluster-test namespace %q: %v", e.namespace, err)
	}
	if err := e.stopManager(); err != nil {
		e.tb.Errorf("stop cluster-test manager: %v", err)
	}
}

func (e *TestEnv) logDiagnostics() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	jobs, err := e.rt.kubeClient.BatchV1().Jobs(e.namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		e.tb.Logf("list cluster-test Jobs for diagnostics: %v", err)
	} else {
		for _, job := range jobs.Items {
			e.tb.Logf("Job %s: active=%d succeeded=%d failed=%d conditions=%v",
				job.Name, job.Status.Active, job.Status.Succeeded, job.Status.Failed, job.Status.Conditions)
		}
	}

	pods, err := e.rt.kubeClient.CoreV1().Pods(e.namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		e.tb.Logf("list cluster-test Pods for diagnostics: %v", err)
	} else {
		tailLines := int64(200)
		for _, pod := range pods.Items {
			e.tb.Logf("Pod %s: phase=%s reason=%s message=%s", pod.Name, pod.Status.Phase, pod.Status.Reason, pod.Status.Message)
			for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
				logs, logErr := e.rt.kubeClient.CoreV1().Pods(e.namespace).GetLogs(pod.Name, &corev1.PodLogOptions{
					Container: container.Name,
					TailLines: &tailLines,
				}).DoRaw(ctx)
				if logErr != nil {
					e.tb.Logf("Pod %s/%s logs unavailable: %v", pod.Name, container.Name, logErr)
					continue
				}
				e.tb.Logf("Pod %s/%s logs:\n%s", pod.Name, container.Name, logs)
			}
		}
	}

	events, err := e.rt.kubeClient.CoreV1().Events(e.namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		e.tb.Logf("list cluster-test Events for diagnostics: %v", err)
		return
	}
	for _, event := range events.Items {
		e.tb.Logf("Event %s/%s: %s: %s", event.InvolvedObject.Kind, event.InvolvedObject.Name, event.Reason, event.Message)
	}
}

func (e *TestEnv) deleteNamespace() error {
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: e.namespace}}
	if err := e.rt.client.Delete(context.Background(), ns); err != nil && !apierrors.IsNotFound(err) {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), e.opts.NamespaceDeleteTimeout)
	defer cancel()
	return wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		err := e.rt.client.Get(ctx, client.ObjectKey{Name: e.namespace}, &corev1.Namespace{})
		switch {
		case apierrors.IsNotFound(err):
			return true, nil
		case err != nil:
			return false, err
		default:
			return false, nil
		}
	})
}

func (e *TestEnv) stopManager() error {
	e.mu.Lock()
	cancel := e.managerCancel
	done := e.managerDone
	e.mu.Unlock()
	if cancel == nil {
		return nil
	}
	cancel()
	if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
		return err
	}
	return nil
}

func normalizeOptions(opts Options) Options {
	if opts.WebhookProxyImage == "" {
		opts.WebhookProxyImage = "python:3.12-alpine"
	}
	if opts.EventuallyTimeout == 0 {
		opts.EventuallyTimeout = 30 * time.Second
	}
	if opts.NamespaceDeleteTimeout == 0 {
		opts.NamespaceDeleteTimeout = 2 * time.Minute
	}
	return opts
}
