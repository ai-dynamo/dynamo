/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorenv

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	goruntime "runtime"
	"sync"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

// AdmissionWebhooks selects the Helm-rendered admission registrations installed in envtest.
type AdmissionWebhooks struct {
	Mutating   bool
	Validating bool
	// BypassUsers excludes named API users from validating admission.
	// It is intended for seeding legacy states that current admission rejects.
	BypassUsers []string
}

// WebhookSetupOptions contains the effective operator settings passed to SetupWebhooks.
type WebhookSetupOptions struct {
	OperatorConfig    *configv1alpha1.OperatorConfiguration
	RuntimeConfig     *commoncontroller.RuntimeConfig
	OperatorVersion   string
	OperatorPrincipal string
}

// WebhookSetupFunc registers the webhook handlers served by an Env.
type WebhookSetupFunc func(ctrl.Manager, WebhookSetupOptions) error

// Options configures an Env.
type Options struct {
	// Admission selects the Helm-rendered admission configurations to install.
	Admission AdmissionWebhooks
	// SetupWebhooks registers the handlers served by the environment.
	SetupWebhooks WebhookSetupFunc

	// OperatorVersion is passed to SetupWebhooks.
	OperatorVersion string
	// OperatorPrincipal is passed to SetupWebhooks.
	OperatorPrincipal string
	// Config overrides the default operator configuration.
	Config *configv1alpha1.OperatorConfiguration
	// RuntimeConfig overrides the default operator runtime configuration.
	RuntimeConfig *commoncontroller.RuntimeConfig

	// CRDDirectoryPaths overrides the default CRD directories loaded by envtest.
	CRDDirectoryPaths []string
	// BinaryAssetsDirectory overrides the envtest Kubernetes binary directory.
	BinaryAssetsDirectory string
	// EventuallyTimeout controls startup and cache synchronization timeouts.
	EventuallyTimeout time.Duration
}

// Env manages a shared or isolated envtest API server.
type Env struct {
	opts Options

	mu        sync.Mutex
	runM      bool
	shared    *runtimeEnv
	sharedErr error
	once      sync.Once
}

// New returns an Env configured with opts.
func New(opts Options) *Env {
	return &Env{opts: normalizeOptions(opts)}
}

// RunM runs m and stops the lazily created shared environment afterwards.
func (e *Env) RunM(m *testing.M) int {
	e.mu.Lock()
	e.runM = true
	e.mu.Unlock()

	code := m.Run()
	if err := e.stopShared(); err != nil {
		fmt.Fprintf(os.Stderr, "operatorenv: stop shared env: %v\n", err)
		if code == 0 {
			code = 1
		}
	}
	return code
}

// RunT starts an isolated environment that is stopped during test cleanup.
func (e *Env) RunT(tb testing.TB) *TestEnv {
	tb.Helper()
	rt, err := startRuntime(e.opts)
	if err != nil {
		tb.Fatalf("start isolated operatorenv: %v", err)
	}
	tb.Cleanup(func() {
		if err := rt.stop(); err != nil {
			tb.Errorf("stop isolated operatorenv: %v", err)
		}
	})
	return newTestEnv(tb, rt, e.opts)
}

// ForTest returns a test environment with a dedicated namespace backed by the shared API server.
// RunM must wrap the package test run before ForTest is called.
func (e *Env) ForTest(tb testing.TB) *TestEnv {
	tb.Helper()
	e.mu.Lock()
	runM := e.runM
	e.mu.Unlock()
	if !runM {
		tb.Fatal("operatorenv.ForTest requires RunM from TestMain; use RunT for an isolated per-test env")
	}
	e.once.Do(func() {
		e.shared, e.sharedErr = startRuntime(e.opts)
	})
	if e.sharedErr != nil {
		tb.Fatalf("start shared operatorenv: %v", e.sharedErr)
	}
	return newTestEnv(tb, e.shared, e.opts)
}

func (e *Env) stopShared() error {
	e.mu.Lock()
	shared := e.shared
	e.mu.Unlock()
	if shared == nil {
		return nil
	}
	return shared.stop()
}

type runtimeEnv struct {
	opts          Options
	env           *envtest.Environment
	config        *rest.Config
	scheme        *k8sruntime.Scheme
	client        client.Client
	operatorCfg   *configv1alpha1.OperatorConfiguration
	runtimeConfig *commoncontroller.RuntimeConfig
	cancel        context.CancelFunc
	done          chan error
}

func startRuntime(opts Options) (*runtimeEnv, error) {
	if opts.SetupWebhooks == nil {
		return nil, fmt.Errorf("operatorenv: SetupWebhooks is required")
	}
	scheme := newScheme()
	operatorCfg := defaultOperatorConfig(opts.Config)
	runtimeConfig := opts.RuntimeConfig
	if runtimeConfig == nil {
		runtimeConfig = defaultRuntimeConfig(operatorCfg)
	}
	webhookOptions, err := webhookInstallOptions(opts)
	if err != nil {
		return nil, err
	}
	testEnv := &envtest.Environment{
		Scheme:                scheme,
		CRDDirectoryPaths:     crdDirectoryPaths(opts),
		ErrorIfCRDPathMissing: false,
		BinaryAssetsDirectory: binaryAssetsDirectory(opts),
		WebhookInstallOptions: webhookOptions,
	}
	cfg, err := testEnv.Start()
	if err != nil {
		return nil, err
	}
	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		_ = testEnv.Stop()
		return nil, err
	}
	rt := &runtimeEnv{
		opts:          opts,
		env:           testEnv,
		config:        cfg,
		scheme:        scheme,
		client:        k8sClient,
		operatorCfg:   operatorCfg,
		runtimeConfig: runtimeConfig,
	}
	if err := rt.startWebhookManager(); err != nil {
		_ = testEnv.Stop()
		return nil, err
	}
	return rt, nil
}

func webhookInstallOptions(opts Options) (envtest.WebhookInstallOptions, error) {
	if !opts.Admission.Mutating && !opts.Admission.Validating {
		return envtest.WebhookInstallOptions{}, nil
	}
	mutating, validating, err := helmWebhookConfigurations()
	if err != nil {
		return envtest.WebhookInstallOptions{}, err
	}
	install := envtest.WebhookInstallOptions{}
	if opts.Admission.Mutating {
		install.MutatingWebhooks = mutating
	}
	if opts.Admission.Validating {
		addValidationBypassUsers(validating, opts.Admission.BypassUsers)
		install.ValidatingWebhooks = validating
	}
	return install, nil
}

func (e *runtimeEnv) startWebhookManager() error {
	ctx, cancel := context.WithCancel(context.Background())
	server := webhook.NewServer(webhook.Options{
		Host:    e.env.WebhookInstallOptions.LocalServingHost,
		Port:    e.env.WebhookInstallOptions.LocalServingPort,
		CertDir: e.env.WebhookInstallOptions.LocalServingCertDir,
	})
	mgr, err := ctrl.NewManager(e.config, ctrl.Options{
		Scheme:        e.scheme,
		Metrics:       metricsserver.Options{BindAddress: "0"},
		WebhookServer: server,
	})
	if err != nil {
		cancel()
		return err
	}
	if err := e.opts.SetupWebhooks(mgr, WebhookSetupOptions{
		OperatorConfig:    e.operatorCfg,
		RuntimeConfig:     e.runtimeConfig,
		OperatorVersion:   e.opts.OperatorVersion,
		OperatorPrincipal: e.opts.OperatorPrincipal,
	}); err != nil {
		cancel()
		return err
	}
	done := make(chan error, 1)
	go func() {
		done <- mgr.Start(ctx)
	}()
	managerStopped, err := waitForWebhookServer(ctx, mgr.GetWebhookServer(), done)
	if err != nil {
		cancel()
		if !managerStopped {
			if managerErr := <-done; managerErr != nil && !errors.Is(managerErr, context.Canceled) {
				return fmt.Errorf("webhook manager exited before server started: %w", managerErr)
			}
		}
		return err
	}
	e.cancel = cancel
	e.done = done
	return nil
}

func (e *runtimeEnv) stop() error {
	var errs []error
	if e.cancel != nil {
		e.cancel()
		if err := <-e.done; err != nil && !errors.Is(err, context.Canceled) {
			errs = append(errs, err)
		}
	}
	if e.env != nil {
		if err := e.env.Stop(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// TestEnv provides shared-cluster clients and a dedicated namespace for test-owned objects.
type TestEnv struct {
	tb        testing.TB
	rt        *runtimeEnv
	namespace string
	opts      Options
}

func newTestEnv(tb testing.TB, rt *runtimeEnv, opts Options) *TestEnv {
	tb.Helper()
	name := "operatorenv-" + rand.String(8)
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if err := rt.client.Create(context.Background(), ns); err != nil {
		tb.Fatalf("create test namespace %q: %v", name, err)
	}
	tb.Cleanup(func() {
		if err := rt.client.Delete(context.Background(), ns); err != nil && !apierrors.IsNotFound(err) {
			tb.Errorf("delete test namespace %q: %v", name, err)
		}
	})
	return &TestEnv{tb: tb, rt: rt, namespace: name, opts: opts}
}

// Namespace returns the namespace dedicated to this test.
func (e *TestEnv) Namespace() string {
	return e.namespace
}

// Client returns an unrestricted client for the envtest API server.
// Tests must use Namespace for namespaced fixtures to preserve isolation.
func (e *TestEnv) Client() client.Client {
	return e.rt.client
}

// RESTConfig returns a copy of the API server REST configuration.
func (e *TestEnv) RESTConfig() *rest.Config {
	return rest.CopyConfig(e.rt.config)
}

// AddUser provisions an authenticated envtest user and returns its REST configuration.
// Authorization must be granted separately so it does not alter the admission identity.
func (e *TestEnv) AddUser(user envtest.User) (*rest.Config, error) {
	authenticated, err := e.rt.env.AddUser(user, nil)
	if err != nil {
		return nil, err
	}
	return rest.CopyConfig(authenticated.Config()), nil
}

// OperatorConfig returns the environment's effective operator configuration.
func (e *TestEnv) OperatorConfig() *configv1alpha1.OperatorConfiguration {
	return e.rt.operatorCfg
}

// RuntimeConfig returns the environment's effective runtime configuration.
func (e *TestEnv) RuntimeConfig() *commoncontroller.RuntimeConfig {
	return e.rt.runtimeConfig
}

// ScaleClient returns a scale client configured for this envtest API server.
func (e *TestEnv) ScaleClient() (scale.ScalesGetter, error) {
	return newScaleClient(e.rt.config)
}

// StartManager starts a namespace-scoped controller manager configured by setup.
func (e *TestEnv) StartManager(setup func(ctrl.Manager) error) {
	e.tb.Helper()
	mgr, err := ctrl.NewManager(e.rt.config, ctrl.Options{
		Scheme:  e.rt.scheme,
		Metrics: metricsserver.Options{BindAddress: "0"},
		Cache: cache.Options{
			DefaultNamespaces: map[string]cache.Config{
				e.namespace: {},
			},
		},
	})
	if err != nil {
		e.tb.Fatalf("create manager: %v", err)
	}
	if err := setup(mgr); err != nil {
		e.tb.Fatalf("setup manager: %v", err)
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
		if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
			e.tb.Fatalf("manager exited before cache sync: %v", err)
		}
		e.tb.Fatal("manager cache did not sync")
	}
	e.tb.Cleanup(func() {
		cancel()
		if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
			e.tb.Errorf("manager stopped with error: %v", err)
		}
	})
}

func waitForWebhookServer(ctx context.Context, server webhook.Server, done <-chan error) (bool, error) {
	waitCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	startedChecker := server.StartedChecker()
	for {
		select {
		case err := <-done:
			return true, webhookManagerStartError(err)
		case <-waitCtx.Done():
			select {
			case err := <-done:
				return true, webhookManagerStartError(err)
			default:
				return false, waitCtx.Err()
			}
		case <-ticker.C:
			if err := startedChecker((*http.Request)(nil)); err == nil {
				return false, nil
			}
		}
	}
}

func webhookManagerStartError(err error) error {
	if err == nil {
		return errors.New("webhook manager exited before server started")
	}
	return fmt.Errorf("webhook manager exited before server started: %w", err)
}

func normalizeOptions(opts Options) Options {
	if opts.OperatorVersion == "" {
		opts.OperatorVersion = "1.0.0"
	}
	if opts.EventuallyTimeout == 0 {
		opts.EventuallyTimeout = 10 * time.Second
	}
	return opts
}

func defaultOperatorConfig(in *configv1alpha1.OperatorConfiguration) *configv1alpha1.OperatorConfiguration {
	cfg := &configv1alpha1.OperatorConfiguration{}
	if in != nil {
		*cfg = *in.DeepCopy()
	}
	configv1alpha1.SetDefaultsOperatorConfiguration(cfg)
	cfg.Server.Metrics.BindAddress = "0"
	cfg.Server.HealthProbe.BindAddress = "0"
	cfg.Server.Webhook.CertProvisionMode = configv1alpha1.CertProvisionModeManual
	if in == nil || in.GPU.DiscoveryEnabled == nil {
		cfg.GPU.DiscoveryEnabled = ptr.To(false)
	}
	return cfg
}

func defaultRuntimeConfig(cfg *configv1alpha1.OperatorConfiguration) *commoncontroller.RuntimeConfig {
	gate := features.Defaults()
	gate.Checkpoint = cfg.Checkpoint.Enabled
	gate.GPUDiscovery = cfg.Namespace.Restricted == "" || ptr.Deref(cfg.GPU.DiscoveryEnabled, true)
	return &commoncontroller.RuntimeConfig{Gate: gate}
}

func newScaleClient(config *rest.Config) (scale.ScalesGetter, error) {
	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	cachedDiscovery := memory.NewMemCacheClient(kubeClient.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)
	return scale.NewForConfig(
		config,
		restMapper,
		dynamic.LegacyAPIPathResolverFunc,
		scale.NewDiscoveryScaleKindResolver(cachedDiscovery),
	)
}

func crdDirectoryPaths(opts Options) []string {
	if len(opts.CRDDirectoryPaths) > 0 {
		return opts.CRDDirectoryPaths
	}
	root := operatorRoot()
	return []string{
		filepath.Join(root, "config", "crd", "bases"),
		filepath.Join(root, "internal", "controller", "testing", "prometheus"),
		filepath.Join(root, "internal", "controller", "testing", "volcano.sh"),
		filepath.Join(root, "internal", "controller", "testing", "run.ai"),
		filepath.Join(root, "internal", "controller", "testing", "nvidia"),
		filepath.Join(root, "internal", "controller", "testing", "inference.networking.k8s.io"),
		filepath.Join(root, "internal", "controller", "testing", "grove.io"),
	}
}

func binaryAssetsDirectory(opts Options) string {
	if opts.BinaryAssetsDirectory != "" {
		return opts.BinaryAssetsDirectory
	}
	if assets := os.Getenv("KUBEBUILDER_ASSETS"); assets != "" {
		return assets
	}
	return filepath.Join(operatorRoot(), "bin", "k8s", fmt.Sprintf("1.30.0-%s-%s", goruntime.GOOS, goruntime.GOARCH))
}

func operatorRoot() string {
	_, file, _, ok := goruntime.Caller(0)
	if !ok {
		panic("operatorenv: runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
}
