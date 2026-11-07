/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	internalcert "github.com/ai-dynamo/dynamo/deploy/operator/internal/cert"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/namespace_scope"
	"github.com/prometheus/client_golang/prometheus"
	coordinationv1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	clientfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

// TestRun_ReleasesRestrictedLeaseBeforeReturning exercises the production startup
// entry point, including lease acquisition, downstream failure, and fatal renewal.
func TestRun_ReleasesRestrictedLeaseBeforeReturning(t *testing.T) {
	tests := []struct {
		name         string
		fatalRenewal bool
	}{
		{name: "downstream startup failure"},
		{name: "unrecoverable renewal", fatalRenewal: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Given restricted-mode startup with a fake Kubernetes API and a real lease manager")
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			transport := &runLeaseTransport{
				client: fake.NewSimpleClientset(), namespace: "run-test",
				fatalRenewal: tt.fatalRenewal, deleted: make(chan struct{}),
			}
			fixture := &runTestManager{
				ctx: ctx, transport: transport, stopped: make(chan struct{}), release: make(chan struct{}),
				startupErr: errors.New("resource counter registration failed"),
				client:     clientfake.NewClientBuilder().WithScheme(crdScheme).Build(),
				config: &rest.Config{
					Host: "https://kubernetes.test", Transport: transport,
					ContentType: "application/json",
				},
			}
			defer fixture.releaseStartup()

			t.Log("And startup uses the test's configuration, constructors, flags, and metrics registry")
			cfg := &configv1alpha1.OperatorConfiguration{}
			configv1alpha1.SetDefaultsOperatorConfiguration(cfg)
			cfg.APIVersion = configv1alpha1.SchemeGroupVersion.String()
			cfg.Kind = "OperatorConfiguration"
			cfg.MPI.SSHSecretName = "mpi-ssh"
			cfg.MPI.SSHSecretNamespace = transport.namespace
			cfg.Namespace.Restricted = transport.namespace
			cfg.Namespace.Scope.LeaseDuration.Duration = 30 * time.Millisecond
			cfg.Namespace.Scope.LeaseRenewInterval.Duration = 10 * time.Millisecond
			cfg.Server.Webhook.CertProvisionMode = configv1alpha1.CertProvisionModeManual
			configBytes, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			configPath := filepath.Join(t.TempDir(), "operator.json")
			if err := os.WriteFile(configPath, configBytes, 0o600); err != nil {
				t.Fatal(err)
			}
			oldConfig, oldManager := loadKubernetesConfig, createOperatorManager
			oldCerts, oldSignals := createCertManager, setupOperatorSignals
			oldFlags, oldArgs, oldRegistry := flag.CommandLine, os.Args, ctrlmetrics.Registry
			t.Cleanup(func() {
				loadKubernetesConfig, createOperatorManager = oldConfig, oldManager
				createCertManager, setupOperatorSignals = oldCerts, oldSignals
				flag.CommandLine, os.Args, ctrlmetrics.Registry = oldFlags, oldArgs, oldRegistry
			})
			loadKubernetesConfig, createOperatorManager = fixture.kubernetesConfig, fixture.newManager
			createCertManager, setupOperatorSignals = fixture.newCertificates, fixture.signalContext
			flag.CommandLine = flag.NewFlagSet("operator", flag.ContinueOnError)
			os.Args = []string{"operator", "--config", configPath, "--operator-version", "1.0.0"}
			ctrlmetrics.Registry = prometheus.NewRegistry()

			t.Log("When run enters downstream startup and that startup fails or renewal becomes fatal")
			runDone := make(chan error, 1)
			go fixture.captureRunResult(runDone)
			select {
			case err = <-runDone:
			case <-ctx.Done():
				fixture.releaseStartup()
				select {
				case shutdownErr := <-runDone:
					t.Logf("run returned after releasing blocked startup: %v", shutdownErr)
				case <-time.After(time.Second):
				}
				t.Fatal("run did not bound restricted-mode startup shutdown")
			}
			if tt.fatalRenewal {
				if err == nil || !strings.Contains(err.Error(), "namespace scope marker lease is unrecoverable") ||
					!strings.Contains(err.Error(), "lease renewal failed 2 consecutive times") {
					t.Fatalf("run error = %v, want the lease manager's fatal renewal error", err)
				}
			} else if !errors.Is(err, fixture.startupErr) {
				t.Fatalf("run error = %v, want the downstream startup error", err)
			}

			t.Log("Then the startup worker finishes and it observed the marker lease before failing")
			select {
			case <-fixture.stopped:
			case <-time.After(time.Second):
				t.Fatal("startup worker did not finish after lease cleanup")
			}
			if !fixture.heldDuringStartup {
				t.Fatal("startup never held a marker lease; deletion would be a vacuous assertion")
			}

			t.Log("And the marker lease is gone before the returned error reaches main's exit boundary")
			_, err = transport.client.CoordinationV1().Leases(transport.namespace).Get(
				context.Background(), namespace_scope.LeaseName, metav1.GetOptions{},
			)
			if !apierrors.IsNotFound(err) {
				t.Fatalf("marker lease after run returned: %v, want NotFound", err)
			}
		})
	}
}

// runTestManager supplies the manager operations reached before downstream startup fails.
// Each test owns its clients and channels; the tests restore process globals and run serially.
type runTestManager struct {
	ctrl.Manager
	ctx               context.Context
	config            *rest.Config
	client            client.Client
	transport         *runLeaseTransport
	startupErr        error
	heldDuringStartup bool
	stopped           chan struct{}
	release           chan struct{}
	releaseOnce       sync.Once
}

// kubernetesConfig supplies the REST transport used by the real lease manager in run.
func (m *runTestManager) kubernetesConfig() *rest.Config { return m.config }

// signalContext supplies a cancellable context without installing process signal handlers.
func (m *runTestManager) signalContext() context.Context { return m.ctx }

// newManager verifies restricted cache scope and supplies the manager for startup.
func (m *runTestManager) newManager(_ *rest.Config, opts ctrl.Options) (ctrl.Manager, error) {
	if _, ok := opts.Cache.DefaultNamespaces[m.transport.namespace]; !ok {
		return nil, errors.New("run did not configure restricted namespace cache scope")
	}
	return m, nil
}

// newCertificates uses the real certificate manager with the test's explicit namespace.
func (m *runTestManager) newCertificates(
	cl client.Client, cfg *configv1alpha1.WebhookServer,
) (*internalcert.CertManager, error) {
	return internalcert.NewCertManagerInNamespace(cl, cfg, m.transport.namespace), nil
}

// GetConfig returns the REST configuration used by startup's real Kubernetes clients.
func (m *runTestManager) GetConfig() *rest.Config { return m.config }

// GetClient returns the fake manager client passed to the resource counter.
func (m *runTestManager) GetClient() client.Client { return m.client }

// Add fails the first downstream registration, or waits until fatal renewal causes cleanup.
func (m *runTestManager) Add(manager.Runnable) error {
	defer close(m.stopped)

	// Confirm that the production caller acquired a lease before registering its first runnable.
	_, err := m.transport.client.CoordinationV1().Leases(m.transport.namespace).Get(
		context.Background(), namespace_scope.LeaseName, metav1.GetOptions{},
	)
	if err != nil {
		return fmt.Errorf("startup ran without the marker lease: %w", err)
	}
	m.heldDuringStartup = true

	// Fatal renewal must bound this blocked startup; releasing it also prevents leaks on test failure.
	if m.transport.fatalRenewal {
		select {
		case <-m.transport.deleted:
		case <-m.release:
		}
		return context.Canceled
	}
	return m.startupErr
}

// releaseStartup unblocks the worker when a regression prevents normal cleanup.
func (m *runTestManager) releaseStartup() { m.releaseOnce.Do(func() { close(m.release) }) }

// captureRunResult delivers run's outcome through a buffered channel so the test can bound its wait.
func (m *runTestManager) captureRunResult(done chan<- error) { done <- run() }

// runLeaseTransport serves real typed-client lease requests from a fake Kubernetes clientset.
type runLeaseTransport struct {
	client       *fake.Clientset
	namespace    string
	fatalRenewal bool
	deleted      chan struct{}
	deleteOnce   sync.Once
}

// RoundTrip handles lease requests without opening sockets. req must be non-nil.
func (s *runLeaseTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !strings.HasPrefix(req.URL.Path, "/apis/coordination.k8s.io/v1/namespaces/"+s.namespace+"/leases") {
		return nil, fmt.Errorf("unexpected Kubernetes request: %s %s", req.Method, req.URL.Path)
	}
	if req.Body != nil {
		// The typed client sends an in-memory body whose close cannot fail.
		defer func() { _ = req.Body.Close() }()
	}

	// Apply each request to the same fake API that startup and the test observe.
	leases := s.client.CoordinationV1().Leases(s.namespace)
	var lease *coordinationv1.Lease
	var err error
	switch req.Method {
	case http.MethodGet:
		lease, err = leases.Get(req.Context(), namespace_scope.LeaseName, metav1.GetOptions{})
	case http.MethodPost, http.MethodPut:
		input := &coordinationv1.Lease{}
		if err := json.NewDecoder(req.Body).Decode(input); err != nil {
			return nil, err
		}
		if req.Method == http.MethodPost {
			lease, err = leases.Create(req.Context(), input, metav1.CreateOptions{})
		} else if s.fatalRenewal {
			err = errors.New("simulated renewal failure")
		} else {
			lease, err = leases.Update(req.Context(), input, metav1.UpdateOptions{})
		}
	case http.MethodDelete:
		options := metav1.DeleteOptions{}
		if err := json.NewDecoder(req.Body).Decode(&options); err != nil {
			return nil, err
		}
		err = leases.Delete(req.Context(), namespace_scope.LeaseName, options)
		if err == nil {
			s.deleteOnce.Do(func() { close(s.deleted) })
			return leaseJSONResponse(req, http.StatusOK, &metav1.Status{
				APIVersion: "v1", Kind: "Status", Status: metav1.StatusSuccess,
			})
		}
	default:
		return nil, fmt.Errorf("unexpected lease request method: %s", req.Method)
	}

	// Encode API errors so the real client preserves Kubernetes error classification.
	if err != nil {
		status := &metav1.Status{
			APIVersion: "v1", Kind: "Status", Status: metav1.StatusFailure,
			Code: http.StatusInternalServerError, Reason: metav1.StatusReasonInternalError, Message: err.Error(),
		}
		if apierrors.IsNotFound(err) {
			status.Code, status.Reason = http.StatusNotFound, metav1.StatusReasonNotFound
		}
		return leaseJSONResponse(req, int(status.Code), status)
	}
	lease.APIVersion, lease.Kind = coordinationv1.SchemeGroupVersion.String(), "Lease"
	return leaseJSONResponse(req, http.StatusOK, lease)
}

// leaseJSONResponse returns a Kubernetes JSON response for the real typed client. req must be non-nil.
func leaseJSONResponse(req *http.Request, code int, value any) (*http.Response, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return &http.Response{
		StatusCode: code, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(bytes.NewReader(payload)), Request: req,
	}, nil
}
