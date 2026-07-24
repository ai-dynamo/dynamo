/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/webhookconfig"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

var conversionCRDNames = []string{
	"dynamocomponentdeployments.nvidia.com",
	"dynamographdeployments.nvidia.com",
	"dynamographdeploymentrequests.nvidia.com",
	"dynamographdeploymentscalingadapters.nvidia.com",
}

type webhookRuntime struct {
	kubeClient kubernetes.Interface
	crdClient  apiextensionsclient.Interface
	proxy      *proxyRuntime
	cert       *servingCertificate
	cancel     context.CancelFunc
	done       chan struct{}
	managerErr error

	mutatingNames   []string
	validatingNames []string
	conversions     map[string]*apiextensionsv1.CustomResourceConversion
}

func startWebhookRuntime(opts Options, config *rest.Config, kubeClient kubernetes.Interface) (*webhookRuntime, error) {
	ctx := context.Background()
	crdClient, err := apiextensionsclient.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("create CRD client: %w", err)
	}
	runtime := &webhookRuntime{kubeClient: kubeClient, crdClient: crdClient}
	succeeded := false
	defer func() {
		if !succeeded {
			_ = runtime.stop()
		}
	}()

	runtime.proxy, err = startProxy(ctx, config, kubeClient, opts.WebhookProxyImage, opts.EventuallyTimeout)
	if err != nil {
		return nil, err
	}
	runtime.cert, err = newServingCertificate(runtime.proxy.service, runtime.proxy.namespace)
	if err != nil {
		return nil, err
	}
	port, err := unusedLocalPort()
	if err != nil {
		return nil, err
	}
	server := webhook.NewServer(webhook.Options{Host: "127.0.0.1", Port: port, CertDir: runtime.cert.directory})
	mgr, err := ctrl.NewManager(config, ctrl.Options{
		Scheme: opts.Scheme, Metrics: metricsserver.Options{BindAddress: "0"}, WebhookServer: server,
	})
	if err != nil {
		return nil, fmt.Errorf("create cluster-test webhook manager: %w", err)
	}
	if err := opts.SetupWebhooks(mgr); err != nil {
		return nil, fmt.Errorf("setup cluster-test webhooks: %w", err)
	}
	managerContext, cancel := context.WithCancel(context.Background())
	runtime.cancel = cancel
	runtime.done = make(chan struct{})
	go func() {
		runtime.managerErr = mgr.Start(managerContext)
		close(runtime.done)
	}()
	if err := waitForWebhookServer(managerContext, server, runtime, opts.EventuallyTimeout); err != nil {
		return nil, err
	}
	if err := runtime.proxy.startBridge(net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), opts.EventuallyTimeout); err != nil {
		return nil, err
	}
	if err := runtime.installRegistrations(ctx, opts.AdditionalAdmission); err != nil {
		return nil, err
	}
	succeeded = true
	return runtime, nil
}

func unusedLocalPort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("reserve local webhook port: %w", err)
	}
	defer func() { _ = listener.Close() }()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

func waitForWebhookServer(
	ctx context.Context,
	server webhook.Server,
	runtime *webhookRuntime,
	timeout time.Duration,
) error {
	waitCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	started := server.StartedChecker()
	for {
		select {
		case <-runtime.done:
			if runtime.managerErr == nil {
				return errors.New("cluster-test webhook manager exited before serving")
			}
			return fmt.Errorf("cluster-test webhook manager exited before serving: %w", runtime.managerErr)
		case <-waitCtx.Done():
			return fmt.Errorf("wait for cluster-test webhook server: %w", waitCtx.Err())
		case <-ticker.C:
			if err := started((*http.Request)(nil)); err == nil {
				return nil
			}
		}
	}
}

func (e *webhookRuntime) installRegistrations(ctx context.Context, additional webhookconfig.Configurations) error {
	mutating, validating, err := webhookconfig.HelmConfigurations()
	if err != nil {
		return err
	}
	mutating = append(mutating, additional.Mutating...)
	validating = append(validating, additional.Validating...)
	suffix := rand.String(8)
	for _, configuration := range mutating {
		configuration = configuration.DeepCopy()
		configuration.Name += "-clusterenv-" + suffix
		for i := range configuration.Webhooks {
			if err := pointAdmissionAtProxy(&configuration.Webhooks[i].ClientConfig, e.proxy, e.cert.caBundle); err != nil {
				return fmt.Errorf("configure mutating webhook %q: %w", configuration.Webhooks[i].Name, err)
			}
		}
		created, err := e.kubeClient.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(ctx, configuration, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("create mutating webhook configuration: %w", err)
		}
		e.mutatingNames = append(e.mutatingNames, created.Name)
	}
	for _, configuration := range validating {
		configuration = configuration.DeepCopy()
		configuration.Name += "-clusterenv-" + suffix
		for i := range configuration.Webhooks {
			if err := pointAdmissionAtProxy(&configuration.Webhooks[i].ClientConfig, e.proxy, e.cert.caBundle); err != nil {
				return fmt.Errorf("configure validating webhook %q: %w", configuration.Webhooks[i].Name, err)
			}
		}
		created, err := e.kubeClient.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(ctx, configuration, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("create validating webhook configuration: %w", err)
		}
		e.validatingNames = append(e.validatingNames, created.Name)
	}

	e.conversions = map[string]*apiextensionsv1.CustomResourceConversion{}
	for _, name := range conversionCRDNames {
		crd, err := e.crdClient.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("get conversion CRD %q: %w", name, err)
		}
		if crd.Spec.Conversion == nil || crd.Spec.Conversion.Webhook == nil {
			return fmt.Errorf("conversion CRD %q has no webhook conversion", name)
		}
		e.conversions[name] = crd.Spec.Conversion.DeepCopy()
		path := crd.Spec.Conversion.Webhook.ClientConfig.Service.Path
		port := int32(443)
		crd.Spec.Conversion.Webhook.ClientConfig.CABundle = e.cert.caBundle
		crd.Spec.Conversion.Webhook.ClientConfig.Service = &apiextensionsv1.ServiceReference{
			Namespace: e.proxy.namespace, Name: e.proxy.service, Path: path, Port: &port,
		}
		if _, err := e.crdClient.ApiextensionsV1().CustomResourceDefinitions().Update(ctx, crd, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("point conversion CRD %q at cluster-test webhook: %w", name, err)
		}
	}
	return nil
}

func pointAdmissionAtProxy(config *admissionregistrationv1.WebhookClientConfig, proxy *proxyRuntime, caBundle []byte) error {
	if config.Service == nil {
		return errors.New("service reference is required")
	}
	path := config.Service.Path
	port := int32(443)
	config.URL = nil
	config.CABundle = caBundle
	config.Service = &admissionregistrationv1.ServiceReference{
		Namespace: proxy.namespace, Name: proxy.service, Path: path, Port: &port,
	}
	return nil
}

func (e *webhookRuntime) stop() error {
	var errs []error
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, name := range e.mutatingNames {
		if err := e.kubeClient.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(ctx, name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			errs = append(errs, fmt.Errorf("delete mutating webhook configuration %q: %w", name, err))
		}
	}
	for _, name := range e.validatingNames {
		if err := e.kubeClient.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(ctx, name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			errs = append(errs, fmt.Errorf("delete validating webhook configuration %q: %w", name, err))
		}
	}
	for name, conversion := range e.conversions {
		crd, err := e.crdClient.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			if !apierrors.IsNotFound(err) {
				errs = append(errs, fmt.Errorf("get conversion CRD %q for restore: %w", name, err))
			}
			continue
		}
		crd.Spec.Conversion = conversion.DeepCopy()
		if _, err := e.crdClient.ApiextensionsV1().CustomResourceDefinitions().Update(ctx, crd, metav1.UpdateOptions{}); err != nil {
			errs = append(errs, fmt.Errorf("restore conversion CRD %q: %w", name, err))
		}
	}
	if e.proxy != nil {
		if err := e.proxy.stop(); err != nil {
			errs = append(errs, err)
		}
	}
	if e.cancel != nil {
		e.cancel()
		<-e.done
		if e.managerErr != nil && !errors.Is(e.managerErr, context.Canceled) {
			errs = append(errs, fmt.Errorf("stop cluster-test webhook manager: %w", e.managerErr))
		}
	}
	if e.cert != nil {
		if err := e.cert.stop(); err != nil && !os.IsNotExist(err) {
			errs = append(errs, fmt.Errorf("remove cluster-test webhook certificates: %w", err))
		}
	}
	return errors.Join(errs...)
}
