/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/portforward"
	transportspdy "k8s.io/client-go/transport/spdy"
	"k8s.io/utils/ptr"
)

const (
	webhookServiceName = "clusterenv-webhook"
	proxyPodName       = "webhook-proxy"
	proxyServicePort   = 9443
	proxyControlPort   = 9444
	proxyHealthPort    = 9445
	proxyBridgeCount   = 64
)

const proxyScript = `import asyncio

waiting = asyncio.Queue()

async def client_connection(reader, writer):
    done = asyncio.get_running_loop().create_future()
    await waiting.put((reader, writer, done))
    try:
        await done
    finally:
        writer.close()
        await writer.wait_closed()

async def copy(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass

async def tunnel_connection(reader, writer):
    try:
        writer.write(b"\x01")
        await writer.drain()
        client_reader, client_writer, done = await waiting.get()
        writer.write(b"\x02")
        await writer.drain()
        await asyncio.gather(
            copy(client_reader, writer),
            copy(reader, client_writer),
        )
    finally:
        writer.close()
        await writer.wait_closed()
        if 'done' in locals() and not done.done():
            done.set_result(None)

async def health_connection(reader, writer):
    writer.close()
    await writer.wait_closed()

async def main():
    clients = await asyncio.start_server(client_connection, "0.0.0.0", 9443)
    tunnels = await asyncio.start_server(tunnel_connection, "0.0.0.0", 9444)
    health = await asyncio.start_server(health_connection, "0.0.0.0", 9445)
    async with clients, tunnels, health:
        await asyncio.gather(clients.serve_forever(), tunnels.serve_forever(), health.serve_forever())

asyncio.run(main())
`

type proxyRuntime struct {
	config     *rest.Config
	kubeClient kubernetes.Interface
	namespace  string
	service    string

	forwardStop chan struct{}
	forwardDone chan struct{}
	forwardErr  error
	forwardPort uint16
	bridgeStop  context.CancelFunc
	bridgeDone  sync.WaitGroup
}

func startProxy(ctx context.Context, config *rest.Config, kubeClient kubernetes.Interface, image string, timeout time.Duration) (*proxyRuntime, error) {
	namespace, err := kubeClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "clusterenv-webhook-"},
	}, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("create webhook proxy namespace: %w", err)
	}
	proxy := &proxyRuntime{config: config, kubeClient: kubeClient, namespace: namespace.Name, service: webhookServiceName}
	succeeded := false
	defer func() {
		if !succeeded {
			_ = proxy.stop()
		}
	}()

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: proxyPodName, Namespace: namespace.Name},
		Data:       map[string]string{"proxy.py": proxyScript},
	}
	if _, err := kubeClient.CoreV1().ConfigMaps(namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create webhook proxy script: %w", err)
	}
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: webhookServiceName, Namespace: namespace.Name},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": proxyPodName},
			Ports: []corev1.ServicePort{{
				Name: "https", Port: 443, TargetPort: intstr.FromInt32(proxyServicePort),
			}},
		},
	}
	if _, err := kubeClient.CoreV1().Services(namespace.Name).Create(ctx, service, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create webhook proxy service: %w", err)
	}
	mode := int32(0o555)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: proxyPodName, Namespace: namespace.Name, Labels: map[string]string{"app": proxyPodName}},
		Spec: corev1.PodSpec{
			RestartPolicy:                 corev1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: ptr.To[int64](0),
			Containers: []corev1.Container{{
				Name: "proxy", Image: image, ImagePullPolicy: corev1.PullIfNotPresent,
				Command: []string{"python", "/proxy/proxy.py"},
				Ports: []corev1.ContainerPort{
					{Name: "https", ContainerPort: proxyServicePort},
					{Name: "control", ContainerPort: proxyControlPort},
					{Name: "health", ContainerPort: proxyHealthPort},
				},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler:  corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromString("health")}},
					PeriodSeconds: 1,
				},
				VolumeMounts: []corev1.VolumeMount{{Name: "script", MountPath: "/proxy", ReadOnly: true}},
			}},
			Volumes: []corev1.Volume{{Name: "script", VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: proxyPodName}, DefaultMode: &mode},
			}}},
		},
	}
	if _, err := kubeClient.CoreV1().Pods(namespace.Name).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create webhook proxy pod: %w", err)
	}

	waitCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if err := wait.PollUntilContextCancel(waitCtx, 200*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		current, err := kubeClient.CoreV1().Pods(namespace.Name).Get(ctx, proxyPodName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		podReady := false
		for _, condition := range current.Status.Conditions {
			if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
				podReady = true
				break
			}
		}
		if !podReady {
			return false, nil
		}
		endpointSlices, err := kubeClient.DiscoveryV1().EndpointSlices(namespace.Name).List(ctx, metav1.ListOptions{
			LabelSelector: discoveryv1.LabelServiceName + "=" + webhookServiceName,
		})
		if err != nil {
			return false, err
		}
		for _, endpointSlice := range endpointSlices.Items {
			for _, endpoint := range endpointSlice.Endpoints {
				if len(endpoint.Addresses) > 0 && (endpoint.Conditions.Ready == nil || *endpoint.Conditions.Ready) {
					return true, nil
				}
			}
		}
		return false, nil
	}); err != nil {
		return nil, fmt.Errorf("wait for webhook proxy service endpoint: %w", err)
	}
	if err := proxy.startPortForward(waitCtx); err != nil {
		return nil, err
	}
	succeeded = true
	return proxy, nil
}

func (p *proxyRuntime) startPortForward(ctx context.Context) error {
	roundTripper, upgrader, err := transportspdy.RoundTripperFor(p.config)
	if err != nil {
		return fmt.Errorf("create webhook proxy port-forward transport: %w", err)
	}
	serverURL, err := url.Parse(p.config.Host)
	if err != nil {
		return fmt.Errorf("parse API server URL: %w", err)
	}
	serverURL.Path = fmt.Sprintf("/api/v1/namespaces/%s/pods/%s/portforward", p.namespace, proxyPodName)
	dialer := transportspdy.NewDialer(upgrader, &http.Client{Transport: roundTripper}, http.MethodPost, serverURL)
	forwardStop := make(chan struct{})
	ready := make(chan struct{})
	forwardDone := make(chan struct{})
	var output bytes.Buffer
	forwarder, err := portforward.NewOnAddresses(
		dialer, []string{"127.0.0.1"}, []string{fmt.Sprintf("0:%d", proxyControlPort)},
		forwardStop, ready, &output, &output,
	)
	if err != nil {
		return fmt.Errorf("create webhook proxy port-forward: %w", err)
	}
	p.forwardStop = forwardStop
	p.forwardDone = forwardDone
	go func() {
		p.forwardErr = forwarder.ForwardPorts()
		close(forwardDone)
	}()
	select {
	case <-ready:
	case <-p.forwardDone:
		if p.forwardErr == nil {
			return fmt.Errorf("webhook proxy port-forward exited before becoming ready: %s", output.String())
		}
		return fmt.Errorf("start webhook proxy port-forward: %w: %s", p.forwardErr, output.String())
	case <-ctx.Done():
		return fmt.Errorf("start webhook proxy port-forward: %w: %s", ctx.Err(), output.String())
	}
	ports, err := forwarder.GetPorts()
	if err != nil || len(ports) != 1 {
		return fmt.Errorf("get webhook proxy forwarded port: ports=%v: %w", ports, err)
	}
	p.forwardPort = ports[0].Local
	return nil
}

func (p *proxyRuntime) startBridge(target string, timeout time.Duration) error {
	ctx, cancel := context.WithCancel(context.Background())
	p.bridgeStop = cancel
	ready := make(chan struct{}, 1)
	for range proxyBridgeCount {
		p.bridgeDone.Add(1)
		go func() {
			defer p.bridgeDone.Done()
			p.bridge(ctx, target, ready)
		}()
	}
	select {
	case <-ready:
		return nil
	case <-time.After(timeout):
		return errors.New("wait for webhook proxy reverse tunnel: timed out")
	}
}

func (p *proxyRuntime) bridge(ctx context.Context, target string, ready chan<- struct{}) {
	controlAddress := net.JoinHostPort("127.0.0.1", strconv.Itoa(int(p.forwardPort)))
	for ctx.Err() == nil {
		control, err := net.DialTimeout("tcp", controlAddress, time.Second)
		if err != nil {
			select {
			case <-ctx.Done():
				return
			case <-time.After(100 * time.Millisecond):
				continue
			}
		}
		if err := waitForProxySignal(ctx, control, 1); err != nil {
			_ = control.Close()
			continue
		}
		select {
		case ready <- struct{}{}:
		default:
		}
		if err := waitForProxySignal(ctx, control, 2); err != nil {
			_ = control.Close()
			continue
		}
		webhook, err := net.DialTimeout("tcp", target, time.Second)
		if err != nil {
			_ = control.Close()
			continue
		}
		proxyConnections(control, webhook)
	}
}

func waitForProxySignal(ctx context.Context, control net.Conn, expected byte) error {
	ready := []byte{0}
	for ctx.Err() == nil {
		if err := control.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
			return err
		}
		if _, err := io.ReadFull(control, ready); err != nil {
			if networkError, ok := err.(net.Error); ok && networkError.Timeout() {
				continue
			}
			return err
		}
		if ready[0] != expected {
			return fmt.Errorf("unexpected webhook proxy signal %d, want %d", ready[0], expected)
		}
		return control.SetReadDeadline(time.Time{})
	}
	return ctx.Err()
}

func proxyConnections(left, right net.Conn) {
	var copies sync.WaitGroup
	copies.Add(2)
	go func() {
		defer copies.Done()
		_, _ = io.Copy(left, right)
		_ = left.SetDeadline(time.Now())
	}()
	go func() {
		defer copies.Done()
		_, _ = io.Copy(right, left)
		_ = right.SetDeadline(time.Now())
	}()
	copies.Wait()
	_ = left.Close()
	_ = right.Close()
}

func (p *proxyRuntime) stop() error {
	var errs []error
	if p.bridgeStop != nil {
		p.bridgeStop()
	}
	if p.forwardStop != nil {
		close(p.forwardStop)
		<-p.forwardDone
		if p.forwardErr != nil && !errors.Is(p.forwardErr, context.Canceled) {
			errs = append(errs, fmt.Errorf("stop webhook proxy port-forward: %w", p.forwardErr))
		}
	}
	p.bridgeDone.Wait()
	if p.namespace != "" {
		propagation := metav1.DeletePropagationBackground
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		err := p.kubeClient.CoreV1().Namespaces().Delete(ctx, p.namespace, metav1.DeleteOptions{PropagationPolicy: &propagation})
		if err != nil && !apierrors.IsNotFound(err) {
			errs = append(errs, fmt.Errorf("delete webhook proxy namespace: %w", err))
		} else if err := wait.PollUntilContextCancel(ctx, 500*time.Millisecond, true, func(ctx context.Context) (bool, error) {
			_, err := p.kubeClient.CoreV1().Namespaces().Get(ctx, p.namespace, metav1.GetOptions{})
			switch {
			case apierrors.IsNotFound(err):
				return true, nil
			case err != nil:
				return false, err
			default:
				return false, nil
			}
		}); err != nil {
			errs = append(errs, fmt.Errorf("wait for webhook proxy namespace deletion: %w", err))
		}
	}
	return errors.Join(errs...)
}
