// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package mcstest holds test helpers shared by the mcs-controller
// packages: a fakeregistry server with a client, a fake Kubernetes client with
// the controller's scheme, and object builders.
package mcstest

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/fakeregistry"
)

// Test identity.
const (
	ClusterSet = "cs"
	Cluster    = "a"
	Region     = "us-east-1"
)

// Identity is the identity of the cluster under test.
func Identity() identity.Identity {
	return identity.Identity{ClusterSet: ClusterSet, Cluster: Cluster, Region: Region}
}

// FastBackoff keeps client retries and reconnects quick.
var FastBackoff = client.Backoff{Initial: 5 * time.Millisecond, Max: 50 * time.Millisecond, Multiplier: 2, Jitter: 0.2}

// QuietLogger discards client logs.
var QuietLogger = slog.New(slog.NewTextHandler(io.Discard, nil))

// Ctx returns a context that ends with the test or after 30 s.
func Ctx(t testing.TB) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// Registry is a fakeregistry server on 127.0.0.1 with a client connected to it.
type Registry struct {
	Server  *fakeregistry.Server
	Running *fakeregistry.Running
	Client  *client.Client
}

// StartRegistry starts a fakeregistry server and a client with fast backoff.
func StartRegistry(t testing.TB, opts ...fakeregistry.Option) *Registry {
	t.Helper()
	srv := fakeregistry.New(opts...)
	run, err := srv.ServeLocal()
	require.NoError(t, err)
	t.Cleanup(run.Stop)
	return &Registry{Server: srv, Running: run, Client: NewClient(t, run.URL())}
}

// NewClient returns a registry client of url with fast backoff.
func NewClient(t testing.TB, url string) *client.Client {
	t.Helper()
	c, err := client.New(context.Background(), client.Config{
		URLs: []string{url}, AllowInsecureHTTP: true, Backoff: FastBackoff, Logger: QuietLogger,
		ConnectTimeout: time.Second, RPCTimeout: 5 * time.Second,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// FakeClient returns a fake Kubernetes client with the controller's scheme,
// the status subresources of ServiceExport and ServiceImport and objs.
func FakeClient(t testing.TB, objs ...ctrlclient.Object) ctrlclient.WithWatch {
	return FakeClientWith(t, interceptor.Funcs{}, objs...)
}

// FakeClientWith is FakeClient with interceptors.
func FakeClientWith(t testing.TB, funcs interceptor.Funcs, objs ...ctrlclient.Object) ctrlclient.WithWatch {
	t.Helper()
	scheme, err := config.NewScheme()
	require.NoError(t, err)
	return fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&mcsv1beta1.ServiceExport{}, &mcsv1beta1.ServiceImport{}).
		WithObjects(objs...).
		WithInterceptorFuncs(funcs).
		Build()
}

// Service returns a ClusterIP Service with one TCP port named http.
func Service(namespace, name string, labels map[string]string) *corev1.Service {
	return &corev1.Service{
		Namespace: namespace, Name: name, Labels: labels, UID: types.UID("uid-" + name),
		Spec: corev1.ServiceSpec{
			Type:      corev1.ServiceTypeClusterIP,
			ClusterIP: "10.0.0.10",
			Ports:     []corev1.ServicePort{{Name: "http", Protocol: corev1.ProtocolTCP, Port: 80}},
		},
	}
}

// ServiceExport returns a ServiceExport.
func ServiceExport(namespace, name string) *mcsv1beta1.ServiceExport {
	return &mcsv1beta1.ServiceExport{
		Namespace: namespace, Name: name, CreationTimestamp: metav1.NewTime(time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC))}
}

// EndpointSlice returns an IPv4 EndpointSlice of Service svc with one
// endpoint per address, ready, and the port http/8080.
func EndpointSlice(namespace, name, svc string, addresses ...string) *discoveryv1.EndpointSlice {
	s := &discoveryv1.EndpointSlice{
		Namespace: namespace, Name: name, Labels: map[string]string{discoveryv1.LabelServiceName: svc},
		AddressType: discoveryv1.AddressTypeIPv4,
		Ports:       []discoveryv1.EndpointPort{{Name: new("http"), Protocol: new(corev1.ProtocolTCP), Port: new(int32(8080))}},
	}
	for _, a := range addresses {
		s.Endpoints = append(s.Endpoints, discoveryv1.Endpoint{
			Addresses:  []string{a},
			Conditions: discoveryv1.EndpointConditions{Ready: new(true)},
		})
	}
	return s
}

// MetricValue returns the value of a counter or gauge, or the sample count
// of a histogram, with the labels in g.
func MetricValue(t testing.TB, g prometheus.Gatherer, name string, labels map[string]string) float64 {
	t.Helper()
	families, err := g.Gather()
	require.NoError(t, err)
	for _, mf := range families {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if matches(m, labels) {
				switch {
				case m.Counter != nil:
					return m.GetCounter().GetValue()
				case m.Gauge != nil:
					return m.GetGauge().GetValue()
				case m.Histogram != nil:
					return float64(m.GetHistogram().GetSampleCount())
				}
			}
		}
	}
	t.Fatalf("metric %s%v not found", name, labels)
	return 0
}

func matches(m *dto.Metric, labels map[string]string) bool {
	got := map[string]string{}
	for _, lp := range m.GetLabel() {
		got[lp.GetName()] = lp.GetValue()
	}
	for k, v := range labels {
		if got[k] != v {
			return false
		}
	}
	return true
}
