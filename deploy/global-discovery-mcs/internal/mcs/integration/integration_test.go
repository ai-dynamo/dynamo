// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package integration runs mcs-controller against a real API server
// started by controller-runtime's envtest and an in-process fakeregistry. The
// tests skip unless KUBEBUILDER_ASSETS points at kube-apiserver and etcd
// binaries; `make test-envtest` sets it. Every test runs its own
// controller, restricted to its own namespaces, against its own fakeregistry.
package integration

import (
	"context"
	"fmt"
	"os"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/testr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	eventsv1 "k8s.io/api/events/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/config/crd"
	registryconfig "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/app"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/server"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/fakeregistry"
)

const (
	waitTimeout  = 30 * time.Second
	pollInterval = 50 * time.Millisecond
	remote       = "b"
)

var (
	restConfig *rest.Config
	k8sClient  ctrlclient.Client
)

func TestMain(m *testing.M) {
	os.Exit(run(m))
}

func run(m *testing.M) int {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		return m.Run()
	}
	ctrl.SetLogger(logr.Discard())
	dir, err := os.MkdirTemp("", "mcs-crds-")
	if err != nil {
		fmt.Fprintf(os.Stderr, "creating the CRD directory: %v\n", err)
		return 1
	}
	defer func() { _ = os.RemoveAll(dir) }()
	if err := crd.WriteDir(dir); err != nil {
		fmt.Fprintf(os.Stderr, "writing CRDs: %v\n", err)
		return 1
	}
	env := &envtest.Environment{CRDDirectoryPaths: []string{dir}, ErrorIfCRDPathMissing: true}
	cfg, err := env.Start()
	if err != nil {
		fmt.Fprintf(os.Stderr, "starting envtest: %v\n", err)
		return 1
	}
	defer func() {
		if err := env.Stop(); err != nil {
			fmt.Fprintf(os.Stderr, "stopping envtest: %v\n", err)
		}
	}()
	scheme, err := config.NewScheme()
	if err != nil {
		fmt.Fprintf(os.Stderr, "building the scheme: %v\n", err)
		return 1
	}
	c, err := ctrlclient.New(cfg, ctrlclient.Options{Scheme: scheme})
	if err != nil {
		fmt.Fprintf(os.Stderr, "building the client: %v\n", err)
		return 1
	}
	restConfig, k8sClient = cfg, c
	return m.Run()
}

func requireEnv(t *testing.T) {
	t.Helper()
	if k8sClient == nil {
		t.Skip("KUBEBUILDER_ASSETS is not set; skipping the envtest suite")
	}
}

// controller is one running mcs-controller.
type controller struct {
	app     *app.App
	fakeReg *mcstest.Registry
}

// startController runs a controller watching namespaces against a
// fakeregistry, and stops it at the end of the test.
func startController(t *testing.T, fakeReg *mcstest.Registry, namespaces ...string) *controller {
	t.Helper()
	return &controller{app: startApp(t, fakeReg.Running.URL(), namespaces...), fakeReg: fakeReg}
}

// startApp runs a controller watching namespaces against the registry at url.
func startApp(t *testing.T, url string, namespaces ...string) *app.App {
	t.Helper()
	a, _ := startAppWith(t, url, nil, namespaces...)
	return a
}

// startAppWith is startApp with a configuration hook and a stop function;
// the controller also stops at the end of the test.
func startAppWith(t *testing.T, url string, configure func(*config.Config), namespaces ...string) (*app.App, func()) {
	t.Helper()
	cfg := config.Config{
		RegistryURLs:              []string{url},
		RegistryAllowInsecureHTTP: true,
		ClusterSetID:              mcstest.ClusterSet,
		ClusterID:                 mcstest.Cluster,
		IdentitySource:            config.IdentitySourceFlags,
		Region:                    mcstest.Region,
		LeaseTTL:                  config.DefaultLeaseTTL,
		WatchNamespaces:           namespaces,
		EndpointsPerSlice:         100,
		MaxConcurrentReconciles:   2,
		MetricsBindAddress:        "0",
		HealthProbeBindAddress:    "0",
		ImportTargetRef:           true,
		ExportResyncPeriod:        config.DefaultResyncPeriod,
		ImportResyncPeriod:        config.DefaultResyncPeriod,
	}
	if configure != nil {
		configure(&cfg)
	}
	require.NoError(t, cfg.Validate())
	a, err := app.New(context.Background(), cfg, restConfig, app.Options{
		Registry: prometheus.NewRegistry(),
		Logger:   testr.NewWithOptions(t, testr.Options{Verbosity: 0}),
		ManagerOptions: func(o *ctrl.Options) {
			o.Controller.SkipNameValidation = new(true)
		},
		AgentID: t.Name(),
	})
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- a.Run(ctx) }()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			cancel()
			select {
			case err := <-done:
				assert.NoError(t, err)
			case <-time.After(waitTimeout):
				t.Error("the controller did not stop")
			}
		})
	}
	t.Cleanup(stop)
	require.NoError(t, a.Lease.WaitRegistered(mcstest.Ctx(t)))
	return a, stop
}

func createNamespace(t *testing.T) string {
	t.Helper()
	name := "mcs-" + rand.String(8)
	require.NoError(t, k8sClient.Create(context.Background(), &corev1.Namespace{Name: name}))
	return name
}

func eventually(t *testing.T, cond func() bool, msg string, args ...any) {
	t.Helper()
	require.Eventually(t, cond, waitTimeout, pollInterval, append([]any{msg}, args...)...)
}

// registryWebExport returns the registry export of ns/web, or nil.
func registryWebExport(t *testing.T, c *controller, ns string) *mcsv1.ServiceExport {
	t.Helper()
	resp, err := c.fakeReg.Client.ServiceExports().List(mcstest.Ctx(t), &mcsv1.ListServiceExportsRequest{Clusterset: mcstest.ClusterSet, Cluster: mcstest.Cluster})
	require.NoError(t, err)
	for _, x := range resp.GetExports() {
		if x.GetNamespace() == ns && x.GetName() == "web" {
			return x
		}
	}
	return nil
}

func get(t *testing.T, ns, name string, obj ctrlclient.Object) bool {
	t.Helper()
	err := k8sClient.Get(context.Background(), types.NamespacedName{Namespace: ns, Name: name}, obj)
	if apierrors.IsNotFound(err) {
		return false
	}
	require.NoError(t, err)
	return true
}

func importedSlices(t *testing.T, ns, name string) []discoveryv1.EndpointSlice {
	t.Helper()
	var list discoveryv1.EndpointSliceList
	require.NoError(t, k8sClient.List(context.Background(), &list, ctrlclient.InNamespace(ns),
		ctrlclient.MatchingLabels{api.LabelMCSServiceName: name, api.LabelEndpointSliceManagedBy: api.EndpointSliceManagedBy}))
	sort.Slice(list.Items, func(i, j int) bool { return list.Items[i].Name < list.Items[j].Name })
	return list.Items
}

func webService(ns string) *corev1.Service {
	return &corev1.Service{
		Namespace: ns, Name: "web",
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": "web"},
			Ports:    []corev1.ServicePort{{Name: "http", Protocol: corev1.ProtocolTCP, Port: 80}},
		},
	}
}

func TestExportLifecycle(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	ns := createNamespace(t)
	c := startController(t, mcstest.StartRegistry(t), ns)

	require.NoError(t, k8sClient.Create(ctx, webService(ns)))
	slice := mcstest.EndpointSlice(ns, "web-1", "web", "10.1.0.1", "10.1.0.2")
	require.NoError(t, k8sClient.Create(ctx, slice))
	require.NoError(t, k8sClient.Create(ctx, &mcsv1beta1.ServiceExport{Namespace: ns, Name: "web"}))

	// Export creates Put and Replace.
	eventually(t, func() bool {
		x := registryWebExport(t, c, ns)
		return x != nil && x.GetEndpointCount() == 2
	}, "export with two endpoints in the registry")
	assert.GreaterOrEqual(t, c.fakeReg.Server.CallCount("PutServiceExport"), 1)
	assert.GreaterOrEqual(t, c.fakeReg.Server.CallCount("ReplaceEndpoints"), 1)
	first := registryWebExport(t, c, ns)
	assert.Equal(t, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, first.GetSpec().GetType())

	// This cluster's conditions land on the ServiceExport.
	var se mcsv1beta1.ServiceExport
	eventually(t, func() bool {
		get(t, ns, "web", &se)
		ready := meta.FindStatusCondition(se.Status.Conditions, api.ConditionReady)
		conflict := meta.FindStatusCondition(se.Status.Conditions, api.ConditionConflict)
		return ready != nil && ready.Status == metav1.ConditionTrue && conflict != nil && conflict.Status == metav1.ConditionFalse
	}, "Ready and Conflict copied from the registry")
	assert.Contains(t, se.Finalizers, api.FinalizerServiceExport)
	assert.True(t, meta.IsStatusConditionTrue(se.Status.Conditions, api.ConditionValid))

	// The exporting cluster imports its own export too.
	eventually(t, func() bool {
		s := importedSlices(t, ns, "web")
		return len(s) == 1 && s[0].Labels[api.LabelMCSSourceCluster] == mcstest.Cluster && len(s[0].Endpoints) == 2
	}, "local endpoints mirrored")

	// Scaling endpoints replaces the set with a higher generation.
	require.NoError(t, k8sClient.Get(ctx, ctrlclient.ObjectKeyFromObject(slice), slice))
	slice.Endpoints = append(slice.Endpoints, discoveryv1.Endpoint{Addresses: []string{"10.1.0.3"}})
	require.NoError(t, k8sClient.Update(ctx, slice))
	eventually(t, func() bool {
		x := registryWebExport(t, c, ns)
		return x != nil && x.GetEndpointCount() == 3 && x.GetEndpointsGeneration() > first.GetEndpointsGeneration()
	}, "endpoints replaced with a higher generation")

	// Deleting the ServiceExport deletes the registry record and the finalizer.
	require.NoError(t, k8sClient.Delete(ctx, &se))
	eventually(t, func() bool {
		return registryWebExport(t, c, ns) == nil && !get(t, ns, "web", &mcsv1beta1.ServiceExport{})
	}, "export withdrawn and ServiceExport gone")
	assert.GreaterOrEqual(t, c.fakeReg.Server.CallCount("DeleteServiceExport"), 1)
	eventually(t, func() bool { return !get(t, ns, "web", &mcsv1beta1.ServiceImport{}) }, "import of the withdrawn export deleted")
}

func remoteSpec() *mcsv1.ServiceExportSpec {
	return &mcsv1.ServiceExportSpec{
		Type:   mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
		Labels: map[string]string{api.Prefix + "tier": "gold"},
		Ports:  []*mcsv1.ServicePort{{Name: "http", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 80}},
	}
}

func remoteEndpoints(addrs ...string) []*mcsv1.Endpoint {
	eps := fakeregistry.ReadyEndpoints(addrs...)
	for _, ep := range eps {
		ep.Ports = []*mcsv1.EndpointPort{{Name: "http", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 8080}}
	}
	return eps
}

func TestImportFromAnotherCluster(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	ns := createNamespace(t)
	fakeReg := mcstest.StartRegistry(t)
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "frontend", remoteSpec(),
		fakeregistry.WithPodNames(remoteEndpoints("10.2.0.1", "10.2.0.2"), "frontend-0"))
	startController(t, fakeReg, ns)

	var si mcsv1beta1.ServiceImport
	derived := api.DerivedServiceName(ns, "frontend")
	var svc corev1.Service
	eventually(t, func() bool {
		return get(t, ns, "frontend", &si) && len(si.Spec.IPs) == 1 && get(t, ns, derived, &svc) && len(importedSlices(t, ns, "frontend")) == 1
	}, "ServiceImport, derived Service and slices")
	assert.Equal(t, api.ManagedByController, si.Labels[api.LabelManagedBy])
	assert.Equal(t, "gold", si.Labels[api.Prefix+"tier"])
	assert.Equal(t, derived, si.Annotations[api.AnnotationMCSDerivedService])
	assert.Equal(t, mcsv1beta1.ClusterSetIP, si.Spec.Type)
	assert.Equal(t, svc.Spec.ClusterIP, si.Spec.IPs[0], "spec.ips is the derived ClusterIP")
	eventually(t, func() bool {
		get(t, ns, "frontend", &si)
		return len(si.Status.Clusters) == 1 && si.Status.Clusters[0].Cluster == remote
	}, "status clusters")

	assert.Empty(t, svc.Spec.Selector)
	assert.Equal(t, int32(80), svc.Spec.Ports[0].Port)
	require.Len(t, svc.OwnerReferences, 1)
	assert.Equal(t, si.UID, svc.OwnerReferences[0].UID)

	s := importedSlices(t, ns, "frontend")[0]
	assert.Equal(t, derived+"-"+remote+"-v4-0", s.Name)
	assert.Equal(t, corev1.IPFamilyPolicySingleStack, *svc.Spec.IPFamilyPolicy, "the API server's choice for an import without families")
	assert.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol}, svc.Spec.IPFamilies)
	assert.Equal(t, derived, s.Labels[discoveryv1.LabelServiceName])
	assert.Equal(t, remote, s.Labels[api.LabelMCSSourceCluster])
	assert.Equal(t, mcstest.ClusterSet, s.Labels[api.LabelClusterSet])
	require.Len(t, s.OwnerReferences, 1)
	assert.Equal(t, si.UID, s.OwnerReferences[0].UID)
	require.Len(t, s.Endpoints, 2)
	require.NotNil(t, s.Endpoints[0].TargetRef, "the exported pod name is the imported target reference")
	assert.Equal(t, corev1.ObjectReference{Kind: "Pod", Name: "frontend-0"}, *s.Endpoints[0].TargetRef)
	assert.Nil(t, s.Endpoints[1].TargetRef)
	require.Len(t, s.Ports, 1)
	assert.Equal(t, int32(8080), *s.Ports[0].Port)

	// A managed object deleted by hand comes back.
	require.NoError(t, k8sClient.Delete(context.Background(), &s))
	eventually(t, func() bool {
		slices := importedSlices(t, ns, "frontend")
		return len(slices) == 1 && slices[0].UID != s.UID
	}, "deleted slice re-created")

	// Reconciliation is level-based: a managed object edited by hand is
	// restored although the registry did not change.
	s = importedSlices(t, ns, "frontend")[0]
	s.Endpoints[0].Addresses = []string{"192.0.2.1"}
	require.NoError(t, k8sClient.Update(ctx, &s))
	eventually(t, func() bool {
		slices := importedSlices(t, ns, "frontend")
		return len(slices) == 1 && slices[0].Endpoints[0].Addresses[0] == "10.2.0.1"
	}, "edited slice restored")
	require.NoError(t, k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: derived}, &svc))
	svc.Spec.Ports[0].Port = 9999
	require.NoError(t, k8sClient.Update(ctx, &svc))
	eventually(t, func() bool {
		get(t, ns, derived, &svc)
		return len(svc.Spec.Ports) == 1 && svc.Spec.Ports[0].Port == 80
	}, "edited derived Service restored without an extra port")
	require.NoError(t, k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "frontend"}, &si))
	si.Spec.Ports = []mcsv1beta1.ServicePort{{Name: "other", Protocol: corev1.ProtocolTCP, Port: 1}}
	require.NoError(t, k8sClient.Update(ctx, &si))
	eventually(t, func() bool {
		get(t, ns, "frontend", &si)
		return len(si.Spec.Ports) == 1 && si.Spec.Ports[0].Port == 80
	}, "edited ServiceImport restored")
	assert.Equal(t, api.ManagedByController, si.Labels[api.LabelManagedBy])

	// Deleting the export in the registry cascades.
	fakeReg.Server.DeleteExport(mcstest.ClusterSet, remote, ns, "frontend")
	eventually(t, func() bool {
		return !get(t, ns, "frontend", &mcsv1beta1.ServiceImport{}) && !get(t, ns, derived, &corev1.Service{}) &&
			len(importedSlices(t, ns, "frontend")) == 0
	}, "import, derived Service and slices deleted")
}

// Slice names carry the address family and index per family, so growth and
// shrink of the IPv4 endpoints across the slice boundary never renames the
// IPv6 slice: the API server rejects a change of the immutable addressType.
func TestDualStackSliceNamesAcrossTheBoundary(t *testing.T) {
	requireEnv(t)
	ns := createNamespace(t)
	fakeReg := mcstest.StartRegistry(t)
	v4 := func(n int) []string {
		addrs := make([]string, 0, n)
		for i := range n {
			addrs = append(addrs, fmt.Sprintf("10.7.%d.%d", i/256, i%256))
		}
		return addrs
	}
	with6 := func(addrs ...string) []*mcsv1.Endpoint { return remoteEndpoints(append(addrs, "fd00:7::1")...) }
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "frontend", remoteSpec(), with6(v4(100)...))
	startController(t, fakeReg, ns)
	derived := api.DerivedServiceName(ns, "frontend")
	names := func() []string {
		var out []string
		for _, s := range importedSlices(t, ns, "frontend") {
			out = append(out, s.Name)
		}
		return out
	}
	eventually(t, func() bool {
		return assert.ObjectsAreEqual([]string{derived + "-b-v4-0", derived + "-b-v6-0"}, names())
	}, "one IPv4 and one IPv6 slice: %v", names())
	v6UID := importedSlices(t, ns, "frontend")[1].UID

	fakeReg.Server.SetEndpoints(mcstest.ClusterSet, remote, ns, "frontend", with6(v4(101)...))
	eventually(t, func() bool {
		return assert.ObjectsAreEqual([]string{derived + "-b-v4-0", derived + "-b-v4-1", derived + "-b-v6-0"}, names())
	}, "the IPv4 endpoints spill into a second slice: %v", names())
	assert.Equal(t, v6UID, importedSlices(t, ns, "frontend")[2].UID, "the IPv6 slice is the same object")

	fakeReg.Server.SetEndpoints(mcstest.ClusterSet, remote, ns, "frontend", with6(v4(100)...))
	eventually(t, func() bool {
		return assert.ObjectsAreEqual([]string{derived + "-b-v4-0", derived + "-b-v6-0"}, names())
	}, "the second IPv4 slice is deleted: %v", names())

	// A family transition: the IPv6 endpoint moves to IPv4.
	fakeReg.Server.SetEndpoints(mcstest.ClusterSet, remote, ns, "frontend", remoteEndpoints(append(v4(100), "10.7.1.0")...))
	eventually(t, func() bool {
		return assert.ObjectsAreEqual([]string{derived + "-b-v4-0", derived + "-b-v4-1"}, names())
	}, "the IPv6 slice is gone and the endpoint is in an IPv4 slice: %v", names())
}

// The import's IP families describe the exporters: an IPv4-only importing
// cluster still gets a derived Service for a dual-stack or IPv6 import.
func TestIPFamiliesFollowTheImportingCluster(t *testing.T) {
	requireEnv(t)
	ns := createNamespace(t)
	fakeReg := mcstest.StartRegistry(t)
	dual := remoteSpec()
	dual.IpFamilies = []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV6, mcsv1.IPFamily_IP_FAMILY_IPV4}
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "dual", dual, remoteEndpoints("10.8.0.1"))
	six := remoteSpec()
	six.IpFamilies = []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV6}
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "six", six, remoteEndpoints("fd00:8::1"))
	c := startController(t, fakeReg, ns)
	require.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol}, c.app.ClusterIPFamilies, "envtest's API server is IPv4 only")

	var svc corev1.Service
	var si mcsv1beta1.ServiceImport
	eventually(t, func() bool {
		return get(t, ns, api.DerivedServiceName(ns, "dual"), &svc) && get(t, ns, "dual", &si) && len(si.Spec.IPs) == 1
	}, "dual-stack import gets a derived Service and spec.ips")
	assert.Equal(t, corev1.IPFamilyPolicyPreferDualStack, *svc.Spec.IPFamilyPolicy)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol}, svc.Spec.IPFamilies, "only the family the cluster has")
	assert.Equal(t, []corev1.IPFamily{corev1.IPv6Protocol, corev1.IPv4Protocol}, si.Spec.IPFamilies, "the ServiceImport keeps the exporters' families")
	eventually(t, func() bool {
		return get(t, ns, api.DerivedServiceName(ns, "six"), &svc) && get(t, ns, "six", &si) && len(si.Spec.IPs) == 1
	}, "IPv6 import gets an IPv4 derived Service, the API server's choice")
	assert.Equal(t, corev1.IPFamilyPolicySingleStack, *svc.Spec.IPFamilyPolicy)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol}, svc.Spec.IPFamilies)
	assert.Equal(t, svc.Spec.ClusterIP, si.Spec.IPs[0])
}

// An import blocked by a ServiceImport of another implementation is applied
// once that object goes away, without a registry change or a restart.
func TestForeignServiceImportReleasesTheImport(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	ns := createNamespace(t)
	foreign := &mcsv1beta1.ServiceImport{
		Namespace: ns, Name: "frontend", Labels: map[string]string{"owner": "someone-else"},
		Spec: mcsv1beta1.ServiceImportSpec{Type: mcsv1beta1.ClusterSetIP, Ports: []mcsv1beta1.ServicePort{{Name: "x", Protocol: corev1.ProtocolTCP, Port: 1}}},
	}
	require.NoError(t, k8sClient.Create(ctx, foreign))
	fakeReg := mcstest.StartRegistry(t)
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "frontend", remoteSpec(), remoteEndpoints("10.9.0.1"))
	startController(t, fakeReg, ns)
	derived := api.DerivedServiceName(ns, "frontend")
	time.Sleep(time.Second)
	assert.False(t, get(t, ns, derived, &corev1.Service{}), "blocked by the foreign ServiceImport")
	var si mcsv1beta1.ServiceImport
	require.True(t, get(t, ns, "frontend", &si))
	assert.Empty(t, si.Labels[api.LabelManagedBy])

	require.NoError(t, k8sClient.Delete(ctx, foreign))
	eventually(t, func() bool {
		return get(t, ns, "frontend", &si) && si.Labels[api.LabelManagedBy] == api.ManagedByController &&
			get(t, ns, derived, &corev1.Service{}) && len(importedSlices(t, ns, "frontend")) == 1
	}, "the import is applied once the foreign ServiceImport is deleted")
	assert.Equal(t, int32(80), si.Spec.Ports[0].Port)
}

// An import whose objects cannot be applied reports Ready=False with reason
// ApplyFailed and a Warning Event instead of retrying silently. The
// fakeregistry unions ports without the registry's merge rule, so two
// exports that mix an unnamed and a named port make an invalid Service.
func TestApplyFailureIsReported(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	ns := createNamespace(t)
	fakeReg := mcstest.StartRegistry(t)
	unnamed := remoteSpec()
	unnamed.Ports = []*mcsv1.ServicePort{{Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 80}}
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "frontend", unnamed, remoteEndpoints("10.10.0.1"))
	named := remoteSpec()
	named.Ports[0].Port = 8080
	fakeReg.Server.PutExport(mcstest.ClusterSet, "c", ns, "frontend", named, remoteEndpoints("10.10.0.2"))
	startController(t, fakeReg, ns)

	var si mcsv1beta1.ServiceImport
	eventually(t, func() bool {
		if !get(t, ns, "frontend", &si) {
			return false
		}
		ready := meta.FindStatusCondition(si.Status.Conditions, string(mcsv1beta1.ServiceImportConditionReady))
		return ready != nil && ready.Status == metav1.ConditionFalse && ready.Reason == api.ConditionReasonApplyFailed
	}, "Ready=False with reason ApplyFailed")
	ready := meta.FindStatusCondition(si.Status.Conditions, string(mcsv1beta1.ServiceImportConditionReady))
	assert.Contains(t, ready.Message, "derived Service")
	assert.False(t, get(t, ns, api.DerivedServiceName(ns, "frontend"), &corev1.Service{}), "no invalid Service was created")
	var events eventsv1.EventList
	eventually(t, func() bool {
		require.NoError(t, k8sClient.List(ctx, &events, ctrlclient.InNamespace(ns)))
		for _, ev := range events.Items {
			if ev.Reason == "ApplyFailed" && ev.Regarding.Name == "frontend" && ev.Type == corev1.EventTypeWarning {
				return true
			}
		}
		return false
	}, "Warning Event on the ServiceImport")

	// The exports agree again: the condition clears and the objects appear.
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "frontend", named, remoteEndpoints("10.10.0.1"))
	eventually(t, func() bool {
		get(t, ns, "frontend", &si)
		return meta.IsStatusConditionTrue(si.Status.Conditions, string(mcsv1beta1.ServiceImportConditionReady)) &&
			get(t, ns, api.DerivedServiceName(ns, "frontend"), &corev1.Service{})
	}, "Ready=True once the import is valid")
}

// Narrowing --watch-namespaces releases the ServiceExports and deletes the
// managed import objects in the namespaces no longer watched; the next
// resync removes their registry records.
func TestNarrowingWatchNamespaces(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	kept, dropped := createNamespace(t), createNamespace(t)
	fakeReg := mcstest.StartRegistry(t)
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, dropped, "frontend", remoteSpec(), remoteEndpoints("10.11.0.1"))
	_, stop := startAppWith(t, fakeReg.Running.URL(), nil, kept, dropped)
	for _, ns := range []string{kept, dropped} {
		require.NoError(t, k8sClient.Create(ctx, webService(ns)))
		require.NoError(t, k8sClient.Create(ctx, mcstest.EndpointSlice(ns, "web-1", "web", "10.11.1.1")))
		require.NoError(t, k8sClient.Create(ctx, &mcsv1beta1.ServiceExport{Namespace: ns, Name: "web"}))
	}
	c := &controller{fakeReg: fakeReg}
	var se mcsv1beta1.ServiceExport
	eventually(t, func() bool {
		return registryWebExport(t, c, kept) != nil && registryWebExport(t, c, dropped) != nil &&
			get(t, dropped, "web", &se) && len(se.Finalizers) == 1 &&
			len(importedSlices(t, dropped, "frontend")) == 1 && get(t, dropped, "frontend", &mcsv1beta1.ServiceImport{})
	}, "both namespaces exported, the dropped one imported")
	stop()

	startAppWith(t, fakeReg.Running.URL(), nil, kept)
	eventually(t, func() bool {
		return get(t, dropped, "web", &se) && len(se.Finalizers) == 0
	}, "the finalizer of the ServiceExport outside the watch set is removed")
	eventually(t, func() bool { return registryWebExport(t, c, dropped) == nil }, "its registry record is deleted by the resync")
	eventually(t, func() bool {
		return !get(t, dropped, "frontend", &mcsv1beta1.ServiceImport{}) &&
			!get(t, dropped, api.DerivedServiceName(dropped, "frontend"), &corev1.Service{}) &&
			len(importedSlices(t, dropped, "frontend")) == 0
	}, "the managed import objects outside the watch set are deleted")
	assert.NotNil(t, registryWebExport(t, c, kept), "the watched namespace is untouched")
	require.True(t, get(t, kept, "web", &se))
	assert.Contains(t, se.Finalizers, api.FinalizerServiceExport)
	require.NoError(t, k8sClient.Delete(ctx, &se))
	eventually(t, func() bool { return !get(t, kept, "web", &mcsv1beta1.ServiceExport{}) }, "the kept export can still be deleted")
}

func TestSnapshotAfterReconnectCollectsOrphans(t *testing.T) {
	requireEnv(t)
	ns := createNamespace(t)
	fakeReg := mcstest.StartRegistry(t)
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, ns, "orphan", remoteSpec(), remoteEndpoints("10.3.0.1"))
	c := startController(t, fakeReg, ns)
	eventually(t, func() bool { return get(t, ns, "orphan", &mcsv1beta1.ServiceImport{}) }, "orphan imported")

	// Another registry replica answers on the same address: its view epoch is
	// new, so the watch restarts with a snapshot that lacks the orphan.
	addr := fakeReg.Running.Addr()
	fakeReg.Running.Stop()
	replica := fakeregistry.New()
	replica.PutExport(mcstest.ClusterSet, remote, ns, "kept", remoteSpec(), remoteEndpoints("10.3.0.2"))
	var run *fakeregistry.Running
	eventually(t, func() bool {
		var err error
		run, err = replica.ServeAddr(addr)
		return err == nil
	}, "serve the replica on %s", addr)
	t.Cleanup(run.Stop)

	eventually(t, func() bool {
		return !get(t, ns, "orphan", &mcsv1beta1.ServiceImport{}) && get(t, ns, "kept", &mcsv1beta1.ServiceImport{}) &&
			len(importedSlices(t, ns, "orphan")) == 0
	}, "orphan collected after SNAPSHOT_END")
	eventually(t, c.app.Lease.Registered, "registered with the replica again")
}

func TestPendingNamespaceIsAppliedWhenCreated(t *testing.T) {
	requireEnv(t)
	late := "mcs-late-" + rand.String(6)
	fakeReg := mcstest.StartRegistry(t)
	fakeReg.Server.PutExport(mcstest.ClusterSet, remote, late, "frontend", remoteSpec(), remoteEndpoints("10.4.0.1"))
	c := startController(t, fakeReg, late)
	eventually(t, func() bool { return len(c.app.Importer.Pending()) == 1 }, "import pending")

	require.NoError(t, k8sClient.Create(context.Background(), &corev1.Namespace{Name: late}))
	eventually(t, func() bool {
		return get(t, late, "frontend", &mcsv1beta1.ServiceImport{}) && len(c.app.Importer.Pending()) == 0
	}, "pending import applied")
}

func TestLabelDrivenExport(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	ns := createNamespace(t)
	c := startController(t, mcstest.StartRegistry(t), ns)
	// A hand-made ServiceExport of a labelled Service is left alone; once it
	// is deleted, a managed one takes its place.
	handMade := &mcsv1beta1.ServiceExport{Namespace: ns, Name: "web"}
	require.NoError(t, k8sClient.Create(ctx, handMade))
	svc := webService(ns)
	svc.Labels = map[string]string{api.LabelExport: api.ExportEnabled, api.Prefix + "tier": "gold"}
	require.NoError(t, k8sClient.Create(ctx, svc))
	eventually(t, func() bool { return registryWebExport(t, c, ns) != nil }, "the hand-made ServiceExport is exported")
	var se mcsv1beta1.ServiceExport
	require.True(t, get(t, ns, "web", &se))
	assert.Empty(t, se.Labels[api.LabelManagedBy], "never adopted")
	require.NoError(t, k8sClient.Delete(ctx, &se))

	eventually(t, func() bool {
		return get(t, ns, "web", &se) && se.UID != handMade.UID && se.Labels[api.LabelManagedBy] == api.ManagedByController
	}, "managed ServiceExport created after the hand-made one went away")
	require.Len(t, se.OwnerReferences, 1)
	assert.Equal(t, svc.UID, se.OwnerReferences[0].UID)
	eventually(t, func() bool {
		x := registryWebExport(t, c, ns)
		return x != nil && x.GetSpec().GetLabels()[api.Prefix+"tier"] == "gold" && x.GetSpec().GetLabels()[api.LabelExport] == ""
	}, "exported with the prefixed label, without the trigger label")

	require.NoError(t, k8sClient.Get(ctx, ctrlclient.ObjectKeyFromObject(svc), svc))
	svc.Labels = nil
	require.NoError(t, k8sClient.Update(ctx, svc))
	eventually(t, func() bool {
		return !get(t, ns, "web", &mcsv1beta1.ServiceExport{}) && registryWebExport(t, c, ns) == nil
	}, "ServiceExport and export removed with the label")
}

// startServer runs an mcs-registry with the memory store on loopback ports and
// returns a client of it and the store, which tests damage directly.
func startServer(t *testing.T) (string, *client.Client, store.Store) {
	t.Helper()
	cfg := registryconfig.Default()
	cfg.GRPCListenAddress, cfg.HTTPListenAddress, cfg.AdminListenAddress = "127.0.0.1:0", "127.0.0.1:0", "127.0.0.1:0"
	cfg.ShutdownDrain, cfg.ShutdownTimeout = 0, 2*time.Second
	// Frequent reloads, so that a record removed behind the registry's back
	// leaves the view quickly.
	cfg.RefreshInterval = 200 * time.Millisecond
	mem := memory.New()
	srv, err := server.NewApp(cfg, server.AppOptions{Store: mem})
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- srv.Server.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		if err := <-done; err != nil {
			t.Logf("registry stopped: %v", err)
		}
	})
	url := "http://" + srv.Server.GRPCAddr()
	return url, mcstest.NewClient(t, url), mem
}

// TestAgainstRegistry runs the controller against the real mcs-registry
// instead of fakeregistry: its validation, merge rules and conditions.
func TestAgainstRegistry(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	ns := createNamespace(t)
	url, fakeReg, backing := startServer(t)
	startApp(t, url, ns)

	require.NoError(t, k8sClient.Create(ctx, webService(ns)))
	require.NoError(t, k8sClient.Create(ctx, mcstest.EndpointSlice(ns, "web-1", "web", "10.5.0.1", "10.5.0.2")))
	require.NoError(t, k8sClient.Create(ctx, &mcsv1beta1.ServiceExport{Namespace: ns, Name: "web"}))

	ownExport := func() *mcsv1.ServiceExport {
		resp, err := fakeReg.ServiceExports().List(ctx, &mcsv1.ListServiceExportsRequest{Clusterset: mcstest.ClusterSet, Cluster: mcstest.Cluster})
		require.NoError(t, err)
		for _, x := range resp.GetExports() {
			if x.GetNamespace() == ns && x.GetName() == "web" {
				return x
			}
		}
		return nil
	}
	eventually(t, func() bool {
		x := ownExport()
		return x != nil && x.GetEndpointCount() == 2
	}, "export and endpoints in mcs-registry")

	var se mcsv1beta1.ServiceExport
	eventually(t, func() bool {
		get(t, ns, "web", &se)
		return meta.IsStatusConditionTrue(se.Status.Conditions, api.ConditionReady) &&
			meta.IsStatusConditionFalse(se.Status.Conditions, api.ConditionConflict)
	}, "Ready and Conflict from mcs-registry")
	assert.Equal(t, api.ConditionReasonReady, meta.FindStatusCondition(se.Status.Conditions, api.ConditionReady).Reason)

	// Cluster b exports the same Service, naming the port differently: the
	// import gets both clusters, and b's endpoints are usable under the
	// merged port name with their own target port.
	_, err := fakeReg.Clusters().Register(ctx, &mcsv1.RegisterClusterRequest{Clusterset: mcstest.ClusterSet, Cluster: remote, Region: "us-west-2"})
	require.NoError(t, err)
	spec := remoteSpec()
	spec.Labels = nil
	spec.Ports[0].Port = 80
	spec.Ports[0].Name = "web"
	remoteEps := remoteEndpoints("10.6.0.1")
	remoteEps[0].Ports[0].Name, remoteEps[0].Ports[0].Port = "web", 9090
	_, err = fakeReg.ServiceExports().Put(ctx, &mcsv1.PutServiceExportRequest{Clusterset: mcstest.ClusterSet, Cluster: remote,
		Namespace: ns, Name: "web", Spec: spec, Generation: client.NextGeneration(0)})
	require.NoError(t, err)
	_, err = fakeReg.ServiceExports().ReplaceEndpoints(ctx, &mcsv1.ReplaceEndpointsRequest{Clusterset: mcstest.ClusterSet, Cluster: remote,
		Namespace: ns, Name: "web", Endpoints: remoteEps, Generation: client.NextGeneration(0)})
	require.NoError(t, err)

	clustersOf := func() []string {
		var out []string
		for _, s := range importedSlices(t, ns, "web") {
			out = append(out, s.Labels[api.LabelMCSSourceCluster])
		}
		sort.Strings(out)
		return out
	}
	eventually(t, func() bool { return assert.ObjectsAreEqual([]string{mcstest.Cluster, remote}, clustersOf()) }, "slices of both clusters")
	var si mcsv1beta1.ServiceImport
	eventually(t, func() bool {
		return get(t, ns, "web", &si) && len(si.Status.Clusters) == 2 && len(si.Spec.IPs) == 1
	}, "ServiceImport of both clusters with a VIP")
	for _, s := range importedSlices(t, ns, "web") {
		require.Len(t, s.Ports, 1, s.Name)
		assert.Equal(t, "http", *s.Ports[0].Name, "%s: the merged port name", s.Name)
		if s.Labels[api.LabelMCSSourceCluster] == remote {
			assert.Equal(t, int32(9090), *s.Ports[0].Port, "cluster b's own target port")
		} else {
			assert.Equal(t, int32(8080), *s.Ports[0].Port)
		}
	}
	assert.Equal(t, "http", si.Spec.Ports[0].Name)
	assert.True(t, meta.IsStatusConditionTrue(si.Status.Conditions, api.ConditionConflict), "the port names differ")

	// The registry loses this cluster's endpoint set, then the whole record,
	// while nothing changes locally and the heartbeats stay healthy: both
	// are written again without a restart or a reconnect.
	ownKey := store.ExportKey{ClusterSet: mcstest.ClusterSet, Cluster: mcstest.Cluster, Namespace: ns, Name: "web"}
	before := ownExport()
	require.NoError(t, backing.DeleteEndpointSet(ctx, ownKey))
	eventually(t, func() bool {
		x := ownExport()
		return x != nil && x.GetEndpointCount() == 2 && x.GetEndpointsGeneration() > before.GetEndpointsGeneration()
	}, "endpoint set written again after the registry lost it")
	before = ownExport()
	require.NoError(t, backing.DeleteExport(ctx, ownKey))
	eventually(t, func() bool {
		x := ownExport()
		return x != nil && x.GetGeneration() > before.GetGeneration() && x.GetEndpointCount() == 2
	}, "export written again after the registry lost it")
	assert.True(t, meta.IsStatusConditionTrue(se.Status.Conditions, api.ConditionValid))

	// Withdrawing the local export leaves cluster b's part.
	require.NoError(t, k8sClient.Delete(ctx, &se))
	eventually(t, func() bool {
		return ownExport() == nil && assert.ObjectsAreEqual([]string{remote}, clustersOf())
	}, "local export withdrawn, remote slices kept")
}
