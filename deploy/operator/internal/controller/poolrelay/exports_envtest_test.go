//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/multicluster"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlconfig "sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/yaml"
)

// testCRD returns a CustomResourceDefinition that serves kind in group at
// version only and keeps any field: a stand-in for the MCS and About APIs, whose
// CRDs the operator does not ship.
func testCRD(group, kind, plural, version string, scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		Name: plural + "." + group,
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: group,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind: kind, ListKind: kind + "List", Plural: plural, Singular: strings.ToLower(kind),
			},
			Scope: scope,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name: version, Served: true, Storage: true,
				Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object", XPreserveUnknownFields: new(true),
				}},
				Subresources: &apiextensionsv1.CustomResourceSubresources{
					Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
				},
			}},
		},
	}
}

// startExportAPIServer starts an API server with the operator's CRDs and
// stand-ins for the ServiceExport API, served at serviceExportVersion only, and
// for the About API's ClusterProperty. It runs no garbage collector.
func startExportAPIServer(t *testing.T, serviceExportVersion string) *rest.Config {
	t.Helper()
	// about.k8s.io is a protected API group, so its CRD needs the approval
	// annotation that the About API's own CRD has.
	clusterProperties := testCRD(multicluster.ClusterPropertyGVK.Group, multicluster.ClusterPropertyGVK.Kind,
		"clusterproperties", multicluster.ClusterPropertyGVK.Version, apiextensionsv1.ClusterScoped)
	clusterProperties.Annotations = map[string]string{
		apiextensionsv1.KubeAPIApprovedAnnotation: "https://github.com/kubernetes/enhancements/pull/3084",
	}
	environment := &envtest.Environment{
		CRDDirectoryPaths: []string{filepath.Join("..", "..", "..", "config", "crd", "bases")},
		CRDs: []*apiextensionsv1.CustomResourceDefinition{
			testCRD(multicluster.ServiceExportGroupKind.Group, multicluster.ServiceExportGroupKind.Kind,
				"serviceexports", serviceExportVersion, apiextensionsv1.NamespaceScoped),
			clusterProperties,
		},
		ErrorIfCRDPathMissing: true,
	}
	config, err := environment.Start()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := environment.Stop(); err != nil {
			t.Error(err)
		}
	})
	return config
}

// relayControllerOptions configures the Relay controller that a test starts.
type relayControllerOptions struct {
	// version is the multicluster.x-k8s.io version that the gate resolved: the
	// one at which the API server serves ServiceExports.
	version string
	// restricted is the namespace that a namespace-restricted operator manages,
	// or empty for an operator that manages every namespace.
	restricted string
	// wrap, when set, wraps the manager's client.
	wrap func(client.Client) client.Client
}

// startRelayController starts a manager that runs the Relay controller with the
// pool's exports on, for standard MCS, and returns a function that stops the
// manager and waits for it.
func startRelayController(t *testing.T, config *rest.Config, options relayControllerOptions) func() {
	t.Helper()
	managerOptions := ctrl.Options{
		Scheme:  newExportTestScheme(t),
		Metrics: server.Options{BindAddress: "0"},
		// A test can start managers one after another, with the same controllers.
		Controller: ctrlconfig.Controller{SkipNameValidation: new(true)},
		NewClient: func(restConfig *rest.Config, clientOptions client.Options) (client.Client, error) {
			c, err := client.New(restConfig, clientOptions)
			if err != nil || options.wrap == nil {
				return c, err
			}
			return options.wrap(c), nil
		},
	}
	if options.restricted != "" {
		// As cmd/main.go configures a namespace-restricted operator's cache.
		managerOptions.Cache.DefaultNamespaces = map[string]cache.Config{options.restricted: {}}
	}
	manager, err := ctrl.NewManager(config, managerOptions)
	if err != nil {
		t.Fatal(err)
	}
	if err := Setup(manager, &configv1alpha1.OperatorConfiguration{
		Discovery:    configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
		MultiCluster: configv1alpha1.MultiClusterConfiguration{MCSImplementation: configv1alpha1.MCSImplementationStandard},
		Namespace:    configv1alpha1.NamespaceConfiguration{Restricted: options.restricted},
	}, &commoncontroller.RuntimeConfig{
		Gate: features.Gates{MultiClusterExport: true, ServiceExportVersion: options.version},
	}); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- manager.Start(ctx) }()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			cancel()
			if err := <-done; err != nil {
				t.Error(err)
			}
		})
	}
	t.Cleanup(stop)
	return stop
}

// eventually polls check until it reports done, and fails the test with what
// after 30 seconds.
func eventually(t *testing.T, what string, check func() (bool, error)) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for {
		done, err := check()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if done {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// readSampleRelay returns the checked-in DynamoPoolRelay sample.
func readSampleRelay(t *testing.T) *v1alpha1.DynamoPoolRelay {
	t.Helper()
	sample, err := os.ReadFile(filepath.Join("..", "..", "..", "samples", "dynamo-pool-relay.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	relay := &v1alpha1.DynamoPoolRelay{}
	if err := yaml.UnmarshalStrict(sample, relay); err != nil {
		t.Fatal(err)
	}
	return relay
}

// createNamespace creates the namespace called name.
func createNamespace(t *testing.T, c client.Client, name string) {
	t.Helper()
	namespace := &corev1.Namespace{}
	namespace.Name = name
	if err := c.Create(context.Background(), namespace); err != nil {
		t.Fatal(err)
	}
}

// createDGD creates the DGD of the Relay's name with a worker and a frontend
// component named frontend, and the frontend's Service, controlled by the DGD as
// the DGD controller creates it, and returns the DGD.
func createDGD(t *testing.T, c client.Client, relay *v1alpha1.DynamoPoolRelay, frontend string) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: relay.Name, Namespace: relay.Namespace,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
			{ComponentName: frontend, ComponentType: v1beta1.ComponentTypeFrontend},
		}},
	}
	if err := c.Create(context.Background(), dgd); err != nil {
		t.Fatal(err)
	}
	for _, name := range multicluster.FrontendServiceNames(dgd) {
		service := &corev1.Service{
			Name: name, Namespace: dgd.Namespace,
			Spec: corev1.ServiceSpec{Ports: []corev1.ServicePort{{Name: "http", Port: 8000}}},
		}
		if err := controllerutil.SetControllerReference(dgd, service, c.Scheme()); err != nil {
			t.Fatal(err)
		}
		if err := c.Create(context.Background(), service); err != nil {
			t.Fatal(err)
		}
	}
	return dgd
}

// exportVersions returns the resourceVersion of each ServiceExport in namespace,
// by name.
func exportVersions(t *testing.T, c client.Client, namespace string) map[string]string {
	t.Helper()
	versions := map[string]string{}
	for name, export := range exportsByName(t, c, namespace) {
		versions[name] = export.GetResourceVersion()
	}
	return versions
}

// reportTerminalFailure writes the status that the DGD controller writes when a
// DGD's workloads cannot be reconciled at all, with message.
func reportTerminalFailure(t *testing.T, c client.Client, dgd *v1beta1.DynamoGraphDeployment, message string) {
	t.Helper()
	if err := c.Get(context.Background(), client.ObjectKeyFromObject(dgd), dgd); err != nil {
		t.Fatal(err)
	}
	dgd.Status.State = v1beta1.DGDStateFailed
	meta.SetStatusCondition(&dgd.Status.Conditions, metav1.Condition{
		Type: "Ready", Status: metav1.ConditionFalse, Reason: "UnsupportedWorkloadProvider", Message: message,
	})
	if err := c.Status().Update(context.Background(), dgd); err != nil {
		t.Fatal(err)
	}
}

// ownedByRelay reports whether relay controls export and blocks its own deletion
// on it, so that the garbage collector deletes the export with the Relay.
func ownedByRelay(export *unstructured.Unstructured, relay *v1alpha1.DynamoPoolRelay) bool {
	controller := metav1.GetControllerOf(export)
	return controller != nil && controller.UID == relay.UID && ptr.Deref(controller.BlockOwnerDeletion, false)
}

// exportsByName returns the ServiceExports in namespace by name.
func exportsByName(t *testing.T, c client.Client, namespace string) map[string]*unstructured.Unstructured {
	t.Helper()
	exports := map[string]*unstructured.Unstructured{}
	items := listServiceExports(t, c, exportTestVersion, namespace)
	for i := range items {
		exports[items[i].GetName()] = &items[i]
	}
	return exports
}

// waitForExports waits until the ServiceExports in the Relay's namespace are
// exactly names, none being deleted, and each owned by relay.
func waitForExports(t *testing.T, c client.Client, relay *v1alpha1.DynamoPoolRelay, names ...string) {
	t.Helper()
	eventually(t, "the exports "+strings.Join(names, ", "), func() (bool, error) {
		exports := exportsByName(t, c, relay.Namespace)
		if len(exports) != len(names) {
			return false, nil
		}
		for name, export := range exports {
			if !slices.Contains(names, name) || export.GetDeletionTimestamp() != nil || !ownedByRelay(export, relay) {
				return false, nil
			}
		}
		return true, nil
	})
}

// waitForRelayService waits until the controller publishes the Relay's Service,
// or withdraws it.
func waitForRelayService(t *testing.T, c client.Client, relay *v1alpha1.DynamoPoolRelay, published bool) {
	t.Helper()
	key := types.NamespacedName{Namespace: relay.Namespace, Name: multicluster.PoolRelayServiceName(relay.Name)}
	eventually(t, fmt.Sprintf("the Relay's Service to be published: %v", published), func() (bool, error) {
		err := c.Get(context.Background(), key, &corev1.Service{})
		if apierrors.IsNotFound(err) {
			return !published, nil
		}
		return published, err
	})
}

// waitForDeletion waits until obj is gone.
func waitForDeletion(t *testing.T, c client.Client, obj client.Object) {
	t.Helper()
	eventually(t, obj.GetName()+" to be deleted", func() (bool, error) {
		err := c.Get(context.Background(), client.ObjectKeyFromObject(obj), obj)
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
}

// markRelayListenersAvailable reports the Relay's Deployment as rolled out, as
// the Deployment controller would if envtest ran it, and waits until the Relay
// is Ready. The controller requeues a Relay that is not Ready every 15 seconds;
// a Ready Relay is requeued only by watch events.
func markRelayListenersAvailable(t *testing.T, c client.Client, relay *v1alpha1.DynamoPoolRelay) {
	t.Helper()
	ctx := context.Background()
	key := types.NamespacedName{Namespace: relay.Namespace, Name: multicluster.PoolRelayServiceName(relay.Name)}
	eventually(t, "the Relay's Deployment", func() (bool, error) {
		err := c.Get(ctx, key, &appsv1.Deployment{})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return err == nil, err
	})
	replicas := ptr.Deref(relay.Spec.Replicas, 2)
	if err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		deployment := &appsv1.Deployment{}
		if err := c.Get(ctx, key, deployment); err != nil {
			return err
		}
		deployment.Status = appsv1.DeploymentStatus{
			ObservedGeneration: deployment.Generation,
			Replicas:           replicas,
			UpdatedReplicas:    replicas,
			ReadyReplicas:      replicas,
			AvailableReplicas:  replicas,
		}
		return c.Status().Update(ctx, deployment)
	}); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the Relay to be Ready", func() (bool, error) {
		current := &v1alpha1.DynamoPoolRelay{}
		if err := c.Get(ctx, client.ObjectKeyFromObject(relay), current); err != nil {
			return false, err
		}
		return meta.IsStatusConditionTrue(current.Status.Conditions, "Ready"), nil
	})
}

//nolint:gocyclo // This test follows one Relay through the replacement of its DGD against the API server.
func TestRelayControllerFollowsAReplacedDGD(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS is needed for the controller API test")
	}
	ctx := context.Background()
	config := startExportAPIServer(t, exportTestVersion)
	apiClient, err := client.New(config, client.Options{Scheme: newExportTestScheme(t)})
	if err != nil {
		t.Fatal(err)
	}
	stop := startRelayController(t, config, relayControllerOptions{version: exportTestVersion})

	t.Log("Create a DGD with a frontend component, and its DynamoPoolRelay from the checked-in sample")
	relay := readSampleRelay(t)
	createNamespace(t, apiClient, relay.Namespace)
	first := createDGD(t, apiClient, relay, "Frontend")
	if err := apiClient.Create(ctx, relay); err != nil {
		t.Fatal(err)
	}
	relayUID := relay.UID
	poolRelayExport := multicluster.PoolRelayServiceName(relay.Name)
	waitForExports(t, apiClient, relay, poolRelayExport, relay.Name+"-frontend")
	waitForRelayService(t, apiClient, relay, true)

	t.Log("An MCS implementation holds the PoolRelay export with a finalizer, so its deletion will stay pending")
	held := exportsByName(t, apiClient, relay.Namespace)[poolRelayExport]
	heldUID := held.GetUID()
	held.SetFinalizers([]string{"example.com/mcs-hold"})
	if err := apiClient.Update(ctx, held); err != nil {
		t.Fatal(err)
	}

	t.Log("Delete the DGD: the Relay's Service and frontend export are withdrawn, and the PoolRelay export's deletion is pending")
	if err := apiClient.Delete(ctx, first); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the exports to be withdrawn", func() (bool, error) {
		exports := exportsByName(t, apiClient, relay.Namespace)
		pending, found := exports[poolRelayExport]
		return len(exports) == 1 && found && pending.GetDeletionTimestamp() != nil, nil
	})
	waitForRelayService(t, apiClient, relay, false)

	t.Log("Create a new DGD of the Relay's name with another frontend: its export is created while the old PoolRelay export is pending")
	second := createDGD(t, apiClient, relay, "Web")
	if second.UID == first.UID {
		t.Fatal("the new DGD has the old DGD's UID")
	}
	eventually(t, "the new frontend's export next to the pending one", func() (bool, error) {
		exports := exportsByName(t, apiClient, relay.Namespace)
		web, webFound := exports[relay.Name+"-web"]
		pending, pendingFound := exports[poolRelayExport]
		return len(exports) == 2 && webFound && ownedByRelay(web, relay) &&
			pendingFound && pending.GetUID() == heldUID && pending.GetDeletionTimestamp() != nil, nil
	})
	waitForRelayService(t, apiClient, relay, true)

	t.Log("The MCS implementation lets go of the old export: a new PoolRelay export is created with no other change")
	held = exportsByName(t, apiClient, relay.Namespace)[poolRelayExport]
	held.SetFinalizers(nil)
	if err := apiClient.Update(ctx, held); err != nil {
		t.Fatal(err)
	}
	waitForExports(t, apiClient, relay, poolRelayExport, relay.Name+"-web")
	if uid := exportsByName(t, apiClient, relay.Namespace)[poolRelayExport].GetUID(); uid == heldUID {
		t.Fatal("the PoolRelay export was not created again")
	}
	if err := apiClient.Get(ctx, client.ObjectKeyFromObject(relay), relay); err != nil {
		t.Fatal(err)
	}
	if relay.UID != relayUID {
		t.Fatal("the DynamoPoolRelay was created again")
	}

	t.Log("Status updates of the DGD, such as a terminal workload failure, write nothing to the exports")
	before := exportVersions(t, apiClient, relay.Namespace)
	for attempt := range 3 {
		reportTerminalFailure(t, apiClient, second, fmt.Sprintf("Grove is disabled (report %d)", attempt))
	}
	time.Sleep(2 * time.Second)
	if after := exportVersions(t, apiClient, relay.Namespace); !maps.Equal(before, after) {
		t.Fatalf("DGD status updates changed the exports: %v -> %v", before, after)
	}

	t.Log("Only the frontend's Service changes, so only the Service watch requeues the Relay, whose listeners are " +
		"available and which therefore polls nothing: deleting the Service withdraws its export, and creating it " +
		"again exports it again")
	markRelayListenersAvailable(t, apiClient, relay)
	frontendService := &corev1.Service{}
	frontendKey := client.ObjectKey{Namespace: relay.Namespace, Name: relay.Name + "-web"}
	if err := apiClient.Get(ctx, frontendKey, frontendService); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Delete(ctx, frontendService); err != nil {
		t.Fatal(err)
	}
	waitForExports(t, apiClient, relay, poolRelayExport)
	recreated := &corev1.Service{
		Name: frontendKey.Name, Namespace: frontendKey.Namespace,
		Spec: corev1.ServiceSpec{Ports: []corev1.ServicePort{{Name: "http", Port: 8000}}},
	}
	if err := controllerutil.SetControllerReference(second, recreated, apiClient.Scheme()); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Create(ctx, recreated); err != nil {
		t.Fatal(err)
	}
	waitForExports(t, apiClient, relay, poolRelayExport, relay.Name+"-web")

	t.Log("With the controller stopped, replace the DGD at once, so that no cache sees it missing")
	stop()
	if err := apiClient.Delete(ctx, second); err != nil {
		t.Fatal(err)
	}
	waitForDeletion(t, apiClient, second)
	createDGD(t, apiClient, relay, "Api")
	startRelayController(t, config, relayControllerOptions{version: exportTestVersion})
	waitForExports(t, apiClient, relay, poolRelayExport, relay.Name+"-api")
	waitForRelayService(t, apiClient, relay, true)
}

// failingCreates refuses the first ServiceExport creates, as an API server that
// is briefly unavailable does, and counts the attempts.
type failingCreates struct {
	client.Client
	remaining atomic.Int32
	attempts  atomic.Int32
}

func (c *failingCreates) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	if obj.GetObjectKind().GroupVersionKind().GroupKind() == multicluster.ServiceExportGroupKind {
		c.attempts.Add(1)
		if c.remaining.Add(-1) >= 0 {
			return apierrors.NewServiceUnavailable("injected failure")
		}
	}
	return c.Client.Create(ctx, obj, opts...)
}

func TestRelayControllerRetriesFailedExports(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS is needed for the controller API test")
	}
	ctx := context.Background()
	config := startExportAPIServer(t, exportTestVersion)
	apiClient, err := client.New(config, client.Options{Scheme: newExportTestScheme(t)})
	if err != nil {
		t.Fatal(err)
	}
	failing := &failingCreates{}
	startRelayController(t, config, relayControllerOptions{
		version: exportTestVersion,
		wrap: func(c client.Client) client.Client {
			failing.Client = c
			return failing
		},
	})

	t.Log("Create a pool whose DGD reports a terminal workload failure, and wait until its Relay is Ready")
	relay := readSampleRelay(t)
	createNamespace(t, apiClient, relay.Namespace)
	dgd := createDGD(t, apiClient, relay, "Frontend")
	reportTerminalFailure(t, apiClient, dgd, "Grove is disabled")
	if err := apiClient.Create(ctx, relay); err != nil {
		t.Fatal(err)
	}
	frontendExport := relay.Name + "-frontend"
	waitForExports(t, apiClient, relay, multicluster.PoolRelayServiceName(relay.Name), frontendExport)
	markRelayListenersAvailable(t, apiClient, relay)

	t.Log("The API server refuses the next two ServiceExport creates, and a user deletes the frontend export")
	failing.attempts.Store(0)
	failing.remaining.Store(2)
	deleted := exportsByName(t, apiClient, relay.Namespace)[frontendExport]
	if err := apiClient.Delete(ctx, deleted); err != nil {
		t.Fatal(err)
	}

	t.Log("Only the retries of the failed reconciles requeue the Ready Relay: the export is created again, with no other change")
	waitForExports(t, apiClient, relay, multicluster.PoolRelayServiceName(relay.Name), frontendExport)
	if exportsByName(t, apiClient, relay.Namespace)[frontendExport].GetUID() == deleted.GetUID() {
		t.Fatal("the frontend export was not created again")
	}
	if attempts := failing.attempts.Load(); attempts <= 2 {
		t.Fatalf("the export was created after %d attempts, without retrying the 2 refused ones", attempts)
	}
}

func TestRelayControllerWorksAtTheResolvedVersion(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS is needed for the controller API test")
	}
	ctx := context.Background()
	// A version other than the one the other tests' clusters serve.
	const version = "v1beta1"
	config := startExportAPIServer(t, version)
	apiClient, err := client.New(config, client.Options{Scheme: newExportTestScheme(t)})
	if err != nil {
		t.Fatal(err)
	}
	startRelayController(t, config, relayControllerOptions{version: version})

	t.Log("Create a pool in a cluster that serves ServiceExports only at " + version)
	relay := readSampleRelay(t)
	createNamespace(t, apiClient, relay.Namespace)
	createDGD(t, apiClient, relay, "Frontend")
	if err := apiClient.Create(ctx, relay); err != nil {
		t.Fatal(err)
	}

	t.Log("The controller watches and writes ServiceExports at that version, so it exports the pool")
	poolRelayExport := func() *unstructured.Unstructured {
		items := listServiceExports(t, apiClient, version, relay.Namespace)
		for i := range items {
			if items[i].GetName() == multicluster.PoolRelayServiceName(relay.Name) && ownedByRelay(&items[i], relay) {
				return &items[i]
			}
		}
		return nil
	}
	eventually(t, "the PoolRelay export at "+version, func() (bool, error) {
		return poolRelayExport() != nil, nil
	})
	markRelayListenersAvailable(t, apiClient, relay)

	t.Log("Delete the export: only the watch at " + version + " requeues the Ready Relay, which exports its Service again")
	deleted := poolRelayExport()
	if err := apiClient.Delete(ctx, deleted); err != nil {
		t.Fatal(err)
	}
	eventually(t, "a new PoolRelay export", func() (bool, error) {
		export := poolRelayExport()
		return export != nil && export.GetUID() != deleted.GetUID(), nil
	})
}

func TestNamespaceRestrictedRelayControllerFollowsTheClusterID(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS is needed for the controller API test")
	}
	ctx := context.Background()
	config := startExportAPIServer(t, exportTestVersion)
	scheme := newExportTestScheme(t)
	if err := eventsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	apiClient, err := client.New(config, client.Options{Scheme: scheme})
	if err != nil {
		t.Fatal(err)
	}
	relay := readSampleRelay(t)
	startRelayController(t, config, relayControllerOptions{version: exportTestVersion, restricted: relay.Namespace})

	t.Log("Create a pool in the namespace that a namespace-restricted operator manages, in a cluster with no ID in its ClusterSet")
	createNamespace(t, apiClient, relay.Namespace)
	createDGD(t, apiClient, relay, "Frontend")
	if err := apiClient.Create(ctx, relay); err != nil {
		t.Fatal(err)
	}
	waitForExports(t, apiClient, relay, multicluster.PoolRelayServiceName(relay.Name), relay.Name+"-frontend")

	t.Log("Each export's creation is an event of its own, though both events regard the Relay at one version")
	eventually(t, "a CreateServiceExport event for each export", func() (bool, error) {
		notes := strings.Join(relayEventNotes(t, apiClient, relay, "CreateServiceExport"), "\n")
		return strings.Contains(notes, "Service "+multicluster.PoolRelayServiceName(relay.Name)+" ") &&
			strings.Contains(notes, "Service "+relay.Name+"-frontend "), nil
	})
	markRelayListenersAvailable(t, apiClient, relay)

	t.Log("Give the cluster an ID in its ClusterSet that differs from the Relay's clusterId")
	property := newClusterIDProperty(relay.Spec.ClusterID + "-east")
	if err := apiClient.Create(ctx, property); err != nil {
		t.Fatal(err)
	}
	mismatchWith := func(clusterID string) func() (bool, error) {
		return func() (bool, error) {
			for _, note := range relayEventNotes(t, apiClient, relay, reasonClusterIDMismatch) {
				if strings.Contains(note, fmt.Sprintf("ID %q", clusterID)) {
					return true, nil
				}
			}
			return false, nil
		}
	}

	t.Log("The cluster-scoped property's event reaches the restricted controller, which reports the mismatch on the Relay")
	eventually(t, "a ClusterIDMismatch event on the Relay", mismatchWith(relay.Spec.ClusterID+"-east"))

	t.Log("Change the cluster's ID: the new mismatch is an event of its own, not a repeat of the old one")
	property.Object["spec"] = map[string]any{"value": relay.Spec.ClusterID + "-west"}
	if err := apiClient.Update(ctx, property); err != nil {
		t.Fatal(err)
	}
	eventually(t, "a ClusterIDMismatch event with the new ID", mismatchWith(relay.Spec.ClusterID+"-west"))
}

// relayEventNotes returns the notes of the events with reason that regard relay.
func relayEventNotes(t *testing.T, c client.Client, relay *v1alpha1.DynamoPoolRelay, reason string) []string {
	t.Helper()
	recorded := &eventsv1.EventList{}
	if err := c.List(context.Background(), recorded, client.InNamespace(relay.Namespace)); err != nil {
		t.Fatal(err)
	}
	var notes []string
	for i := range recorded.Items {
		if recorded.Items[i].Reason == reason && recorded.Items[i].Regarding.Name == relay.Name {
			notes = append(notes, recorded.Items[i].Note)
		}
	}
	return notes
}
