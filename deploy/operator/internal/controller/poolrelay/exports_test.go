/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/multicluster"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

// exportTestVersion is the multicluster.x-k8s.io version that the tests' clusters
// serve.
const exportTestVersion = "v1alpha1"

// newExportTestScheme returns a scheme with the kinds that the controller reads
// and writes.
func newExportTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		v1alpha1.AddToScheme, v1beta1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme, rbacv1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatalf("add to scheme: %v", err)
		}
	}
	return scheme
}

// newExportTestClient returns a fake client that serves the scheme's kinds, the
// ServiceExport kind at version only, and the ClusterProperty kind, which the
// scheme does not register.
func newExportTestClient(
	scheme *runtime.Scheme,
	version string,
	funcs interceptor.Funcs,
	objects ...client.Object,
) client.WithWatch {
	mapper := meta.NewDefaultRESTMapper(nil)
	for gvk := range scheme.AllKnownTypes() {
		mapper.Add(gvk, meta.RESTScopeNamespace)
	}
	mapper.Add(multicluster.ServiceExportGroupKind.WithVersion(version), meta.RESTScopeNamespace)
	mapper.Add(multicluster.ClusterPropertyGVK, meta.RESTScopeRoot)
	return fake.NewClientBuilder().
		WithScheme(scheme).
		WithRESTMapper(mapper).
		WithStatusSubresource(&v1alpha1.DynamoPoolRelay{}, &v1beta1.DynamoGraphDeployment{}, &appsv1.Deployment{}).
		WithInterceptorFuncs(funcs).
		WithObjects(objects...).
		Build()
}

// newExportTestDGD returns a DGD named name in namespace inference that a Relay
// can publish: a decode worker, and a frontend component for each of frontends.
func newExportTestDGD(name string, frontends ...string) *v1beta1.DynamoGraphDeployment {
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: name, Namespace: "inference", UID: uuid.NewUUID(),
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
		Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{
			"decode": {RuntimeNamespace: "inference-" + name + "-hash"},
		}},
	}
	for _, frontend := range frontends {
		dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: frontend, ComponentType: v1beta1.ComponentTypeFrontend,
		})
	}
	return dgd
}

// newExportTestRelay returns the Relay that makes dgd a pool of cluster east.
func newExportTestRelay(dgd *v1beta1.DynamoGraphDeployment) *v1alpha1.DynamoPoolRelay {
	return &v1alpha1.DynamoPoolRelay{
		Name: dgd.Name, Namespace: dgd.Namespace, UID: uuid.NewUUID(),
		Spec: relaySpec("east"),
	}
}

// ownedExport returns the export at version of the Service named name that relay
// controls, labeled with the DGD of the Relay's name.
func ownedExport(
	t *testing.T,
	scheme *runtime.Scheme,
	version string,
	relay *v1alpha1.DynamoPoolRelay,
	name string,
) *unstructured.Unstructured {
	t.Helper()
	export := multicluster.RenderServiceExport(version, relay.Namespace, name, relay.Name)
	if err := controllerutil.SetControllerReference(relay, export, scheme); err != nil {
		t.Fatalf("set the owner of ServiceExport %s: %v", name, err)
	}
	return export
}

// poolServices returns the Services that the operator creates for the pool: the
// PoolRelay Service, which relay controls, and a Service for each frontend, which
// dgd controls. It leaves out the Services named in skip.
func poolServices(
	t *testing.T,
	scheme *runtime.Scheme,
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
	skip ...string,
) []client.Object {
	t.Helper()
	var services []client.Object
	for _, name := range multicluster.ServiceNames(dgd) {
		if slices.Contains(skip, name) {
			continue
		}
		var controller metav1.Object = dgd
		if name == multicluster.PoolRelayServiceName(dgd.Name) {
			controller = relay
		}
		service := &corev1.Service{Name: name, Namespace: dgd.Namespace}
		if err := controllerutil.SetControllerReference(controller, service, scheme); err != nil {
			t.Fatalf("set the controller of Service %s: %v", name, err)
		}
		services = append(services, service)
	}
	return services
}

// listServiceExports returns the ServiceExports in namespace, read at version.
func listServiceExports(t *testing.T, c client.Client, version, namespace string) []unstructured.Unstructured {
	t.Helper()
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(multicluster.ServiceExportGroupKind.WithVersion(version).GroupVersion().WithKind("ServiceExportList"))
	if err := c.List(context.Background(), list, client.InNamespace(namespace)); err != nil {
		t.Fatalf("list ServiceExports: %v", err)
	}
	return list.Items
}

// exportNames returns the names of exports.
func exportNames(exports []unstructured.Unstructured) []string {
	var names []string
	for i := range exports {
		names = append(names, exports[i].GetName())
	}
	return names
}

// newExportTestReconciler returns the Relay controller with the pool's exports on
// for implementation, reading and writing exports at version through c.
func newExportTestReconciler(
	c client.WithWatch,
	recorder events.EventRecorder,
	implementation configv1alpha1.MCSImplementation,
	version string,
) *reconciler {
	return &reconciler{
		Client:         c,
		scheme:         c.Scheme(),
		defaultBackend: configv1alpha1.DiscoveryBackendKubernetes,
		exports: &exportSettings{
			cache:    c,
			recorder: recorder,
			config: &configv1alpha1.OperatorConfiguration{
				MultiCluster: configv1alpha1.MultiClusterConfiguration{MCSImplementation: implementation},
			},
			runtimeConfig: &commoncontroller.RuntimeConfig{
				Gate: features.Gates{MultiClusterExport: true, ServiceExportVersion: version},
			},
		},
	}
}

// drainEvents returns the events recorder holds.
func drainEvents(recorder *events.FakeRecorder) []string {
	var recorded []string
	for len(recorder.Events) > 0 {
		recorded = append(recorded, <-recorder.Events)
	}
	return recorded
}

// eventReasons returns the reason of each recorded event, in order.
func eventReasons(recorded []string) []string {
	var reasons []string
	for _, recordedEvent := range recorded {
		if fields := strings.Fields(recordedEvent); len(fields) >= 2 {
			reasons = append(reasons, fields[1])
		}
	}
	return reasons
}

// countReasons returns how many of the recorded events have reason.
func countReasons(recorded []string, reason string) int {
	count := 0
	for _, recordedReason := range eventReasons(recorded) {
		if recordedReason == reason {
			count++
		}
	}
	return count
}

func TestReconcileExports(t *testing.T) {
	// <name>-pool-relay is 56 characters, one more than Karmada imports;
	// <name>-frontend is 54.
	longName := strings.Repeat("a", 45)

	for _, tc := range []struct {
		name           string
		dgdName        string
		frontends      []string
		implementation configv1alpha1.MCSImplementation
		ownedBefore    []string
		foreign        []string
		// ownedByAnother are exports that another Relay controls.
		ownedByAnother []string
		// elsewhere are a user's exports in namespace other.
		elsewhere []string
		// foreignServices are Services of the pool's names that no one controls,
		// missingServices do not exist, and dcdServices are controlled by a DCD
		// that the DGD controls.
		foreignServices []string
		missingServices []string
		dcdServices     []string
		// otherServices are controlled by another DGD, and otherDCDServices by a
		// DCD of another DGD.
		otherServices    []string
		otherDCDServices []string
		wantOwned        []string
		// wantReasons are the reasons of the events of the first reconcile. The
		// second changes nothing, so it records none.
		wantReasons []string
		// wantCondition is the reason of the Relay's Exported condition.
		wantCondition string
	}{
		{
			name:    "a pool exports its PoolRelay and frontend Services",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{"green-pool-relay", "green-frontend"},
			wantReasons:    []string{"CreateServiceExport", "CreateServiceExport"},
			wantCondition:  reasonExported,
		},
		{
			name:    "each frontend component's Service is exported under its own name",
			dgdName: "green", frontends: []string{"Frontend", "Web.v2"},
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{"green-pool-relay", "green-frontend", "green-web-v2"},
			wantReasons:    []string{"CreateServiceExport", "CreateServiceExport", "CreateServiceExport"},
			wantCondition:  reasonExported,
		},
		{
			name:    "a renamed frontend's export moves to its new Service",
			dgdName: "green", frontends: []string{"Web"},
			implementation: configv1alpha1.MCSImplementationKarmada,
			ownedBefore:    []string{"green-pool-relay", "green-frontend"},
			wantOwned:      []string{"green-pool-relay", "green-web"},
			wantReasons:    []string{"CreateServiceExport", "DeleteServiceExport"},
			wantCondition:  reasonExported,
		},
		{
			name:           "a pool without a frontend exports its PoolRelay Service and warns",
			dgdName:        "green",
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{"green-pool-relay"},
			wantReasons:    []string{"CreateServiceExport", reasonNoFrontend},
			wantCondition:  reasonNoFrontend,
		},
		{
			// A Service name has at most 63 characters, and the operator does not
			// shorten the names of the frontends' Services.
			name:    "a frontend Service name that no Service can have is not exported",
			dgdName: "green", frontends: []string{"Frontend", strings.Repeat("w", 60)},
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{"green-pool-relay", "green-frontend"},
			wantReasons:    []string{"CreateServiceExport", "CreateServiceExport", reasonInvalidServiceName},
			wantCondition:  reasonInvalidServiceName,
		},
		{
			name:    "an export a user created is neither adopted nor deleted",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationKarmada,
			foreign:        []string{"green-frontend"},
			wantOwned:      []string{"green-pool-relay"},
			wantReasons:    []string{"CreateServiceExport"},
			wantCondition:  reasonExported,
		},
		{
			name:    "another Relay's export of a Service the pool no longer has is left alone",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationStandard,
			ownedByAnother: []string{"green-old"},
			wantOwned:      []string{"green-pool-relay", "green-frontend"},
			wantReasons:    []string{"CreateServiceExport", "CreateServiceExport"},
			wantCondition:  reasonExported,
		},
		{
			name:    "Karmada skips a Service name it cannot import",
			dgdName: longName, frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{longName + "-frontend"},
			wantReasons:    []string{"CreateServiceExport", reasonServiceNameTooLong},
			wantCondition:  reasonServiceNameTooLong,
		},
		{
			name:    "standard MCS exports the same long names",
			dgdName: longName, frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{longName + "-pool-relay", longName + "-frontend"},
			wantReasons:    []string{"CreateServiceExport", "CreateServiceExport"},
			wantCondition:  reasonExported,
		},
		{
			name:    "Karmada withdraws the Relay's export of a Service name it cannot import",
			dgdName: longName, frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationKarmada,
			ownedBefore:    []string{longName + "-pool-relay", longName + "-frontend"},
			wantOwned:      []string{longName + "-frontend"},
			wantReasons:    []string{"DeleteServiceExport", reasonServiceNameTooLong},
			wantCondition:  reasonServiceNameTooLong,
		},
		{
			name:    "a Service that the pool does not control is not exported",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation:  configv1alpha1.MCSImplementationStandard,
			foreignServices: []string{"green-frontend"},
			wantOwned:       []string{"green-pool-relay"},
			wantReasons:     []string{"CreateServiceExport", reasonServiceNotOwned},
			wantCondition:   reasonServiceNotOwned,
		},
		{
			name:    "the Relay withdraws its export of a Service that the pool no longer controls",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation:  configv1alpha1.MCSImplementationKarmada,
			ownedBefore:     []string{"green-pool-relay", "green-frontend"},
			foreignServices: []string{"green-frontend"},
			wantOwned:       []string{"green-pool-relay"},
			wantReasons:     []string{"DeleteServiceExport", reasonServiceNotOwned},
			wantCondition:   reasonServiceNotOwned,
		},
		{
			name:    "a Service that another DGD controls is not exported",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationStandard,
			otherServices:  []string{"green-frontend"},
			wantOwned:      []string{"green-pool-relay"},
			wantReasons:    []string{"CreateServiceExport", reasonServiceNotOwned},
			wantCondition:  reasonServiceNotOwned,
		},
		{
			name:    "a Service that a DCD of another DGD controls is not exported",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation:   configv1alpha1.MCSImplementationStandard,
			otherDCDServices: []string{"green-frontend"},
			wantOwned:        []string{"green-pool-relay"},
			wantReasons:      []string{"CreateServiceExport", reasonServiceNotOwned},
			wantCondition:    reasonServiceNotOwned,
		},
		{
			name:    "a Service that does not exist yet is not exported",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation:  configv1alpha1.MCSImplementationStandard,
			missingServices: []string{"green-frontend"},
			wantOwned:       []string{"green-pool-relay"},
			wantReasons:     []string{"CreateServiceExport"},
			wantCondition:   reasonExported,
		},
		{
			name:    "a frontend Service that a DCD of the DGD controls is exported",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationKarmada,
			dcdServices:    []string{"green-frontend"},
			wantOwned:      []string{"green-pool-relay", "green-frontend"},
			wantReasons:    []string{"CreateServiceExport", "CreateServiceExport"},
			wantCondition:  reasonExported,
		},
		{
			name:    "exports of the same names in another namespace do not count",
			dgdName: "green", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationStandard,
			elsewhere:      []string{"green-pool-relay", "green-frontend"},
			wantOwned:      []string{"green-pool-relay", "green-frontend"},
			wantReasons:    []string{"CreateServiceExport", "CreateServiceExport"},
			wantCondition:  reasonExported,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			ctx := context.Background()

			t.Log("Create the Relay, its DGD, and the exports that exist before reconciling")
			scheme := newExportTestScheme(t)
			dgd := newExportTestDGD(tc.dgdName, tc.frontends...)
			relay := newExportTestRelay(dgd)
			objects := []client.Object{relay, dgd}
			skip := slices.Concat(tc.foreignServices, tc.missingServices, tc.dcdServices, tc.otherServices, tc.otherDCDServices)
			objects = append(objects, poolServices(t, scheme, relay, dgd, skip...)...)
			for _, name := range tc.foreignServices {
				objects = append(objects, &corev1.Service{Name: name, Namespace: dgd.Namespace})
			}
			other := newExportTestDGD("other")
			for _, name := range tc.otherServices {
				service := &corev1.Service{Name: name, Namespace: dgd.Namespace}
				g.Expect(controllerutil.SetControllerReference(other, service, scheme)).To(gomega.Succeed())
				objects = append(objects, service)
			}
			for _, name := range slices.Concat(tc.dcdServices, tc.otherDCDServices) {
				parent := dgd
				if slices.Contains(tc.otherDCDServices, name) {
					parent = other
				}
				dcd := &v1beta1.DynamoComponentDeployment{Name: name, Namespace: dgd.Namespace, UID: uuid.NewUUID()}
				service := &corev1.Service{Name: name, Namespace: dgd.Namespace}
				g.Expect(controllerutil.SetControllerReference(parent, dcd, scheme)).To(gomega.Succeed())
				g.Expect(controllerutil.SetControllerReference(dcd, service, scheme)).To(gomega.Succeed())
				objects = append(objects, dcd, service)
			}
			for _, name := range tc.ownedBefore {
				objects = append(objects, ownedExport(t, scheme, exportTestVersion, relay, name))
			}
			for _, name := range tc.foreign {
				objects = append(objects, multicluster.RenderServiceExport(exportTestVersion, dgd.Namespace, name, dgd.Name))
			}
			another := newExportTestRelay(newExportTestDGD("another"))
			for _, name := range tc.ownedByAnother {
				objects = append(objects, ownedExport(t, scheme, exportTestVersion, another, name))
			}
			for _, name := range tc.elsewhere {
				objects = append(objects, multicluster.RenderServiceExport(exportTestVersion, "other", name, dgd.Name))
			}
			fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, objects...)

			t.Log("Reconcile the pool's exports, as the Relay controller does when it publishes the Relay")
			recorder := events.NewFakeRecorder(20)
			r := newExportTestReconciler(fakeClient, recorder, tc.implementation, exportTestVersion)
			g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
			first := eventReasons(drainEvents(recorder))
			g.Expect(first).To(gomega.ConsistOf(tc.wantReasons))

			t.Log("Reconcile again: every export is in place and the condition is unchanged, so nothing is recorded")
			g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
			g.Expect(drainEvents(recorder)).To(gomega.BeEmpty())

			t.Log("Check the Relay's Exported condition")
			stored := &v1alpha1.DynamoPoolRelay{}
			g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(relay), stored)).To(gomega.Succeed())
			condition := meta.FindStatusCondition(stored.Status.Conditions, conditionExported)
			g.Expect(condition).NotTo(gomega.BeNil())
			g.Expect(condition.Reason).To(gomega.Equal(tc.wantCondition))
			wantStatus := metav1.ConditionFalse
			if tc.wantCondition == reasonExported {
				wantStatus = metav1.ConditionTrue
			}
			g.Expect(condition.Status).To(gomega.Equal(wantStatus))

			t.Log("Check which exports the Relay controls, and that other exports are untouched")
			var owned, others []string
			for i, export := range listServiceExports(t, fakeClient, exportTestVersion, dgd.Namespace) {
				if !metav1.IsControlledBy(&export, relay) {
					others = append(others, export.GetName())
					continue
				}
				owned = append(owned, export.GetName())
				g.Expect(export.GetLabels()).To(gomega.HaveKeyWithValue(commonconsts.KubeLabelDynamoGraphDeploymentName, dgd.Name), i)
			}
			g.Expect(owned).To(gomega.ConsistOf(tc.wantOwned))
			g.Expect(others).To(gomega.ConsistOf(append(append([]string{}, tc.foreign...), tc.ownedByAnother...)))
			var elsewhere []string
			for i, export := range listServiceExports(t, fakeClient, exportTestVersion, "other") {
				elsewhere = append(elsewhere, export.GetName())
				g.Expect(metav1.GetControllerOf(&export)).To(gomega.BeNil(), i)
			}
			g.Expect(elsewhere).To(gomega.ConsistOf(tc.elsewhere))
		})
	}
}

func TestWithdrawExports(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a Relay's exports, one of them already being deleted, a user's export and another Relay's")
	scheme := newExportTestScheme(t)
	relay := newExportTestRelay(newExportTestDGD("green"))
	terminating := ownedExport(t, scheme, exportTestVersion, relay, "green-old")
	terminating.SetFinalizers([]string{"example.com/hold"})
	terminating.SetDeletionTimestamp(&metav1.Time{Time: time.Now()})
	another := newExportTestRelay(newExportTestDGD("blue"))
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, relay,
		ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay"),
		ownedExport(t, scheme, exportTestVersion, relay, "green-frontend"),
		terminating,
		multicluster.RenderServiceExport(exportTestVersion, relay.Namespace, "green-web", relay.Name),
		ownedExport(t, scheme, exportTestVersion, another, "blue-frontend"))

	t.Log("Withdraw the Relay's exports, as the Relay controller does when it withdraws the Relay")
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)
	g.Expect(r.withdrawExports(ctx, relay)).To(gomega.Succeed())
	g.Expect(eventReasons(drainEvents(recorder))).To(gomega.ConsistOf("DeleteServiceExport", "DeleteServiceExport"))
	g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).
		To(gomega.ConsistOf("green-old", "green-web", "blue-frontend"))

	t.Log("The Relay's Exported condition says that the pool's exports are withdrawn")
	stored := &v1alpha1.DynamoPoolRelay{}
	g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(relay), stored)).To(gomega.Succeed())
	condition := meta.FindStatusCondition(stored.Status.Conditions, conditionExported)
	g.Expect(condition).NotTo(gomega.BeNil())
	g.Expect(condition.Status).To(gomega.Equal(metav1.ConditionFalse))
	g.Expect(condition.Reason).To(gomega.Equal(reasonWithdrawn))
}

func TestRelayControllerMaintainsThePoolsExports(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a DGD that a Relay can publish, and its Relay")
	scheme := newExportTestScheme(t)
	first := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(first)
	poolRelayService := multicluster.PoolRelayServiceName(relay.Name)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{},
		append([]client.Object{relay, first}, poolServices(t, scheme, relay, first, poolRelayService)...)...)
	r := newExportTestReconciler(fakeClient, events.NewFakeRecorder(50), configv1alpha1.MCSImplementationKarmada, exportTestVersion)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)}
	serviceKey := client.ObjectKey{Namespace: relay.Namespace, Name: poolRelayService}

	t.Log("Reconcile: the Relay's Service is published, and the pool's exports are created")
	_, err := r.Reconcile(ctx, request)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(fakeClient.Get(ctx, serviceKey, &corev1.Service{})).To(gomega.Succeed())
	g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).
		To(gomega.ConsistOf("green-pool-relay", "green-frontend"))

	t.Log("Delete the DGD: the Relay's Service and the pool's exports are withdrawn, and the Relay stays")
	g.Expect(fakeClient.Delete(ctx, first)).To(gomega.Succeed())
	_, err = r.Reconcile(ctx, request)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(apierrors.IsNotFound(fakeClient.Get(ctx, serviceKey, &corev1.Service{}))).To(gomega.BeTrue())
	g.Expect(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace)).To(gomega.BeEmpty())

	t.Log("Create a new DGD of the Relay's name with another frontend and its Service: the Relay publishes it, and its exports")
	second := newExportTestDGD("green", "Web")
	g.Expect(fakeClient.Create(ctx, second)).To(gomega.Succeed())
	for _, service := range poolServices(t, scheme, relay, second, poolRelayService) {
		g.Expect(fakeClient.Create(ctx, service)).To(gomega.Succeed())
	}
	_, err = r.Reconcile(ctx, request)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(fakeClient.Get(ctx, serviceKey, &corev1.Service{})).To(gomega.Succeed())
	g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).
		To(gomega.ConsistOf("green-pool-relay", "green-web"))

	t.Log("Leave the DGD without workers: the Relay withdraws, and the pool's exports with it")
	g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(second), second)).To(gomega.Succeed())
	second.Spec.Components = second.Spec.Components[1:]
	g.Expect(fakeClient.Update(ctx, second)).To(gomega.Succeed())
	_, err = r.Reconcile(ctx, request)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(apierrors.IsNotFound(fakeClient.Get(ctx, serviceKey, &corev1.Service{}))).To(gomega.BeTrue())
	g.Expect(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace)).To(gomega.BeEmpty())
}

func TestPoolExportsDoNotWaitForTheRelaysDeployment(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a publishable Relay while the API server refuses its Deployment")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(dgd)
	refused := apierrors.NewServiceUnavailable("the API server is restarting")
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			if _, ok := obj.(*appsv1.Deployment); ok {
				return refused
			}
			return c.Create(ctx, obj, opts...)
		},
	}, append([]client.Object{relay, dgd}, poolServices(t, scheme, relay, dgd, multicluster.PoolRelayServiceName(relay.Name))...)...)
	r := newExportTestReconciler(fakeClient, events.NewFakeRecorder(20), configv1alpha1.MCSImplementationKarmada, exportTestVersion)

	t.Log("Reconcile: the Deployment's failure is returned, and the pool's exports are created anyway")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)})
	g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("restarting")))
	g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).
		To(gomega.ConsistOf("green-pool-relay", "green-frontend"))
}

func TestRelayControllerReturnsAFailedExportWrite(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a publishable Relay while the API server refuses ServiceExports")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(dgd)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			if obj.GetObjectKind().GroupVersionKind().GroupKind() == multicluster.ServiceExportGroupKind {
				return apierrors.NewServiceUnavailable("the API server is restarting")
			}
			return c.Create(ctx, obj, opts...)
		},
	}, append([]client.Object{relay, dgd}, poolServices(t, scheme, relay, dgd, multicluster.PoolRelayServiceName(relay.Name))...)...)
	recorder := events.NewFakeRecorder(20)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationKarmada, exportTestVersion)

	t.Log("Reconcile: the Relay is published, and the failed export is returned for controller-runtime to retry with backoff")
	result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)})
	g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("restarting")))
	g.Expect(result).To(gomega.Equal(ctrl.Result{}))
	key := client.ObjectKey{Namespace: relay.Namespace, Name: multicluster.PoolRelayServiceName(relay.Name)}
	g.Expect(fakeClient.Get(ctx, key, &appsv1.Deployment{})).To(gomega.Succeed())
	g.Expect(countReasons(drainEvents(recorder), reasonExportFailed)).To(gomega.Equal(1))
}

func TestOwnershipConflictWithdrawsThePoolsExports(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a published pool's exports, and a user's Service that takes the Relay Service's name")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(dgd)
	foreign := &corev1.Service{Name: multicluster.PoolRelayServiceName(relay.Name), Namespace: relay.Namespace}
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, relay, dgd, foreign,
		ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay"),
		ownedExport(t, scheme, exportTestVersion, relay, "green-frontend"))
	r := newExportTestReconciler(fakeClient, events.NewFakeRecorder(20), configv1alpha1.MCSImplementationKarmada, exportTestVersion)

	t.Log("Reconcile: the conflict withdraws the Relay, and the pool's exports with it")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)})
	g.Expect(errors.Is(err, errOwnershipConflict)).To(gomega.BeTrue(), "%v", err)
	g.Expect(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace)).To(gomega.BeEmpty())
}

func TestReconcileExportsUsesTheResolvedVersion(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	const version = "v1beta1"

	t.Log("Serve ServiceExports at a version other than the tests' default, with an export to withdraw")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Web")
	relay := newExportTestRelay(dgd)
	fakeClient := newExportTestClient(scheme, version, interceptor.Funcs{}, append([]client.Object{
		dgd, relay, ownedExport(t, scheme, version, relay, "green-frontend"),
	}, poolServices(t, scheme, relay, dgd)...)...)

	t.Log("Reconcile with the gates resolved to that version: exports are listed, created and withdrawn at it")
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationKarmada, version)
	g.Expect(r.reconcileExports(context.Background(), relay, dgd)).To(gomega.Succeed())
	g.Expect(recorder.Events).To(gomega.HaveLen(3))
	exports := listServiceExports(t, fakeClient, version, dgd.Namespace)
	for i := range exports {
		g.Expect(exports[i].GetAPIVersion()).To(gomega.Equal("multicluster.x-k8s.io/" + version))
	}
	g.Expect(exportNames(exports)).To(gomega.ConsistOf("green-pool-relay", "green-web"))
}

func TestReconcileExportsToleratesAStaleCache(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	t.Log("The cache has not yet seen exports that the API server already has")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(dgd)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			return apierrors.NewAlreadyExists(
				schema.GroupResource{Group: multicluster.ServiceExportGroupKind.Group, Resource: "serviceexports"}, obj.GetName())
		},
	}, append([]client.Object{dgd, relay}, poolServices(t, scheme, relay, dgd)...)...)

	t.Log("Reconcile: AlreadyExists is no error, and nothing is announced")
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationKarmada, exportTestVersion)
	g.Expect(r.reconcileExports(context.Background(), relay, dgd)).To(gomega.Succeed())
	g.Expect(recorder.Events).To(gomega.BeEmpty())
}

func TestPoolExportsRetryFailedWrites(t *testing.T) {
	unavailable := apierrors.NewServiceUnavailable("the API server is restarting")
	for _, tc := range []struct {
		name string
		// failOnce fails the first write of the case's kind.
		failOnce func(failed *bool) interceptor.Funcs
		// withdraw withdraws the exports instead of reconciling them.
		withdraw bool
		// wantAfterFailure are the exports after the failed write: the failure
		// does not hold back the other export's write.
		wantAfterFailure []string
		want             []string
	}{
		{
			name: "a failed create",
			failOnce: func(failed *bool) interceptor.Funcs {
				return interceptor.Funcs{
					Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
						if !*failed {
							*failed = true
							return unavailable
						}
						return c.Create(ctx, obj, opts...)
					},
				}
			},
			wantAfterFailure: []string{"green-frontend"},
			want:             []string{"green-pool-relay", "green-frontend"},
		},
		{
			name: "a failed delete",
			failOnce: func(failed *bool) interceptor.Funcs {
				return interceptor.Funcs{
					Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
						if !*failed {
							*failed = true
							return unavailable
						}
						return c.Delete(ctx, obj, opts...)
					},
				}
			},
			withdraw:         true,
			wantAfterFailure: []string{"green-frontend"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			ctx := context.Background()

			t.Log("Create a Relay whose exports are to be created or withdrawn, and fail the first such write")
			scheme := newExportTestScheme(t)
			dgd := newExportTestDGD("green", "Frontend")
			relay := newExportTestRelay(dgd)
			objects := append([]client.Object{relay, dgd}, poolServices(t, scheme, relay, dgd)...)
			if tc.withdraw {
				objects = append(objects,
					ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay"),
					ownedExport(t, scheme, exportTestVersion, relay, "green-frontend"))
			}
			failed := false
			fakeClient := newExportTestClient(scheme, exportTestVersion, tc.failOnce(&failed), objects...)
			recorder := events.NewFakeRecorder(10)
			r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationKarmada, exportTestVersion)

			t.Log("The failure is returned, for controller-runtime to retry with backoff, and is announced")
			var err error
			if tc.withdraw {
				err = r.withdrawExports(ctx, relay)
			} else {
				err = r.reconcileExports(ctx, relay, dgd)
			}
			g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("restarting")))
			g.Expect(drainEvents(recorder)).To(gomega.ContainElement(gomega.HavePrefix(corev1.EventTypeWarning + " " + reasonExportFailed)))
			g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).To(gomega.ConsistOf(tc.wantAfterFailure))

			t.Log("The retry runs once the API server answers: the exports converge with no other change")
			if tc.withdraw {
				err = r.withdrawExports(ctx, relay)
			} else {
				err = r.reconcileExports(ctx, relay, dgd)
			}
			g.Expect(err).To(gomega.Succeed())
			g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).To(gomega.ConsistOf(tc.want))
		})
	}
}

func TestPoolExportsAreRepaired(t *testing.T) {
	for _, tc := range []struct {
		name  string
		drift func(export *unstructured.Unstructured)
	}{
		{
			name: "no DGD label",
			drift: func(export *unstructured.Unstructured) {
				labels := export.GetLabels()
				delete(labels, commonconsts.KubeLabelDynamoGraphDeploymentName)
				export.SetLabels(labels)
			},
		},
		{
			name: "a DGD label naming another DGD",
			drift: func(export *unstructured.Unstructured) {
				labels := export.GetLabels()
				labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = "blue"
				export.SetLabels(labels)
			},
		},
		{
			name: "a controller reference that no longer blocks the Relay's deletion",
			drift: func(export *unstructured.Unstructured) {
				references := export.GetOwnerReferences()
				references[0].BlockOwnerDeletion = new(false)
				export.SetOwnerReferences(references)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			ctx := context.Background()

			t.Log("Create a pool whose frontend export, which the Relay controls, has " + tc.name +
				", and carries the MCS implementation's status and others' finalizer, label and annotation")
			scheme := newExportTestScheme(t)
			dgd := newExportTestDGD("green", "Frontend")
			relay := newExportTestRelay(dgd)
			intact := ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay")
			changed := ownedExport(t, scheme, exportTestVersion, relay, "green-frontend")
			changed.Object["status"] = map[string]any{"conditions": []any{
				map[string]any{"type": "Ready", "status": "True", "reason": "Exported", "message": ""},
			}}
			changed.SetUID("export-frontend")
			changed.SetFinalizers([]string{"example.com/hold"})
			changed.SetAnnotations(map[string]string{"example.com/note": "kept"})
			changed.SetLabels(map[string]string{commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name, "team": "router"})
			tc.drift(changed)
			fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, append([]client.Object{
				dgd, relay, intact.DeepCopy(), changed.DeepCopy(),
			}, poolServices(t, scheme, relay, dgd)...)...)

			t.Log("Reconcile: the export is repaired, and the repair is recorded")
			recorder := events.NewFakeRecorder(10)
			r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationKarmada, exportTestVersion)
			g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
			g.Expect(eventReasons(drainEvents(recorder))).To(gomega.ConsistOf("RepairServiceExport"))

			t.Log("Check that it is the same export, with the controller's fields restored and everything else kept")
			repaired := multicluster.NewServiceExport(exportTestVersion)
			g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(changed), repaired)).To(gomega.Succeed())
			g.Expect(repaired.GetUID()).To(gomega.Equal(changed.GetUID()))
			g.Expect(repaired.GetLabels()).To(gomega.Equal(map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name, "team": "router",
			}))
			g.Expect(metav1.GetControllerOf(repaired)).To(gomega.Equal(metav1.GetControllerOf(intact)))
			g.Expect(repaired.GetAnnotations()).To(gomega.Equal(changed.GetAnnotations()))
			g.Expect(repaired.GetFinalizers()).To(gomega.Equal(changed.GetFinalizers()))
			g.Expect(repaired.Object["status"]).To(gomega.Equal(changed.Object["status"]))

			t.Log("Reconcile again: the export has the controller's fields, so nothing is applied or recorded")
			g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
			g.Expect(drainEvents(recorder)).To(gomega.BeEmpty())
		})
	}
}

func TestPoolExportsBeingDeletedAreLeftAlone(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a pool whose frontend export, which the Relay controls, has no DGD label and is being deleted")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(dgd)
	terminating := ownedExport(t, scheme, exportTestVersion, relay, "green-frontend")
	terminating.SetLabels(nil)
	terminating.SetFinalizers([]string{"example.com/hold"})
	terminating.SetDeletionTimestamp(&metav1.Time{Time: time.Now()})
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, append([]client.Object{
		dgd, relay, ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay"), terminating.DeepCopy(),
	}, poolServices(t, scheme, relay, dgd)...)...)

	t.Log("Reconcile: while its deletion is pending, the export is neither repaired nor created again")
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
	g.Expect(drainEvents(recorder)).To(gomega.BeEmpty())
	current := multicluster.NewServiceExport(exportTestVersion)
	g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(terminating), current)).To(gomega.Succeed())
	g.Expect(current.GetLabels()).To(gomega.BeEmpty())
	g.Expect(current.GetDeletionTimestamp()).NotTo(gomega.BeNil())
}

// staleCacheSetUp returns a pool's Relay and DGD; a cache that holds both of the
// Relay's exports as the API server held them when the cache read them, with the
// frontend export's DGD label removed; the API server, which holds the same
// exports for now; and the controller, which reads exports from the cache and
// writes them to the API server.
func staleCacheSetUp(t *testing.T) (
	*v1alpha1.DynamoPoolRelay, *v1beta1.DynamoGraphDeployment, client.WithWatch, *reconciler, *events.FakeRecorder, *runtime.Scheme,
) {
	t.Helper()
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(dgd)
	relayExport := ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay")
	relayExport.SetUID("export-relay")
	frontendExport := ownedExport(t, scheme, exportTestVersion, relay, "green-frontend")
	frontendExport.SetUID("export-frontend")
	frontendExport.SetLabels(nil)
	cache := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, relayExport.DeepCopy(), frontendExport.DeepCopy())
	apiServer := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, append([]client.Object{
		relay, dgd, relayExport.DeepCopy(), frontendExport.DeepCopy(),
	}, poolServices(t, scheme, relay, dgd)...)...)
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(apiServer, recorder, configv1alpha1.MCSImplementationKarmada, exportTestVersion)
	r.exports.cache = cache
	return relay, dgd, apiServer, r, recorder, scheme
}

func TestPoolExportsAreRepairedOnlyAsTheyWereRead(t *testing.T) {
	frontendKey := client.ObjectKey{Namespace: "inference", Name: "green-frontend"}

	t.Run("an export that a user took over stays theirs", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		relay, dgd, apiServer, r, recorder, _ := staleCacheSetUp(t)

		t.Log("After the cache read the export, a user removes its controller reference")
		export := multicluster.NewServiceExport(exportTestVersion)
		g.Expect(apiServer.Get(ctx, frontendKey, export)).To(gomega.Succeed())
		export.SetOwnerReferences(nil)
		g.Expect(apiServer.Update(ctx, export)).To(gomega.Succeed())

		t.Log("Reconcile: the apply names the version the cache read, so the API server refuses it, which is no error")
		g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
		g.Expect(drainEvents(recorder)).To(gomega.BeEmpty())

		t.Log("Check that the Relay did not take the export back")
		g.Expect(apiServer.Get(ctx, frontendKey, export)).To(gomega.Succeed())
		g.Expect(export.GetOwnerReferences()).To(gomega.BeEmpty())
		g.Expect(export.GetLabels()).To(gomega.BeEmpty())
	})

	t.Run("a deleted export is not created again by a repair", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		relay, dgd, apiServer, r, recorder, _ := staleCacheSetUp(t)

		t.Log("After the cache read the export, it is deleted")
		export := multicluster.NewServiceExport(exportTestVersion)
		export.SetNamespace(frontendKey.Namespace)
		export.SetName(frontendKey.Name)
		g.Expect(apiServer.Delete(ctx, export)).To(gomega.Succeed())

		t.Log("Reconcile: the apply names the UID the cache read, so the API server refuses it instead of creating the export")
		g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
		g.Expect(drainEvents(recorder)).To(gomega.BeEmpty())
		err := apiServer.Get(ctx, frontendKey, multicluster.NewServiceExport(exportTestVersion))
		g.Expect(apierrors.IsNotFound(err)).To(gomega.BeTrue(), "%v", err)
	})
}

func TestPoolExportsAreWithdrawnOnlyAsTheyWereRead(t *testing.T) {
	for _, tc := range []struct {
		name string
		// change changes the controller reference of an export, but not its UID.
		change func(t *testing.T, scheme *runtime.Scheme, export *unstructured.Unstructured)
	}{
		{
			name: "a user removes the controller reference",
			change: func(_ *testing.T, _ *runtime.Scheme, export *unstructured.Unstructured) {
				export.SetOwnerReferences(nil)
			},
		},
		{
			name: "another Relay takes the export over",
			change: func(t *testing.T, scheme *runtime.Scheme, export *unstructured.Unstructured) {
				export.SetOwnerReferences(nil)
				if err := controllerutil.SetControllerReference(newExportTestRelay(newExportTestDGD("blue")), export, scheme); err != nil {
					t.Fatalf("set the owner of ServiceExport %s: %v", export.GetName(), err)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			ctx := context.Background()

			t.Log("The cache lists both exports of a Relay that is being withdrawn")
			relay, _, apiServer, r, recorder, scheme := staleCacheSetUp(t)

			t.Log("Before the DELETE, " + tc.name + " in the API server, and the export keeps its UID")
			changed := multicluster.NewServiceExport(exportTestVersion)
			g.Expect(apiServer.Get(ctx, client.ObjectKey{Namespace: "inference", Name: "green-frontend"}, changed)).To(gomega.Succeed())
			tc.change(t, scheme, changed)
			g.Expect(apiServer.Update(ctx, changed)).To(gomega.Succeed())
			g.Expect(changed.GetUID()).To(gomega.BeEquivalentTo("export-frontend"))

			t.Log("Withdraw from the cache: the stale delete is refused, which is no error, and the Relay's other export is withdrawn")
			g.Expect(r.withdrawExports(ctx, relay)).To(gomega.Succeed())
			g.Expect(drainEvents(recorder)).To(gomega.ConsistOf(
				corev1.EventTypeNormal + " DeleteServiceExport Withdrew the export of Service green-pool-relay"))
			g.Expect(exportNames(listServiceExports(t, apiServer, exportTestVersion, relay.Namespace))).To(gomega.ConsistOf("green-frontend"))

			t.Log("Withdraw once the cache has the change: the Relay no longer controls the export, so it stays")
			r.exports.cache = apiServer
			g.Expect(r.withdrawExports(ctx, relay)).To(gomega.Succeed())
			g.Expect(drainEvents(recorder)).To(gomega.BeEmpty())
			remaining := listServiceExports(t, apiServer, exportTestVersion, relay.Namespace)
			g.Expect(remaining).To(gomega.HaveLen(1))
			g.Expect(remaining[0].GetUID()).To(gomega.BeEquivalentTo("export-frontend"))
			g.Expect(remaining[0].GetOwnerReferences()).To(gomega.Equal(changed.GetOwnerReferences()))
		})
	}

	t.Run("an export that is already gone is no error", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()

		t.Log("The cache lists both exports of a Relay that is being withdrawn")
		relay, _, apiServer, r, recorder, _ := staleCacheSetUp(t)

		t.Log("Before the DELETE, the frontend export is deleted in the API server")
		gone := multicluster.NewServiceExport(exportTestVersion)
		gone.SetNamespace(relay.Namespace)
		gone.SetName("green-frontend")
		g.Expect(apiServer.Delete(ctx, gone)).To(gomega.Succeed())

		t.Log("Withdraw from the cache: the missing export is no error, and the Relay's other export is withdrawn")
		g.Expect(r.withdrawExports(ctx, relay)).To(gomega.Succeed())
		g.Expect(drainEvents(recorder)).To(gomega.ConsistOf(
			corev1.EventTypeNormal + " DeleteServiceExport Withdrew the export of Service green-pool-relay"))
		g.Expect(listServiceExports(t, apiServer, exportTestVersion, relay.Namespace)).To(gomega.BeEmpty())
	})
}

// newClusterIDProperty returns the ClusterProperty that holds the cluster's ID in
// its ClusterSet, with value.
func newClusterIDProperty(value string) *unstructured.Unstructured {
	property := multicluster.NewClusterProperty()
	property.SetName(multicluster.ClusterIDProperty)
	property.Object["spec"] = map[string]any{"value": value}
	return property
}

func TestPoolExportsCheckTheClusterID(t *testing.T) {
	for _, tc := range []struct {
		name string
		// watched says whether the controller watches ClusterProperties, as it does
		// with a standard implementation in a cluster that serves them.
		watched bool
		// property is the cluster's ID in its ClusterSet, or empty without the
		// property.
		property string
		// emptyProperty adds the property with no value.
		emptyProperty bool
		readFails     bool
		wantMismatch  bool
	}{
		{name: "a Relay with the cluster's ID", watched: true, property: "east"},
		{name: "a Relay with another cluster ID", watched: true, property: "west", wantMismatch: true},
		{name: "a cluster without the property", watched: true},
		{name: "a property without a value gives no cluster ID", watched: true, emptyProperty: true},
		{name: "without the watch the check is skipped", property: "west"},
		{name: "a failed read fails the reconcile instead of skipping the check", watched: true, readFails: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			t.Log("Create a pool of cluster east, and the cluster's ID property if the case has one")
			scheme := newExportTestScheme(t)
			dgd := newExportTestDGD("green", "Frontend")
			relay := newExportTestRelay(dgd)
			objects := []client.Object{dgd, relay}
			if tc.property != "" || tc.emptyProperty {
				objects = append(objects, newClusterIDProperty(tc.property))
			}
			funcs := interceptor.Funcs{}
			if tc.readFails {
				funcs.Get = func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
					if obj.GetObjectKind().GroupVersionKind() == multicluster.ClusterPropertyGVK {
						return apierrors.NewForbidden(schema.GroupResource{Group: "about.k8s.io", Resource: "clusterproperties"}, key.Name, errors.New("denied"))
					}
					return c.Get(ctx, key, obj, opts...)
				}
			}
			fakeClient := newExportTestClient(scheme, exportTestVersion, funcs, objects...)
			recorder := events.NewFakeRecorder(10)
			r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)
			r.exports.clusterPropertyWatched = tc.watched

			t.Log("Reconcile the pool's exports, and look for the mismatch warning")
			err := r.reconcileExports(context.Background(), relay, dgd)
			recorded := drainEvents(recorder)
			if tc.readFails {
				g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("denied")))
				g.Expect(countReasons(recorded, reasonExportFailed)).To(gomega.Equal(1))
				return
			}
			g.Expect(err).NotTo(gomega.HaveOccurred())
			if !tc.wantMismatch {
				g.Expect(countReasons(recorded, reasonClusterIDMismatch)).To(gomega.BeZero())
				return
			}
			g.Expect(recorded).To(gomega.ContainElement(gomega.SatisfyAll(
				gomega.HavePrefix(corev1.EventTypeWarning+" "+reasonClusterIDMismatch),
				gomega.ContainSubstring(`clusterId "east"`),
				gomega.ContainSubstring(`the ID "west"`),
			)))
		})
	}
}

func TestPoolsFollowTheClusterIDProperty(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create two pools of cluster east in a cluster whose ID property says east, read through the watch's cache")
	scheme := newExportTestScheme(t)
	green := newExportTestDGD("green", "Frontend")
	blue := newExportTestDGD("blue", "Frontend")
	greenRelay, blueRelay := newExportTestRelay(green), newExportTestRelay(blue)
	property := newClusterIDProperty("east")
	objects := []client.Object{green, blue, greenRelay, blueRelay, property}
	objects = append(objects, poolServices(t, scheme, greenRelay, green)...)
	objects = append(objects, poolServices(t, scheme, blueRelay, blue)...)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, objects...)
	recorder := events.NewFakeRecorder(20)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)
	r.exports.clusterPropertyWatched = true
	pools := []ctrl.Request{
		{NamespacedName: client.ObjectKeyFromObject(greenRelay)},
		{NamespacedName: client.ObjectKeyFromObject(blueRelay)},
	}

	for _, step := range []struct {
		description string
		// relayClusterID, when set, is the clusterId that users give both Relays,
		// whose own watch requeues them.
		relayClusterID string
		// propertyValue, when set, is the property's new value.
		propertyValue string
		// deleteProperty deletes the property, and createProperty, when set,
		// creates it again with that value.
		deleteProperty bool
		createProperty string
		// wantMismatches is how many Relays record a mismatch.
		wantMismatches int
	}{
		{description: "Reconcile both pools: their IDs match the property"},
		{
			description:    "The cluster is renamed west: users correct both Relays first, and their watch requeues them while the cache still says east",
			relayClusterID: "west", wantMismatches: 2,
		},
		{
			description:   "The cache sees the property's new value: its event requeues both Relays, which no longer report a mismatch",
			propertyValue: "west",
		},
		{description: "The value changes again: both Relays report the mismatch", propertyValue: "south", wantMismatches: 2},
		{description: "The property is deleted: its event requeues both Relays, which skip the check", deleteProperty: true},
		{
			description:    "The property is created again with another value: its event requeues both Relays, which report the mismatch",
			createProperty: "north", wantMismatches: 2,
		},
	} {
		t.Log(step.description)
		requests := pools
		switch {
		case step.relayClusterID != "":
			for _, relay := range []*v1alpha1.DynamoPoolRelay{greenRelay, blueRelay} {
				g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(relay), relay)).To(gomega.Succeed())
				relay.Spec.ClusterID = step.relayClusterID
				g.Expect(fakeClient.Update(ctx, relay)).To(gomega.Succeed())
			}
		case step.propertyValue != "":
			g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(property), property)).To(gomega.Succeed())
			before := property.DeepCopy()
			property.Object["spec"] = map[string]any{"value": step.propertyValue}
			g.Expect(fakeClient.Update(ctx, property)).To(gomega.Succeed())
			g.Expect(clusterIDPropertyEvents.Update(event.UpdateEvent{ObjectOld: before, ObjectNew: property})).To(gomega.BeTrue())
			requests = r.mapClusterPropertyToRelays(ctx, property)
		case step.deleteProperty:
			g.Expect(fakeClient.Delete(ctx, property)).To(gomega.Succeed())
			g.Expect(clusterIDPropertyEvents.Delete(event.DeleteEvent{Object: property})).To(gomega.BeTrue())
			requests = r.mapClusterPropertyToRelays(ctx, property)
		case step.createProperty != "":
			created := newClusterIDProperty(step.createProperty)
			g.Expect(fakeClient.Create(ctx, created)).To(gomega.Succeed())
			g.Expect(clusterIDPropertyEvents.Create(event.CreateEvent{Object: created})).To(gomega.BeTrue())
			requests = r.mapClusterPropertyToRelays(ctx, created)
		}
		for _, request := range requests {
			relay := &v1alpha1.DynamoPoolRelay{}
			g.Expect(fakeClient.Get(ctx, request.NamespacedName, relay)).To(gomega.Succeed())
			dgd := &v1beta1.DynamoGraphDeployment{}
			g.Expect(fakeClient.Get(ctx, request.NamespacedName, dgd)).To(gomega.Succeed())
			g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
		}
		g.Expect(countReasons(drainEvents(recorder), reasonClusterIDMismatch)).To(gomega.Equal(step.wantMismatches))
	}
}

func TestPoolExportsConvergePastAFailedWrite(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a pool whose frontend was renamed from Frontend to Web, in a cluster whose ID differs from the " +
		"Relay's, while a quota refuses every new export")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Web")
	relay := newExportTestRelay(dgd)
	quota := apierrors.NewForbidden(schema.GroupResource{Group: multicluster.ServiceExportGroupKind.Group, Resource: "serviceexports"},
		"green-web", errors.New("exceeded quota"))
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			if obj.GetObjectKind().GroupVersionKind().GroupKind() == multicluster.ServiceExportGroupKind {
				return quota
			}
			return c.Create(ctx, obj, opts...)
		},
	}, append([]client.Object{
		relay, dgd, newClusterIDProperty("west"),
		ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay"),
		ownedExport(t, scheme, exportTestVersion, relay, "green-frontend"),
	}, poolServices(t, scheme, relay, dgd)...)...)
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)
	r.exports.clusterPropertyWatched = true

	t.Log("Reconcile: the refused create is returned, but the stale export is withdrawn, which frees the quota, " +
		"and the mismatch is still reported")
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.MatchError(gomega.ContainSubstring("exceeded quota")))
	g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).To(gomega.ConsistOf("green-pool-relay"))
	g.Expect(eventReasons(drainEvents(recorder))).To(gomega.ConsistOf("DeleteServiceExport", reasonExportFailed, reasonClusterIDMismatch))
	condition := meta.FindStatusCondition(relay.Status.Conditions, conditionExported)
	g.Expect(condition).NotTo(gomega.BeNil())
	g.Expect(condition.Reason).To(gomega.Equal(reasonExportFailed))
	g.Expect(condition.Message).To(gomega.And(gomega.ContainSubstring("exceeded quota"), gomega.ContainSubstring(`clusterId "east"`)))
}

func TestPoolExportProblemsAreReportedPastAFailedWithdrawal(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a pool whose frontend was renamed from Frontend to Web, in a cluster whose ID differs from the " +
		"Relay's, while the API server refuses deletes")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Web")
	relay := newExportTestRelay(dgd)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{
		Delete: func(context.Context, client.WithWatch, client.Object, ...client.DeleteOption) error {
			return apierrors.NewServiceUnavailable("the API server is restarting")
		},
	}, append([]client.Object{
		relay, dgd, newClusterIDProperty("west"),
		ownedExport(t, scheme, exportTestVersion, relay, "green-frontend"),
	}, poolServices(t, scheme, relay, dgd)...)...)
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)
	r.exports.clusterPropertyWatched = true

	t.Log("Reconcile: the refused delete is returned, the new exports are created, and the mismatch is still reported")
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.MatchError(gomega.ContainSubstring("restarting")))
	g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).
		To(gomega.ConsistOf("green-pool-relay", "green-web", "green-frontend"))
	g.Expect(eventReasons(drainEvents(recorder))).To(gomega.ConsistOf(
		"CreateServiceExport", "CreateServiceExport", reasonExportFailed, reasonClusterIDMismatch))
}

func TestPoolExportsSurviveAFailedServiceRead(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a pool with both exports while reading its Services fails")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend")
	relay := newExportTestRelay(dgd)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*corev1.Service); ok {
				return apierrors.NewServiceUnavailable("the API server is restarting")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	}, append([]client.Object{
		relay, dgd,
		ownedExport(t, scheme, exportTestVersion, relay, "green-pool-relay"),
		ownedExport(t, scheme, exportTestVersion, relay, "green-frontend"),
	}, poolServices(t, scheme, relay, dgd)...)...)
	r := newExportTestReconciler(fakeClient, events.NewFakeRecorder(10), configv1alpha1.MCSImplementationStandard, exportTestVersion)

	t.Log("Reconcile: the failure is returned, and the exports stay until the Services can be read again")
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.MatchError(gomega.ContainSubstring("restarting")))
	g.Expect(exportNames(listServiceExports(t, fakeClient, exportTestVersion, relay.Namespace))).
		To(gomega.ConsistOf("green-pool-relay", "green-frontend"))
}

func TestPoolExportWarningsFollowTheExportedCondition(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a pool without a frontend")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green")
	relay := newExportTestRelay(dgd)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{},
		append([]client.Object{relay, dgd}, poolServices(t, scheme, relay, dgd)...)...)
	recorder := events.NewFakeRecorder(10)
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)

	t.Log("Reconcile: the condition reports the problem, and its Warning is recorded once")
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
	g.Expect(eventReasons(drainEvents(recorder))).To(gomega.ConsistOf("CreateServiceExport", reasonNoFrontend))

	t.Log("A DGD status write changes only the DGD's resourceVersion, so the condition is unchanged and nothing is recorded")
	g.Expect(fakeClient.Get(ctx, client.ObjectKeyFromObject(dgd), dgd)).To(gomega.Succeed())
	dgd.Status.State = "pending"
	g.Expect(fakeClient.Status().Update(ctx, dgd)).To(gomega.Succeed())
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
	g.Expect(drainEvents(recorder)).To(gomega.BeEmpty())

	t.Log("A frontend is added: the condition turns True, and only the new export is recorded")
	dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "Frontend", ComponentType: v1beta1.ComponentTypeFrontend,
	})
	g.Expect(fakeClient.Update(ctx, dgd)).To(gomega.Succeed())
	for _, service := range poolServices(t, scheme, relay, dgd, multicluster.PoolRelayServiceName(dgd.Name)) {
		g.Expect(fakeClient.Create(ctx, service)).To(gomega.Succeed())
	}
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
	g.Expect(eventReasons(drainEvents(recorder))).To(gomega.ConsistOf("CreateServiceExport"))
	condition := meta.FindStatusCondition(relay.Status.Conditions, conditionExported)
	g.Expect(condition).NotTo(gomega.BeNil())
	g.Expect(condition.Status).To(gomega.Equal(metav1.ConditionTrue))
	g.Expect(condition.Message).To(gomega.Equal("Exported Services green-pool-relay, green-frontend through MCS implementation mcs"))
}

func TestMapServiceToRelays(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create two pools in namespace inference, one with a frontend named Frontend")
	scheme := newExportTestScheme(t)
	green := newExportTestDGD("green", "Frontend")
	blue := newExportTestDGD("blue")
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{},
		green, blue, newExportTestRelay(green), newExportTestRelay(blue))
	r := newExportTestReconciler(fakeClient, events.NewFakeRecorder(1), configv1alpha1.MCSImplementationStandard, exportTestVersion)

	t.Log("A Service requeues the Relays whose DGD exports a Service of its name")
	g.Expect(r.mapServiceToRelays(ctx, &corev1.Service{Name: "green-frontend", Namespace: "inference"})).
		To(gomega.ConsistOf(ctrl.Request{Namespace: "inference", Name: "green"}))
	g.Expect(r.mapServiceToRelays(ctx, &corev1.Service{Name: "blue-pool-relay", Namespace: "inference"})).
		To(gomega.ConsistOf(ctrl.Request{Namespace: "inference", Name: "blue"}))
	g.Expect(r.mapServiceToRelays(ctx, &corev1.Service{Name: "blue-frontend", Namespace: "inference"})).To(gomega.BeEmpty())
	g.Expect(r.mapServiceToRelays(ctx, &corev1.Service{Name: "green-frontend", Namespace: "other"})).To(gomega.BeEmpty())

	t.Log("A Service's creation and deletion count, and an update only when it changes the Service's controller")
	scheme = newExportTestScheme(t)
	service := &corev1.Service{Name: "green-frontend", Namespace: "inference"}
	g.Expect(serviceControllerChanged.Create(event.CreateEvent{Object: service})).To(gomega.BeTrue())
	g.Expect(serviceControllerChanged.Delete(event.DeleteEvent{Object: service})).To(gomega.BeTrue())
	relabeled := service.DeepCopy()
	relabeled.Labels = map[string]string{"team": "router"}
	g.Expect(serviceControllerChanged.Update(event.UpdateEvent{ObjectOld: service, ObjectNew: relabeled})).To(gomega.BeFalse())
	controlled := service.DeepCopy()
	g.Expect(controllerutil.SetControllerReference(green, controlled, scheme)).To(gomega.Succeed())
	g.Expect(serviceControllerChanged.Update(event.UpdateEvent{ObjectOld: service, ObjectNew: controlled})).To(gomega.BeTrue())
	g.Expect(serviceControllerChanged.Update(event.UpdateEvent{ObjectOld: controlled, ObjectNew: service})).To(gomega.BeTrue())
}

func TestTruncateKeepsWholeCharacters(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	g.Expect(truncate("short", 10)).To(gomega.Equal("short"))
	g.Expect(truncate("ab€cd", 4)).To(gomega.Equal("ab"))
	g.Expect(truncate("ab€cd", 5)).To(gomega.Equal("ab€"))
	g.Expect(truncate(strings.Repeat("x", 2000), maxEventNote)).To(gomega.HaveLen(maxEventNote))
}

func TestMapServiceExportToRelays(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create pools green and one with a dot in its name, a DGD without a Relay, a pool in another namespace, " +
		"and a pool there whose frontend Service has the name of green's PoolRelay Service")
	scheme := newExportTestScheme(t)
	green := newExportTestDGD("green", "Frontend", "Web")
	dotted := newExportTestDGD("green3-0.6b", "Frontend")
	unpooled := newExportTestDGD("blue", "Frontend")
	elsewhere := newExportTestDGD("green", "Frontend")
	elsewhere.Namespace = "other"
	sameServiceName := newExportTestDGD("green-pool", "Relay")
	sameServiceName.Namespace = "other"
	greenRelay := newExportTestRelay(green)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{},
		green, dotted, unpooled, elsewhere, sameServiceName,
		greenRelay, newExportTestRelay(dotted), newExportTestRelay(elsewhere), newExportTestRelay(sameServiceName))
	r := newExportTestReconciler(fakeClient, events.NewFakeRecorder(1), configv1alpha1.MCSImplementationStandard, exportTestVersion)

	t.Log("Create an export that a Deployment named blue controls")
	controlledByDeployment := multicluster.RenderServiceExport(exportTestVersion, "inference", "blue-cache", "blue")
	if err := controllerutil.SetControllerReference(
		&appsv1.Deployment{Name: "blue", Namespace: "inference", UID: uuid.NewUUID()}, controlledByDeployment, scheme,
	); err != nil {
		t.Fatal(err)
	}

	request := func(dgd *v1beta1.DynamoGraphDeployment) ctrl.Request {
		return ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)}
	}
	for _, tc := range []struct {
		name   string
		export *unstructured.Unstructured
		want   []ctrl.Request
	}{
		{
			name:   "a user's export of a PoolRelay Service",
			export: multicluster.RenderServiceExport(exportTestVersion, "inference", "green-pool-relay", "green"),
			want:   []ctrl.Request{request(green)},
		},
		{
			name:   "a frontend Service under its component's name",
			export: multicluster.RenderServiceExport(exportTestVersion, "inference", "green-web", "green"),
			want:   []ctrl.Request{request(green)},
		},
		{
			name:   "the frontend Service of a DGD with a dot in its name",
			export: multicluster.RenderServiceExport(exportTestVersion, "inference", "green3-0-6b-frontend", "green3-0.6b"),
			want:   []ctrl.Request{request(dotted)},
		},
		{
			name:   "an export the Relay controls for a frontend its DGD no longer has",
			export: ownedExport(t, scheme, exportTestVersion, greenRelay, "green-old"),
			want:   []ctrl.Request{request(green)},
		},
		{
			name:   "a Service of a DGD without a Relay",
			export: multicluster.RenderServiceExport(exportTestVersion, "inference", "blue-frontend", "blue"),
		},
		{
			name:   "another Service",
			export: multicluster.RenderServiceExport(exportTestVersion, "inference", "green", "green"),
		},
		{
			name:   "an export that another kind controls",
			export: controlledByDeployment,
		},
	} {
		t.Log("Map an event for " + tc.name + " in namespace inference to the Relays it concerns")
		g.Expect(r.mapServiceExportToRelays(ctx, tc.export)).To(gomega.ConsistOf(tc.want), tc.name)
	}
}

func TestMapClusterPropertyToRelays(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	t.Log("Create Relays in two namespaces and one in an ephemeral namespace")
	scheme := newExportTestScheme(t)
	green := newExportTestRelay(newExportTestDGD("green"))
	blue := newExportTestRelay(newExportTestDGD("blue"))
	blue.Namespace = "team-b"
	ephemeral := newExportTestRelay(newExportTestDGD("red"))
	ephemeral.Namespace = "inference-ephemeral"
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, green, blue, ephemeral)
	r := newExportTestReconciler(fakeClient, events.NewFakeRecorder(1), configv1alpha1.MCSImplementationStandard, exportTestVersion)

	t.Log("An event of the cluster ID property requeues every Relay that the operator manages")
	g.Expect(r.mapClusterPropertyToRelays(context.Background(), newClusterIDProperty("east"))).To(gomega.ConsistOf(
		ctrl.Request{NamespacedName: client.ObjectKeyFromObject(green)},
		ctrl.Request{NamespacedName: client.ObjectKeyFromObject(blue)},
	))
}

func TestPoolExportWatchPredicates(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	t.Log("An update of a user's export does not count; its creation and deletion do")
	export := multicluster.RenderServiceExport(exportTestVersion, "inference", "green-frontend", "green")
	statusChanged := export.DeepCopy()
	statusChanged.Object["status"] = map[string]any{"conditions": []any{map[string]any{"type": "Ready", "status": "True"}}}
	g.Expect(serviceExportChanged.Update(event.UpdateEvent{ObjectOld: export, ObjectNew: statusChanged})).To(gomega.BeFalse())
	g.Expect(serviceExportChanged.Create(event.CreateEvent{Object: export})).To(gomega.BeTrue())
	g.Expect(serviceExportChanged.Delete(event.DeleteEvent{Object: export})).To(gomega.BeTrue())

	t.Log("Any update of an export with a controller counts, also one that adds or removes the controller reference")
	relay := newExportTestRelay(newExportTestDGD("green"))
	owned := ownedExport(t, newExportTestScheme(t), exportTestVersion, relay, "green-frontend")
	unlabeled := owned.DeepCopy()
	unlabeled.SetLabels(nil)
	released := owned.DeepCopy()
	released.SetOwnerReferences(nil)
	g.Expect(serviceExportChanged.Update(event.UpdateEvent{ObjectOld: owned, ObjectNew: unlabeled})).To(gomega.BeTrue())
	g.Expect(serviceExportChanged.Update(event.UpdateEvent{ObjectOld: owned, ObjectNew: released})).To(gomega.BeTrue())
	g.Expect(serviceExportChanged.Update(event.UpdateEvent{ObjectOld: released, ObjectNew: owned})).To(gomega.BeTrue())

	t.Log("A namespace-restricted operator still sees the cluster-scoped ClusterProperty, and filters other objects by namespace")
	filter := eventFilter(
		&configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "inference"}},
		&commoncontroller.RuntimeConfig{},
	)
	g.Expect(filter.Create(event.CreateEvent{Object: newClusterIDProperty("east")})).To(gomega.BeTrue())
	g.Expect(filter.Create(event.CreateEvent{Object: export})).To(gomega.BeTrue())
	elsewhere := multicluster.RenderServiceExport(exportTestVersion, "other", "green-frontend", "green")
	g.Expect(filter.Create(event.CreateEvent{Object: elsewhere})).To(gomega.BeFalse())
}

func TestClusterIDPropertyEvents(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	property := newClusterIDProperty("east")
	other := newClusterIDProperty("east")
	other.SetName("clusterset.k8s.io")

	t.Log("The creation and deletion of the cluster ID property count, another property's do not")
	g.Expect(clusterIDPropertyEvents.Create(event.CreateEvent{Object: property})).To(gomega.BeTrue())
	g.Expect(clusterIDPropertyEvents.Delete(event.DeleteEvent{Object: property})).To(gomega.BeTrue())
	g.Expect(clusterIDPropertyEvents.Create(event.CreateEvent{Object: other})).To(gomega.BeFalse())
	g.Expect(clusterIDPropertyEvents.Delete(event.DeleteEvent{Object: other})).To(gomega.BeFalse())

	t.Log("An update counts when it changes the cluster ID property's value")
	relabeled := property.DeepCopy()
	relabeled.SetLabels(map[string]string{"team": "router"})
	otherChanged := other.DeepCopy()
	otherChanged.Object["spec"] = map[string]any{"value": "west"}
	g.Expect(clusterIDPropertyEvents.Update(event.UpdateEvent{ObjectOld: property, ObjectNew: newClusterIDProperty("west")})).To(gomega.BeTrue())
	g.Expect(clusterIDPropertyEvents.Update(event.UpdateEvent{ObjectOld: property, ObjectNew: relabeled})).To(gomega.BeFalse())
	g.Expect(clusterIDPropertyEvents.Update(event.UpdateEvent{ObjectOld: other, ObjectNew: otherChanged})).To(gomega.BeFalse())
}

// failingRESTMapper fails every REST mapping, as when discovery is down.
type failingRESTMapper struct {
	meta.RESTMapper
}

func (failingRESTMapper) RESTMapping(schema.GroupKind, ...string) (*meta.RESTMapping, error) {
	return nil, errors.New("discovery is down")
}

func TestClusterPropertyServed(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	t.Log("A cluster that serves the About API gets the watch")
	served := meta.NewDefaultRESTMapper(nil)
	served.Add(multicluster.ClusterPropertyGVK, meta.RESTScopeRoot)
	g.Expect(clusterPropertyServed(served)).To(gomega.BeTrue())

	t.Log("Without the API, the controller starts without the watch, and so skips the check")
	g.Expect(clusterPropertyServed(meta.NewDefaultRESTMapper(nil))).To(gomega.BeFalse())

	t.Log("A failed discovery fails the start")
	_, err := clusterPropertyServed(failingRESTMapper{RESTMapper: meta.NewDefaultRESTMapper(nil)})
	g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("discovery is down")))
}

func TestWatchExportsChecksTheClusterIDOnlyWhereItCan(t *testing.T) {
	aboutAPI := meta.NewDefaultRESTMapper(nil)
	aboutAPI.Add(multicluster.ClusterPropertyGVK, meta.RESTScopeRoot)
	for _, tc := range []struct {
		name           string
		implementation configv1alpha1.MCSImplementation
		mapper         meta.RESTMapper
		wantWatched    bool
		wantError      string
	}{
		{
			name:           "standard MCS in a cluster that serves the About API checks the cluster ID",
			implementation: configv1alpha1.MCSImplementationStandard, mapper: aboutAPI, wantWatched: true,
		},
		{
			name:           "Karmada names its member clusters itself, so it does not check",
			implementation: configv1alpha1.MCSImplementationKarmada, mapper: aboutAPI,
		},
		{
			name:           "without the About API, the controller starts without the check",
			implementation: configv1alpha1.MCSImplementationStandard, mapper: meta.NewDefaultRESTMapper(nil),
		},
		{
			name:           "a failed discovery fails the start",
			implementation: configv1alpha1.MCSImplementationStandard,
			mapper:         failingRESTMapper{RESTMapper: meta.NewDefaultRESTMapper(nil)},
			wantError:      "discovery is down",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			t.Log("Create a manager whose REST mapper serves the case's APIs; the test never starts it")
			scheme := newExportTestScheme(t)
			manager, err := ctrl.NewManager(&rest.Config{Host: "https://127.0.0.1:1"}, ctrl.Options{
				Scheme:  scheme,
				Metrics: server.Options{BindAddress: "0"},
				MapperProvider: func(*rest.Config, *http.Client) (meta.RESTMapper, error) {
					return tc.mapper, nil
				},
			})
			g.Expect(err).NotTo(gomega.HaveOccurred())

			t.Log("Add the watches of the pool's exports, as Setup does when the gate is on")
			r := &reconciler{Client: manager.GetClient(), scheme: scheme}
			_, err = r.watchExports(manager, ctrl.NewControllerManagedBy(manager).For(&v1alpha1.DynamoPoolRelay{}),
				&configv1alpha1.OperatorConfiguration{
					MultiCluster: configv1alpha1.MultiClusterConfiguration{MCSImplementation: tc.implementation},
				},
				&commoncontroller.RuntimeConfig{
					Gate: features.Gates{MultiClusterExport: true, ServiceExportVersion: exportTestVersion},
				})
			if tc.wantError != "" {
				g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring(tc.wantError)))
				return
			}
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(r.exports.clusterPropertyWatched).To(gomega.Equal(tc.wantWatched))
		})
	}
}

// relatedRecorder records each event's reason with the kind and name of the
// event's related object.
type relatedRecorder struct {
	recorded []string
}

func (r *relatedRecorder) Eventf(_, related runtime.Object, _, reason, _, _ string, _ ...any) {
	name := "none"
	switch object := related.(type) {
	case *unstructured.Unstructured:
		name = object.GetKind() + "/" + object.GetName()
	case *v1beta1.DynamoGraphDeployment:
		name = "DynamoGraphDeployment/" + object.Name
	}
	r.recorded = append(r.recorded, reason+" "+name)
}

func TestPoolExportEventsNameTheirObjects(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create a pool with a frontend whose Service no Service can be named, an export to repair and one to " +
		"withdraw, in a cluster whose ID in its ClusterSet differs from the Relay's")
	scheme := newExportTestScheme(t)
	dgd := newExportTestDGD("green", "Frontend", strings.Repeat("w", 60))
	relay := newExportTestRelay(dgd)
	unlabeled := ownedExport(t, scheme, exportTestVersion, relay, "green-frontend")
	unlabeled.SetLabels(nil)
	fakeClient := newExportTestClient(scheme, exportTestVersion, interceptor.Funcs{}, append([]client.Object{
		relay, dgd, unlabeled, ownedExport(t, scheme, exportTestVersion, relay, "green-old"), newClusterIDProperty("west"),
	}, poolServices(t, scheme, relay, dgd)...)...)
	recorder := &relatedRecorder{}
	r := newExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationStandard, exportTestVersion)
	r.exports.clusterPropertyWatched = true

	t.Log("Reconcile: each event names the export it reports, or the DGD or property that its problem is about")
	g.Expect(r.reconcileExports(ctx, relay, dgd)).To(gomega.Succeed())
	g.Expect(recorder.recorded).To(gomega.ConsistOf(
		"CreateServiceExport ServiceExport/green-pool-relay",
		"RepairServiceExport ServiceExport/green-frontend",
		"DeleteServiceExport ServiceExport/green-old",
		"InvalidServiceName DynamoGraphDeployment/green",
		"ClusterIDMismatch ClusterProperty/"+multicluster.ClusterIDProperty,
	))

	t.Log("Remove the frontends: the withdrawal names its export, and the missing frontend the DGD")
	recorder.recorded = nil
	g.Expect(r.reconcileExports(ctx, relay, newExportTestDGD("green"))).To(gomega.Succeed())
	g.Expect(recorder.recorded).To(gomega.ConsistOf(
		"DeleteServiceExport ServiceExport/green-frontend",
		"NoFrontend DynamoGraphDeployment/green",
		"ClusterIDMismatch ClusterProperty/"+multicluster.ClusterIDProperty,
	))

	t.Log("With Karmada, a Service name too long to import names the DGD too")
	recorder.recorded = nil
	r.exports.config.MultiCluster.MCSImplementation = configv1alpha1.MCSImplementationKarmada
	long := newExportTestDGD(strings.Repeat("a", 45), "Frontend")
	g.Expect(r.reconcileExports(ctx, relay, long)).To(gomega.Succeed())
	g.Expect(recorder.recorded).To(gomega.ContainElement("ServiceNameTooLong DynamoGraphDeployment/" + long.Name))
}
