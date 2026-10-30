/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/multicluster"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// newServiceExportTestClient returns a fake client that also serves the
// ServiceExport and ClusterProperty kinds, which the operator's scheme does not
// register, and the DGD's status subresource.
func newServiceExportTestClient(
	scheme *runtime.Scheme,
	funcs interceptor.Funcs,
	objects ...client.Object,
) client.WithWatch {
	mapper := meta.NewDefaultRESTMapper(nil)
	for gvk := range scheme.AllKnownTypes() {
		mapper.Add(gvk, meta.RESTScopeNamespace)
	}
	mapper.Add(multicluster.ServiceExportGVK, meta.RESTScopeNamespace)
	mapper.Add(multicluster.ClusterPropertyGVK, meta.RESTScopeRoot)
	return fake.NewClientBuilder().
		WithScheme(scheme).
		WithRESTMapper(mapper).
		WithStatusSubresource(&v1beta1.DynamoGraphDeployment{}).
		WithInterceptorFuncs(funcs).
		WithObjects(objects...).
		Build()
}

// listServiceExports returns the ServiceExports in namespace.
func listServiceExports(t *testing.T, c client.Client, namespace string) []unstructured.Unstructured {
	t.Helper()
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(multicluster.ServiceExportGVK.GroupVersion().WithKind("ServiceExportList"))
	if err := c.List(context.Background(), list, client.InNamespace(namespace)); err != nil {
		t.Fatalf("list ServiceExports: %v", err)
	}
	return list.Items
}

// newPoolTestDGD returns a DGD named name in namespace dynamo with a decode
// worker and a frontend component for each of frontends.
func newPoolTestDGD(name string, frontends ...string) *v1beta1.DynamoGraphDeployment {
	dgd := &v1beta1.DynamoGraphDeployment{Name: name, Namespace: "dynamo", UID: uuid.NewUUID()}
	dgd.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "VllmDecodeWorker", ComponentType: v1beta1.ComponentTypeDecode},
	}
	for _, frontend := range frontends {
		dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: frontend, ComponentType: v1beta1.ComponentTypeFrontend,
		})
	}
	return dgd
}

// newPoolTestRelay returns the DynamoPoolRelay that makes dgd a pool of cluster
// east.
func newPoolTestRelay(dgd *v1beta1.DynamoGraphDeployment) *v1alpha1.DynamoPoolRelay {
	return &v1alpha1.DynamoPoolRelay{
		Name: dgd.Name, Namespace: dgd.Namespace,
		Spec: v1alpha1.DynamoPoolRelaySpec{ClusterID: "east"},
	}
}

// ownedServiceExport returns the export of the Service named name that dgd
// controls.
func ownedServiceExport(
	t *testing.T,
	scheme *runtime.Scheme,
	dgd *v1beta1.DynamoGraphDeployment,
	name string,
) *unstructured.Unstructured {
	t.Helper()
	export := multicluster.RenderServiceExport(dgd.Namespace, name, dgd.Name)
	if err := controllerutil.SetControllerReference(dgd, export, scheme); err != nil {
		t.Fatalf("set the owner of ServiceExport %s: %v", name, err)
	}
	return export
}

// withExportConditions sets the conditions an MCS implementation writes on export.
func withExportConditions(export *unstructured.Unstructured, conditions ...metav1.Condition) *unstructured.Unstructured {
	items := make([]any, 0, len(conditions))
	for _, condition := range conditions {
		items = append(items, map[string]any{
			"type": condition.Type, "status": string(condition.Status),
			"reason": condition.Reason, "message": condition.Message,
		})
	}
	export.Object["status"] = map[string]any{"conditions": items}
	return export
}

// imported is the condition a standard implementation writes on an export it has
// imported.
var imported = metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, Reason: "Exported"}

// newServiceExportTestReconciler returns the export reconciler with the gate
// that implementation resolves to, reading exports through c, in a cluster with
// clusterSetID as its ID in its ClusterSet.
func newServiceExportTestReconciler(
	c client.Client,
	recorder events.EventRecorder,
	implementation configv1alpha1.MCSImplementation,
	clusterSetID string,
) *dgdServiceExportReconciler {
	config := &configv1alpha1.OperatorConfiguration{
		MultiCluster: configv1alpha1.MultiClusterConfiguration{MCSImplementation: implementation},
	}
	gates := features.Gates{MultiClusterExport: implementation != configv1alpha1.MCSImplementationNone}
	return newDGDServiceExportReconciler(newDGDResourceSyncer(c, recorder), config, gates, c,
		func(context.Context) string { return clusterSetID })
}

func TestDGDServiceExportReconciler(t *testing.T) {
	// <name>-pool-relay is 56 characters, one more than Karmada imports;
	// <name>-frontend is 54.
	longName := strings.Repeat("a", 45)

	for _, tc := range []struct {
		name             string
		dgdName          string
		frontends        []string
		relay            bool
		relayTerminating bool
		implementation   configv1alpha1.MCSImplementation
		ownedBefore      []string
		terminating      bool
		foreign          []string
		// ownedByAnother are exports another DGD controls.
		ownedByAnother []string
		wantOwned      []string
		wantEvents     int
		// wantReason is the condition's reason once the exports are in place, or
		// empty when the DGD gets no condition. The fake client runs no MCS
		// implementation, so a standard implementation never reports an import.
		wantReason string
	}{
		{
			name:    "a pool exports its PoolRelay and frontend Services",
			dgdName: "qwen", frontends: []string{"Frontend"}, relay: true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{"qwen-pool-relay", "qwen-frontend"},
			wantEvents:     2,
			wantReason:     reasonPoolExported,
		},
		{
			name:    "a standard implementation is waited for",
			dgdName: "qwen", frontends: []string{"Frontend"}, relay: true,
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{"qwen-pool-relay", "qwen-frontend"},
			wantEvents:     2,
			wantReason:     reasonPoolExportPending,
		},
		{
			name:    "each frontend component's Service is exported under its own name",
			dgdName: "qwen", frontends: []string{"Frontend", "Web.v2"}, relay: true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{"qwen-pool-relay", "qwen-frontend", "qwen-web-v2"},
			wantEvents:     3,
			wantReason:     reasonPoolExported,
		},
		{
			name:    "a renamed frontend's export moves to its new Service",
			dgdName: "qwen", frontends: []string{"Web"}, relay: true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			ownedBefore:    []string{"qwen-pool-relay", "qwen-frontend"},
			wantOwned:      []string{"qwen-pool-relay", "qwen-web"},
			wantEvents:     2,
			wantReason:     reasonPoolExported,
		},
		{
			name:    "a pool without a frontend exports its PoolRelay Service and reports it",
			dgdName: "qwen", relay: true,
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{"qwen-pool-relay"},
			wantEvents:     1,
			wantReason:     reasonPoolNoFrontend,
		},
		{
			name:    "a DGD with a dot in its name has no PoolRelay Service to export",
			dgdName: "qwen3-0.6b", frontends: []string{"Frontend"}, relay: true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{"qwen3-0-6b-frontend"},
			wantEvents:     1,
			wantReason:     reasonPoolInvalidServiceName,
		},
		{
			name:    "deleting the DynamoPoolRelay withdraws the exports",
			dgdName: "qwen", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationKarmada,
			ownedBefore:    []string{"qwen-pool-relay", "qwen-frontend"},
			wantEvents:     2,
		},
		{
			name:    "a DynamoPoolRelay being deleted withdraws the exports",
			dgdName: "qwen", frontends: []string{"Frontend"}, relay: true, relayTerminating: true,
			implementation: configv1alpha1.MCSImplementationStandard,
			ownedBefore:    []string{"qwen-pool-relay", "qwen-frontend"},
			wantEvents:     2,
		},
		{
			name:    "an export already being deleted is not deleted again",
			dgdName: "qwen", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationStandard,
			ownedBefore:    []string{"qwen-pool-relay"},
			terminating:    true,
			wantOwned:      []string{"qwen-pool-relay"},
		},
		{
			name:    "an export a user created is neither adopted nor deleted",
			dgdName: "qwen", frontends: []string{"Frontend"}, relay: true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			foreign:        []string{"qwen-frontend"},
			wantOwned:      []string{"qwen-pool-relay"},
			wantEvents:     1,
			wantReason:     reasonPoolExported,
		},
		{
			name:    "a DGD that is not a pool leaves a user's export alone",
			dgdName: "qwen", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationStandard,
			foreign:        []string{"qwen-pool-relay"},
		},
		{
			name:    "a DGD that is not a pool leaves another DGD's export alone",
			dgdName: "qwen", frontends: []string{"Frontend"},
			implementation: configv1alpha1.MCSImplementationStandard,
			ownedByAnother: []string{"qwen-frontend"},
		},
		{
			name:    "without an MCS implementation nothing is exported",
			dgdName: "qwen", frontends: []string{"Frontend"}, relay: true,
			implementation: configv1alpha1.MCSImplementationNone,
		},
		{
			name:    "Karmada skips a Service name it cannot import",
			dgdName: longName, frontends: []string{"Frontend"}, relay: true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{longName + "-frontend"},
			wantEvents:     1,
			wantReason:     reasonPoolServiceNameTooLong,
		},
		{
			name:    "standard MCS exports the same long names",
			dgdName: longName, frontends: []string{"Frontend"}, relay: true,
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{longName + "-pool-relay", longName + "-frontend"},
			wantEvents:     2,
			wantReason:     reasonPoolExportPending,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			ctx := context.Background()

			t.Log("Create the DGD, its DynamoPoolRelay if it is a pool, and the exports that exist before reconciling")
			scheme := newDynamoGraphDeploymentControllerTestScheme(t)
			dgd := newPoolTestDGD(tc.dgdName, tc.frontends...)
			objects := []client.Object{dgd}
			if tc.relay {
				relay := newPoolTestRelay(dgd)
				if tc.relayTerminating {
					relay.Finalizers = []string{"example.com/hold"}
					relay.DeletionTimestamp = &metav1.Time{Time: time.Now()}
				}
				objects = append(objects, relay)
			}
			for _, name := range tc.ownedBefore {
				owned := ownedServiceExport(t, scheme, dgd, name)
				if tc.terminating {
					owned.SetFinalizers([]string{"example.com/hold"})
					owned.SetDeletionTimestamp(&metav1.Time{Time: time.Now()})
				}
				objects = append(objects, owned)
			}
			for _, name := range tc.foreign {
				objects = append(objects, multicluster.RenderServiceExport(dgd.Namespace, name, dgd.Name))
			}
			another := newPoolTestDGD("another", "Frontend")
			for _, name := range tc.ownedByAnother {
				objects = append(objects, ownedServiceExport(t, scheme, another, name))
			}
			fakeClient := newServiceExportTestClient(scheme, interceptor.Funcs{}, objects...)

			t.Log("Reconcile the exports, with the gate the configured implementation resolves to")
			recorder := events.NewFakeRecorder(10)
			reconciler := newServiceExportTestReconciler(fakeClient, recorder, tc.implementation, "")
			_, err := reconciler.Reconcile(ctx, dgd)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(recorder.Events).To(gomega.HaveLen(tc.wantEvents))

			t.Log("Reconcile again: every export is in place, so nothing more is recorded, and the condition settles")
			condition, err := reconciler.Reconcile(ctx, dgd)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(recorder.Events).To(gomega.HaveLen(tc.wantEvents))
			if tc.wantReason == "" {
				g.Expect(condition).To(gomega.BeNil())
			} else {
				g.Expect(condition).NotTo(gomega.BeNil())
				g.Expect(condition.Type).To(gomega.Equal(globalRouterPoolExportedConditionType))
				g.Expect(condition.Reason).To(gomega.Equal(tc.wantReason), condition.Message)
			}

			t.Log("Check which exports the DGD controls, and that other exports are untouched")
			var owned, others []string
			for i, export := range listServiceExports(t, fakeClient, dgd.Namespace) {
				if !metav1.IsControlledBy(&export, dgd) {
					others = append(others, export.GetName())
					continue
				}
				owned = append(owned, export.GetName())
				g.Expect(export.GetLabels()).To(gomega.HaveKeyWithValue(consts.KubeLabelDynamoGraphDeploymentName, dgd.Name), i)
			}
			g.Expect(owned).To(gomega.ConsistOf(tc.wantOwned))
			g.Expect(others).To(gomega.ConsistOf(append(append([]string{}, tc.foreign...), tc.ownedByAnother...)))
		})
	}
}

func TestDGDServiceExportReconcilerReportsTheImport(t *testing.T) {
	for _, tc := range []struct {
		name                string
		implementation      configv1alpha1.MCSImplementation
		relayConditions     []metav1.Condition
		frontendTerminating bool
		wantStatus          metav1.ConditionStatus
		wantReason          string
		wantMessage         string
	}{
		{
			name:           "Karmada reports nothing",
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantStatus:     metav1.ConditionTrue,
			wantReason:     reasonPoolExported,
			wantMessage:    "Exported qwen-pool-relay, qwen-frontend through karmada",
		},
		{
			name:            "a standard implementation imported both",
			implementation:  configv1alpha1.MCSImplementationStandard,
			relayConditions: []metav1.Condition{imported},
			wantStatus:      metav1.ConditionTrue,
			wantReason:      reasonPoolExported,
			wantMessage:     "Exported qwen-pool-relay, qwen-frontend through mcs",
		},
		{
			name:           "a standard implementation has not reported yet",
			implementation: configv1alpha1.MCSImplementationStandard,
			wantStatus:     metav1.ConditionUnknown,
			wantReason:     reasonPoolExportPending,
			wantMessage:    "Waiting for mcs to import qwen-pool-relay, qwen-frontend",
		},
		{
			name:           "the MCS implementation finds no Service to export",
			implementation: configv1alpha1.MCSImplementationStandard,
			relayConditions: []metav1.Condition{
				{Type: "Valid", Status: metav1.ConditionFalse, Reason: "NoService", Message: "Service not found"},
			},
			wantStatus:  metav1.ConditionFalse,
			wantReason:  reasonPoolExportRejected,
			wantMessage: "mcs reports on the export of Service qwen-pool-relay: Valid is False (NoService): Service not found",
		},
		{
			name:           "the import is pending",
			implementation: configv1alpha1.MCSImplementationStandard,
			relayConditions: []metav1.Condition{
				{Type: "Valid", Status: metav1.ConditionTrue}, {Type: "Ready", Status: metav1.ConditionFalse, Reason: "Pending"},
			},
			wantStatus:  metav1.ConditionUnknown,
			wantReason:  reasonPoolExportPending,
			wantMessage: "Waiting for mcs to import qwen-pool-relay",
		},
		{
			name:           "the import failed",
			implementation: configv1alpha1.MCSImplementationStandard,
			relayConditions: []metav1.Condition{
				{Type: "Ready", Status: metav1.ConditionFalse, Reason: "Failed", Message: "refused"},
			},
			wantStatus:  metav1.ConditionFalse,
			wantReason:  reasonPoolExportRejected,
			wantMessage: "mcs reports on the export of Service qwen-pool-relay: Ready is False (Failed): refused",
		},
		{
			name:           "the implementation resolved a conflict",
			implementation: configv1alpha1.MCSImplementationStandard,
			relayConditions: []metav1.Condition{
				imported, {Type: "Conflict", Status: metav1.ConditionTrue, Reason: "PortConflict", Message: "ports differ"},
			},
			wantStatus: metav1.ConditionTrue,
			wantReason: reasonPoolExportConflict,
			wantMessage: "Exported qwen-pool-relay, qwen-frontend through mcs, which resolves a conflict with another " +
				"cluster's export of qwen-pool-relay (Conflict is True (PortConflict): ports differ)",
		},
		{
			name:                "an export waits to be replaced",
			implementation:      configv1alpha1.MCSImplementationStandard,
			relayConditions:     []metav1.Condition{imported},
			frontendTerminating: true,
			wantStatus:          metav1.ConditionUnknown,
			wantReason:          reasonPoolExportPending,
			wantMessage:         "Waiting for mcs to import qwen-frontend",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			t.Log("Create a pool whose exports exist, with what the MCS implementation reported on them")
			scheme := newDynamoGraphDeploymentControllerTestScheme(t)
			dgd := newPoolTestDGD("qwen", "Frontend")
			relayExport := ownedServiceExport(t, scheme, dgd, "qwen-pool-relay")
			frontendExport := ownedServiceExport(t, scheme, dgd, "qwen-frontend")
			if len(tc.relayConditions) > 0 {
				withExportConditions(relayExport, tc.relayConditions...)
				withExportConditions(frontendExport, imported)
			}
			if tc.frontendTerminating {
				frontendExport.SetFinalizers([]string{"example.com/hold"})
				frontendExport.SetDeletionTimestamp(&metav1.Time{Time: time.Now()})
			}
			fakeClient := newServiceExportTestClient(scheme, interceptor.Funcs{},
				dgd, newPoolTestRelay(dgd), relayExport, frontendExport)

			t.Log("Reconcile and read the pool's condition")
			recorder := events.NewFakeRecorder(10)
			reconciler := newServiceExportTestReconciler(fakeClient, recorder, tc.implementation, "")
			condition, err := reconciler.Reconcile(context.Background(), dgd)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(recorder.Events).To(gomega.BeEmpty())
			g.Expect(condition.Status).To(gomega.Equal(tc.wantStatus))
			g.Expect(condition.Reason).To(gomega.Equal(tc.wantReason))
			g.Expect(condition.Message).To(gomega.Equal(tc.wantMessage))
		})
	}
}

func TestDGDServiceExportReconcilerChecksTheClusterID(t *testing.T) {
	for _, tc := range []struct {
		name           string
		implementation configv1alpha1.MCSImplementation
		// clusterSetID is the cluster's ID in its ClusterSet, or empty when unknown.
		clusterSetID   string
		noClusterSetID bool
		wantReason     string
	}{
		{
			name:           "a DynamoPoolRelay with the cluster's ID",
			implementation: configv1alpha1.MCSImplementationStandard,
			clusterSetID:   "east",
			wantReason:     reasonPoolExported,
		},
		{
			name:           "a DynamoPoolRelay with another cluster ID",
			implementation: configv1alpha1.MCSImplementationStandard,
			clusterSetID:   "west",
			wantReason:     reasonPoolClusterIDMismatch,
		},
		{
			name:           "a cluster whose ID in its ClusterSet is unknown",
			implementation: configv1alpha1.MCSImplementationStandard,
			wantReason:     reasonPoolExported,
		},
		{
			name:           "Karmada names member clusters itself",
			implementation: configv1alpha1.MCSImplementationKarmada,
			clusterSetID:   "west",
			wantReason:     reasonPoolExported,
		},
		{
			name:           "without a way to read the ID the check is skipped",
			implementation: configv1alpha1.MCSImplementationStandard,
			noClusterSetID: true,
			wantReason:     reasonPoolExported,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			t.Log("Create a pool of cluster east whose exports are imported")
			scheme := newDynamoGraphDeploymentControllerTestScheme(t)
			dgd := newPoolTestDGD("qwen", "Frontend")
			fakeClient := newServiceExportTestClient(scheme, interceptor.Funcs{},
				dgd, newPoolTestRelay(dgd),
				withExportConditions(ownedServiceExport(t, scheme, dgd, "qwen-pool-relay"), imported),
				withExportConditions(ownedServiceExport(t, scheme, dgd, "qwen-frontend"), imported))

			t.Log("Reconcile in a cluster with the given ID in its ClusterSet, and read the pool's condition")
			reconciler := newServiceExportTestReconciler(fakeClient, events.NewFakeRecorder(10), tc.implementation, tc.clusterSetID)
			if tc.noClusterSetID {
				reconciler.clusterSetID = nil
			}
			condition, err := reconciler.Reconcile(context.Background(), dgd)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(condition.Reason).To(gomega.Equal(tc.wantReason), condition.Message)
			if tc.wantReason == reasonPoolClusterIDMismatch {
				g.Expect(condition.Status).To(gomega.Equal(metav1.ConditionFalse))
				g.Expect(condition.Message).To(gomega.ContainSubstring(`clusterId "east"`))
				g.Expect(condition.Message).To(gomega.ContainSubstring(`the ID "west"`))
			}
		})
	}
}

func TestClusterSetIDs(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Serve the cluster's ID in its ClusterSet, counting reads and failing them on demand")
	property := &unstructured.Unstructured{Object: map[string]any{"spec": map[string]any{"value": "east"}}}
	property.SetGroupVersionKind(multicluster.ClusterPropertyGVK)
	property.SetName(multicluster.ClusterIDProperty)
	reads := 0
	var failure error
	reader := newServiceExportTestClient(newDynamoGraphDeploymentControllerTestScheme(t), interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			reads++
			if failure != nil {
				return failure
			}
			return c.Get(ctx, key, obj, opts...)
		},
	}, property)
	ids := &clusterSetIDs{}
	start := time.Now()

	t.Log("The first call reads the ID, and calls within the TTL reuse it")
	g.Expect(ids.get(ctx, reader, start)).To(gomega.Equal("east"))
	g.Expect(ids.get(ctx, reader, start.Add(clusterSetIDTTL-time.Second))).To(gomega.Equal("east"))
	g.Expect(reads).To(gomega.Equal(1))

	t.Log("After the TTL a failed read skips the check and is not kept")
	failure = errors.New("connection refused")
	g.Expect(ids.get(ctx, reader, start.Add(clusterSetIDTTL))).To(gomega.BeEmpty())
	failure = nil
	g.Expect(ids.get(ctx, reader, start.Add(clusterSetIDTTL))).To(gomega.Equal("east"))
	g.Expect(reads).To(gomega.Equal(3))
}

func TestDGDServiceExportReconcilerToleratesAStaleCache(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("The cache has not yet seen exports that the API server already has")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := newPoolTestDGD("qwen", "Frontend")
	fakeClient := newServiceExportTestClient(scheme, interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			return apierrors.NewAlreadyExists(multicluster.ServiceExportGVK.GroupVersion().WithResource("serviceexports").GroupResource(), obj.GetName())
		},
	}, dgd, newPoolTestRelay(dgd))

	t.Log("Reconcile: AlreadyExists is no error, nothing is announced, and the exports are pending")
	recorder := events.NewFakeRecorder(10)
	reconciler := newServiceExportTestReconciler(fakeClient, recorder, configv1alpha1.MCSImplementationKarmada, "")
	condition, err := reconciler.Reconcile(ctx, dgd)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(recorder.Events).To(gomega.BeEmpty())
	g.Expect(condition.Reason).To(gomega.Equal(reasonPoolExportPending))
}

func TestMapServiceExportToDGDRequests(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create two DGDs in one namespace, one with a dot in its name, and a third elsewhere")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	llama := newPoolTestDGD("llama", "Frontend", "Web")
	dotted := newPoolTestDGD("qwen3-0.6b", "Frontend")
	elsewhere := newPoolTestDGD("llama", "Frontend")
	elsewhere.Namespace = "other"
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: newServiceExportTestClient(scheme, interceptor.Funcs{}, llama, dotted, elsewhere),
	}

	for _, tc := range []struct {
		name   string
		export *unstructured.Unstructured
		want   []ctrl.Request
	}{
		{
			name:   "a PoolRelay Service",
			export: multicluster.RenderServiceExport("dynamo", "llama-pool-relay", "llama"),
			want:   []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(llama)}},
		},
		{
			name:   "a frontend Service under its component's name",
			export: multicluster.RenderServiceExport("dynamo", "llama-web", "llama"),
			want:   []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(llama)}},
		},
		{
			name:   "the frontend Service of a DGD with a dot in its name",
			export: multicluster.RenderServiceExport("dynamo", "qwen3-0-6b-frontend", "qwen3-0.6b"),
			want:   []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(dotted)}},
		},
		{
			name:   "an export the DGD controls for a frontend it no longer has",
			export: ownedServiceExport(t, scheme, llama, "llama-old"),
			want:   []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(llama)}},
		},
		{
			name:   "another Service",
			export: multicluster.RenderServiceExport("dynamo", "llama", "llama"),
			want:   []ctrl.Request{},
		},
	} {
		t.Log("Map an event for " + tc.name + " in namespace dynamo to the DGDs it concerns")
		g.Expect(reconciler.mapServiceExportToDGDRequests(ctx, tc.export)).To(gomega.Equal(tc.want), tc.name)
	}
}

func TestPoolWatchPredicates(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	deleting := &metav1.Time{Time: time.Now()}

	t.Log("A DynamoPoolRelay event requeues the DGD of the same name")
	relay := newPoolTestRelay(newPoolTestDGD("qwen"))
	g.Expect(mapPoolRelayToDGDRequests(context.Background(), relay)).
		To(gomega.Equal([]ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(relay)}}))

	t.Log("A DynamoPoolRelay update counts when its spec changes or its deletion starts")
	respecified := relay.DeepCopy()
	respecified.Generation = 2
	relabeled := relay.DeepCopy()
	relabeled.Labels = map[string]string{"team": "router"}
	withdrawn := relay.DeepCopy()
	withdrawn.DeletionTimestamp = deleting
	g.Expect(poolRelayChanged(event.UpdateEvent{ObjectOld: relay, ObjectNew: respecified})).To(gomega.BeTrue())
	g.Expect(poolRelayChanged(event.UpdateEvent{ObjectOld: relay, ObjectNew: withdrawn})).To(gomega.BeTrue())
	g.Expect(poolRelayChanged(event.UpdateEvent{ObjectOld: relay, ObjectNew: relabeled})).To(gomega.BeFalse())

	t.Log("A ServiceExport update counts when the MCS implementation reports or its deletion starts")
	export := multicluster.RenderServiceExport("dynamo", "qwen-frontend", "qwen")
	reported := withExportConditions(export.DeepCopy(), imported)
	relabeledExport := export.DeepCopy()
	relabeledExport.SetLabels(map[string]string{"team": "router"})
	deletingExport := export.DeepCopy()
	deletingExport.SetDeletionTimestamp(deleting)
	g.Expect(serviceExportStateChanged(event.UpdateEvent{ObjectOld: export, ObjectNew: reported})).To(gomega.BeTrue())
	g.Expect(serviceExportStateChanged(event.UpdateEvent{ObjectOld: export, ObjectNew: deletingExport})).To(gomega.BeTrue())
	g.Expect(serviceExportStateChanged(event.UpdateEvent{ObjectOld: export, ObjectNew: relabeledExport})).To(gomega.BeFalse())
	g.Expect(serviceExportStateChanged(event.UpdateEvent{ObjectOld: &corev1.Service{}, ObjectNew: &corev1.Service{}})).To(gomega.BeFalse())
}

func TestApplyPoolExportCondition(t *testing.T) {
	condition := func(status metav1.ConditionStatus, reason string) *metav1.Condition {
		return &metav1.Condition{
			Type: globalRouterPoolExportedConditionType, Status: status, Reason: reason, Message: reason + " message",
		}
	}
	for _, tc := range []struct {
		name       string
		previous   *metav1.Condition
		condition  *metav1.Condition
		wantEvents []string
	}{
		{
			name:       "a new problem is set and announced",
			condition:  condition(metav1.ConditionFalse, reasonPoolNoFrontend),
			wantEvents: []string{reasonPoolNoFrontend},
		},
		{
			name:      "a problem already reported is not announced again",
			previous:  condition(metav1.ConditionFalse, reasonPoolNoFrontend),
			condition: condition(metav1.ConditionFalse, reasonPoolNoFrontend),
		},
		{
			name:       "another problem is announced",
			previous:   condition(metav1.ConditionFalse, reasonPoolNoFrontend),
			condition:  condition(metav1.ConditionFalse, reasonPoolClusterIDMismatch),
			wantEvents: []string{reasonPoolClusterIDMismatch},
		},
		{
			name:       "a conflict the implementation resolved is announced",
			previous:   condition(metav1.ConditionTrue, reasonPoolExported),
			condition:  condition(metav1.ConditionTrue, reasonPoolExportConflict),
			wantEvents: []string{reasonPoolExportConflict},
		},
		{
			name:      "a failed reconcile records its own event",
			condition: condition(metav1.ConditionFalse, reasonPoolExportFailed),
		},
		{
			name:      "a pending import is not announced",
			condition: condition(metav1.ConditionUnknown, reasonPoolExportPending),
		},
		{
			name:      "an export that works is not announced",
			previous:  condition(metav1.ConditionFalse, reasonPoolNoFrontend),
			condition: condition(metav1.ConditionTrue, reasonPoolExported),
		},
		{
			name:     "a DGD that is no longer a pool loses the condition",
			previous: condition(metav1.ConditionTrue, reasonPoolExported),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			t.Log("Start from the status the DGD has, as the workload program does")
			dgd := newPoolTestDGD("qwen", "Frontend")
			if tc.previous != nil {
				meta.SetStatusCondition(&dgd.Status.Conditions, *tc.previous)
			}
			result := newWorkloadProgramResult(dgd)

			t.Log("Apply the pool's condition")
			applyPoolExportCondition(dgd, &result, tc.condition)
			applied := meta.FindStatusCondition(result.Status.Conditions, globalRouterPoolExportedConditionType)
			if tc.condition == nil {
				g.Expect(applied).To(gomega.BeNil())
			} else {
				g.Expect(applied).NotTo(gomega.BeNil())
				g.Expect(applied.Status).To(gomega.Equal(tc.condition.Status))
				g.Expect(applied.Reason).To(gomega.Equal(tc.condition.Reason))
			}
			reasons := make([]string, 0, len(result.Events))
			for _, recorded := range result.Events {
				g.Expect(recorded.Type).To(gomega.Equal(corev1.EventTypeWarning))
				reasons = append(reasons, recorded.Reason)
			}
			g.Expect(reasons).To(gomega.ConsistOf(tc.wantEvents))
		})
	}
}

func TestReconcilePoolExportsRetriesWithoutFailingTheDGD(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Make a DGD a pool, and refuse every ServiceExport create, as an admission policy would")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := newPoolTestDGD("llama", "Frontend")
	relay := newPoolTestRelay(dgd)
	refused := interceptor.Funcs{
		Create: func(context.Context, client.WithWatch, client.Object, ...client.CreateOption) error {
			return apierrors.NewForbidden(
				schema.GroupResource{Group: multicluster.ServiceExportGVK.Group, Resource: "serviceexports"},
				"llama-pool-relay", errors.New("denied by policy"))
		},
	}
	recorder := events.NewFakeRecorder(10)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:   newServiceExportTestClient(scheme, refused, dgd, relay),
		Recorder: recorder,
		Config: &configv1alpha1.OperatorConfiguration{
			MultiCluster: configv1alpha1.MultiClusterConfiguration{
				MCSImplementation: configv1alpha1.MCSImplementationKarmada,
			},
		},
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{MultiClusterExport: true}},
	}

	t.Log("The failure asks for a retry, is recorded as a warning and in the condition, and returns no error that would fail the DGD")
	result := reconciler.reconcilePoolExports(ctx, dgd)
	g.Expect(result.retry).To(gomega.BeTrue())
	g.Expect(result.condition.Status).To(gomega.Equal(metav1.ConditionFalse))
	g.Expect(result.condition.Reason).To(gomega.Equal(reasonPoolExportFailed))
	g.Expect(recorder.Events).To(gomega.HaveLen(1))
	g.Expect(<-recorder.Events).To(gomega.HavePrefix(corev1.EventTypeWarning + " PoolExportFailed"))

	t.Log("A failure to read the DynamoPoolRelay leaves the condition as it was, since the DGD may not be a pool")
	meta.SetStatusCondition(&dgd.Status.Conditions, *newPoolCondition(dgd, metav1.ConditionTrue, reasonPoolExported, "exported"))
	reconciler.Client = newServiceExportTestClient(scheme, interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*v1alpha1.DynamoPoolRelay); ok {
				return errors.New("cache not synced")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	}, dgd, relay)
	result = reconciler.reconcilePoolExports(ctx, dgd)
	g.Expect(result.retry).To(gomega.BeTrue())
	g.Expect(result.condition).NotTo(gomega.BeNil())
	g.Expect(result.condition.Reason).To(gomega.Equal(reasonPoolExported))
	g.Expect(<-recorder.Events).To(gomega.HavePrefix(corev1.EventTypeWarning + " PoolExportFailed"))

	t.Log("A failed withdrawal from a DGD that is no longer a pool keeps the condition until the exports are gone")
	former := newPoolTestDGD("qwen", "Frontend")
	meta.SetStatusCondition(&former.Status.Conditions, *newPoolCondition(former, metav1.ConditionTrue, reasonPoolExported, "exported"))
	reconciler.Client = newServiceExportTestClient(scheme, interceptor.Funcs{
		Delete: func(context.Context, client.WithWatch, client.Object, ...client.DeleteOption) error {
			return errors.New("denied by policy")
		},
	}, former, ownedServiceExport(t, scheme, former, "qwen-frontend"))
	result = reconciler.reconcilePoolExports(ctx, former)
	g.Expect(result.retry).To(gomega.BeTrue())
	g.Expect(result.condition.Reason).To(gomega.Equal(reasonPoolExported))
	g.Expect(<-recorder.Events).To(gomega.HavePrefix(corev1.EventTypeWarning + " PoolExportFailed"))

	t.Log("Exports that succeed ask for no retry")
	reconciler.Client = newServiceExportTestClient(scheme, interceptor.Funcs{}, dgd, relay)
	result = reconciler.reconcilePoolExports(ctx, dgd)
	g.Expect(result.retry).To(gomega.BeFalse())
	g.Expect(result.condition.Reason).To(gomega.Equal(reasonPoolExportPending))
}

// newPoolTestDGDReconciler returns a DGD reconciler over objects, with standard
// MCS and the gates given.
func newPoolTestDGDReconciler(
	t *testing.T,
	gates features.Gates,
	objects ...client.Object,
) (*DynamoGraphDeploymentReconciler, *events.FakeRecorder) {
	gates.MultiClusterExport = true
	recorder := events.NewFakeRecorder(10)
	return &DynamoGraphDeploymentReconciler{
		Client:   newServiceExportTestClient(newDynamoGraphDeploymentControllerTestScheme(t), interceptor.Funcs{}, objects...),
		Recorder: recorder,
		Config: &configv1alpha1.OperatorConfiguration{
			MultiCluster: configv1alpha1.MultiClusterConfiguration{
				MCSImplementation: configv1alpha1.MCSImplementationStandard,
			},
		},
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: gates},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) { return nil, nil },
		},
	}, recorder
}

// drainEvents returns the events recorder holds.
func drainEvents(recorder *events.FakeRecorder) []string {
	var recorded []string
	for len(recorder.Events) > 0 {
		recorded = append(recorded, <-recorder.Events)
	}
	return recorded
}

func TestDynamoGraphDeploymentReconcilePersistsThePoolCondition(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := t.Context()

	t.Log("Create a finalized DGD on the component provider, whose program rejects its only component, and its DynamoPoolRelay")
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: "llama", Namespace: "dynamo", Generation: 4, UID: uuid.NewUUID(),
		Annotations: map[string]string{consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent},
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "serving", ComponentType: v1beta1.ComponentTypeLPX},
		}},
	}
	commoncontroller.AddFinalizer(dgd)
	relay := newPoolTestRelay(dgd)
	reconciler, recorder := newPoolTestDGDReconciler(t, features.Gates{LPX: true}, dgd, relay)
	reconcileAndRead := func() (*v1beta1.DynamoGraphDeployment, []string) {
		_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
		g.Expect(errors.Is(err, reconcile.TerminalError(nil))).To(gomega.BeTrue(), "%v", err)
		stored := &v1beta1.DynamoGraphDeployment{}
		g.Expect(reconciler.Get(ctx, client.ObjectKeyFromObject(dgd), stored)).To(gomega.Succeed())
		return stored, drainEvents(recorder)
	}
	noFrontend := gomega.HavePrefix(corev1.EventTypeWarning + " " + reasonPoolNoFrontend)

	t.Log("Reconcile: the program's status is persisted with the pool's condition, and the problem is announced")
	stored, recorded := reconcileAndRead()
	ready := meta.FindStatusCondition(stored.Status.Conditions, "Ready")
	g.Expect(ready).NotTo(gomega.BeNil())
	g.Expect(ready.Reason).To(gomega.Equal("UnsupportedComponent"))
	pool := meta.FindStatusCondition(stored.Status.Conditions, globalRouterPoolExportedConditionType)
	g.Expect(pool).NotTo(gomega.BeNil())
	g.Expect(pool.Status).To(gomega.Equal(metav1.ConditionFalse))
	g.Expect(pool.Reason).To(gomega.Equal(reasonPoolNoFrontend))
	g.Expect(pool.ObservedGeneration).To(gomega.Equal(dgd.Generation))
	g.Expect(recorded).To(gomega.ContainElement(noFrontend))

	t.Log("Reconcile again: the same problem is not announced twice")
	_, recorded = reconcileAndRead()
	g.Expect(recorded).NotTo(gomega.ContainElement(noFrontend))

	t.Log("Delete the DynamoPoolRelay: the export is withdrawn and the condition removed")
	g.Expect(reconciler.Delete(ctx, relay)).To(gomega.Succeed())
	stored, _ = reconcileAndRead()
	g.Expect(meta.FindStatusCondition(stored.Status.Conditions, globalRouterPoolExportedConditionType)).To(gomega.BeNil())
	g.Expect(listServiceExports(t, reconciler.Client, dgd.Namespace)).To(gomega.BeEmpty())
}

func TestDynamoGraphDeploymentReconcileKeepsThePoolConditionOnEarlyFailures(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := t.Context()

	t.Log("Store a DGD with an LPX component while LPX is disabled, and its DynamoPoolRelay")
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: "mixed", Namespace: "dynamo", Generation: 4, UID: uuid.NewUUID(),
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeLPX},
		}},
	}
	relay := newPoolTestRelay(dgd)
	reconciler, _ := newPoolTestDGDReconciler(t, features.Gates{}, dgd, relay)

	t.Log("Reconcile: the early failure's status carries the pool's condition")
	_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
	g.Expect(err).NotTo(gomega.HaveOccurred())
	stored := &v1beta1.DynamoGraphDeployment{}
	g.Expect(reconciler.Get(ctx, client.ObjectKeyFromObject(dgd), stored)).To(gomega.Succeed())
	g.Expect(meta.FindStatusCondition(stored.Status.Conditions, "Ready").Reason).To(gomega.Equal("lpx_disabled"))
	g.Expect(meta.FindStatusCondition(stored.Status.Conditions, globalRouterPoolExportedConditionType).Reason).
		To(gomega.Equal(reasonPoolNoFrontend))

	t.Log("Delete the DynamoPoolRelay: the next early failure removes the condition")
	g.Expect(reconciler.Delete(ctx, relay)).To(gomega.Succeed())
	_, err = reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(reconciler.Get(ctx, client.ObjectKeyFromObject(dgd), stored)).To(gomega.Succeed())
	g.Expect(meta.FindStatusCondition(stored.Status.Conditions, globalRouterPoolExportedConditionType)).To(gomega.BeNil())
}

func TestRetryPoolExports(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	for _, tc := range []struct {
		name   string
		result ctrl.Result
		err    error
		want   ctrl.Result
	}{
		{name: "a reconcile that would not run again retries the exports", want: ctrl.Result{RequeueAfter: poolExportRetry}},
		{name: "a later retry comes sooner", result: ctrl.Result{RequeueAfter: time.Hour}, want: ctrl.Result{RequeueAfter: poolExportRetry}},
		{name: "a sooner retry stays", result: ctrl.Result{RequeueAfter: time.Second}, want: ctrl.Result{RequeueAfter: time.Second}},
		{name: "an error is retried anyway", err: errors.New("failed"), want: ctrl.Result{}},
	} {
		t.Log(tc.name)
		g.Expect(retryPoolExports(tc.result, tc.err)).To(gomega.Equal(tc.want), tc.name)
	}
}
