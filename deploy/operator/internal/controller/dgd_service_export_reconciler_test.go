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
	"slices"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// newServiceExportTestClient returns a fake client that also serves the
// ServiceExport kind, which the operator's scheme does not register.
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
	return fake.NewClientBuilder().
		WithScheme(scheme).
		WithRESTMapper(mapper).
		WithInterceptorFuncs(funcs).
		WithObjects(objects...).
		Build()
}

func TestDGDServiceExportReconciler(t *testing.T) {
	const namespace = "dynamo"
	// <name>-pool-relay is 56 characters, one more than Karmada imports;
	// <name>-frontend is 54.
	longName := strings.Repeat("a", 45)

	for _, tc := range []struct {
		name           string
		dgdName        string
		pool           bool
		implementation configv1alpha1.MCSImplementation
		ownedBefore    []string
		terminating    bool
		foreign        []string
		wantOwned      []string
		wantEvents     int
		// Warnings that every reconcile repeats.
		repeatedEvents int
	}{
		{
			name:           "a pool exports its PoolRelay and frontend Services",
			dgdName:        "qwen",
			pool:           true,
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{"qwen-pool-relay", "qwen-frontend"},
			wantEvents:     2,
		},
		{
			name:           "exports use the Service names, with dots replaced",
			dgdName:        "qwen3-0.6b",
			pool:           true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{"qwen3-0-6b-pool-relay", "qwen3-0-6b-frontend"},
			wantEvents:     2,
		},
		{
			name:           "removing the pool mark withdraws the exports",
			dgdName:        "qwen",
			implementation: configv1alpha1.MCSImplementationKarmada,
			ownedBefore:    []string{"qwen-pool-relay", "qwen-frontend"},
			wantEvents:     2,
		},
		{
			name:           "an export already being deleted is not deleted again",
			dgdName:        "qwen",
			implementation: configv1alpha1.MCSImplementationStandard,
			ownedBefore:    []string{"qwen-pool-relay"},
			terminating:    true,
			wantOwned:      []string{"qwen-pool-relay"},
		},
		{
			name:           "an export a user created is neither adopted nor deleted",
			dgdName:        "qwen",
			pool:           true,
			implementation: configv1alpha1.MCSImplementationStandard,
			foreign:        []string{"qwen-frontend"},
			wantOwned:      []string{"qwen-pool-relay"},
			wantEvents:     1,
		},
		{
			name:           "a DGD that is not a pool leaves a user's export alone",
			dgdName:        "qwen",
			implementation: configv1alpha1.MCSImplementationStandard,
			foreign:        []string{"qwen-pool-relay"},
		},
		{
			name:           "without an MCS implementation nothing is exported",
			dgdName:        "qwen",
			pool:           true,
			implementation: configv1alpha1.MCSImplementationNone,
		},
		{
			name:           "Karmada skips a Service name it cannot import",
			dgdName:        longName,
			pool:           true,
			implementation: configv1alpha1.MCSImplementationKarmada,
			wantOwned:      []string{longName + "-frontend"},
			wantEvents:     2,
			repeatedEvents: 1,
		},
		{
			name:           "standard MCS exports the same long names",
			dgdName:        longName,
			pool:           true,
			implementation: configv1alpha1.MCSImplementationStandard,
			wantOwned:      []string{longName + "-pool-relay", longName + "-frontend"},
			wantEvents:     2,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			ctx := context.Background()

			t.Log("Create the DGD and the exports that exist before reconciling")
			scheme := newDynamoGraphDeploymentControllerTestScheme(t)
			dgd := &v1beta1.DynamoGraphDeployment{
				Name: tc.dgdName, Namespace: namespace, UID: uuid.NewUUID(),
			}
			if tc.pool {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGlobalRouterPool: "true"}
			}
			objects := []client.Object{dgd}
			for _, name := range tc.ownedBefore {
				owned := multicluster.RenderServiceExport(namespace, name, dgd.Name)
				g.Expect(controllerutil.SetControllerReference(dgd, owned, scheme)).To(gomega.Succeed())
				if tc.terminating {
					owned.SetFinalizers([]string{"example.com/hold"})
					owned.SetDeletionTimestamp(&metav1.Time{Time: metav1.Now().Time})
				}
				objects = append(objects, owned)
			}
			for _, name := range tc.foreign {
				objects = append(objects, multicluster.RenderServiceExport(namespace, name, dgd.Name))
			}
			fakeClient := newServiceExportTestClient(scheme, interceptor.Funcs{}, objects...)

			t.Log("Reconcile the exports, with the gate the configured implementation resolves to")
			config := &configv1alpha1.OperatorConfiguration{
				MultiCluster: configv1alpha1.MultiClusterConfiguration{MCSImplementation: tc.implementation},
			}
			gates := features.Gates{
				MultiClusterExport: tc.implementation != configv1alpha1.MCSImplementationNone,
			}
			recorder := events.NewFakeRecorder(10)
			reconciler := newDGDServiceExportReconciler(newDGDResourceSyncer(fakeClient, recorder), config, gates)
			g.Expect(reconciler.Reconcile(ctx, dgd)).To(gomega.Succeed())
			g.Expect(recorder.Events).To(gomega.HaveLen(tc.wantEvents))

			t.Log("Reconcile again: every export is in place, so only warnings repeat")
			g.Expect(reconciler.Reconcile(ctx, dgd)).To(gomega.Succeed())
			g.Expect(recorder.Events).To(gomega.HaveLen(tc.wantEvents + tc.repeatedEvents))

			t.Log("Check which exports the DGD controls, and that a user's export is untouched")
			for _, name := range multicluster.ServiceNames(dgd.Name) {
				export := multicluster.ServiceExportMetadata()
				err := fakeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, export)
				switch {
				case slices.Contains(tc.wantOwned, name):
					g.Expect(err).NotTo(gomega.HaveOccurred(), name)
					g.Expect(metav1.IsControlledBy(export, dgd)).To(gomega.BeTrue(), name)
					g.Expect(export.Labels).To(gomega.HaveKeyWithValue(consts.KubeLabelDynamoGraphDeploymentName, dgd.Name))
				case slices.Contains(tc.foreign, name):
					g.Expect(err).NotTo(gomega.HaveOccurred(), name)
					g.Expect(metav1.IsControlledBy(export, dgd)).To(gomega.BeFalse(), name)
				default:
					g.Expect(apierrors.IsNotFound(err)).To(gomega.BeTrue(), "%s: %v", name, err)
				}
			}
		})
	}
}

func TestDGDServiceExportReconcilerToleratesAStaleCache(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("The cache has not yet seen exports that the API server already has")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: "qwen", Namespace: "dynamo", UID: uuid.NewUUID(),
		Annotations: map[string]string{consts.KubeAnnotationGlobalRouterPool: "true"},
	}
	fakeClient := newServiceExportTestClient(scheme, interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			return apierrors.NewAlreadyExists(multicluster.ServiceExportGVK.GroupVersion().WithResource("serviceexports").GroupResource(), obj.GetName())
		},
	}, dgd)

	t.Log("Reconcile: AlreadyExists is no error, and nothing is announced")
	config := &configv1alpha1.OperatorConfiguration{
		MultiCluster: configv1alpha1.MultiClusterConfiguration{MCSImplementation: configv1alpha1.MCSImplementationStandard},
	}
	recorder := events.NewFakeRecorder(10)
	reconciler := newDGDServiceExportReconciler(
		newDGDResourceSyncer(fakeClient, recorder),
		config,
		features.Gates{MultiClusterExport: true},
	)
	g.Expect(reconciler.Reconcile(ctx, dgd)).To(gomega.Succeed())
	g.Expect(recorder.Events).To(gomega.BeEmpty())
}

func TestMapServiceExportToDGDRequests(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Create two DGDs in one namespace, one with a dot in its name, and a third elsewhere")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dotted := &v1beta1.DynamoGraphDeployment{Name: "qwen3-0.6b", Namespace: "dynamo"}
	llama := &v1beta1.DynamoGraphDeployment{Name: "llama", Namespace: "dynamo"}
	elsewhere := &v1beta1.DynamoGraphDeployment{Name: "llama", Namespace: "other"}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: newServiceExportTestClient(scheme, interceptor.Funcs{}, dotted, llama, elsewhere),
	}

	for _, tc := range []struct {
		export string
		want   []ctrl.Request
	}{
		{export: "qwen3-0-6b-pool-relay", want: []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(dotted)}}},
		{export: "llama-frontend", want: []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(llama)}}},
		{export: "llama", want: []ctrl.Request{}},
	} {
		t.Log("Map an event for export " + tc.export + " in namespace dynamo to the DGDs that export it")
		export := multicluster.ServiceExportMetadata()
		export.Namespace = "dynamo"
		export.Name = tc.export
		g.Expect(reconciler.mapServiceExportToDGDRequests(ctx, export)).To(gomega.Equal(tc.want), tc.export)
	}
}

func TestReconcilePoolExportsRetriesWithoutFailingTheDGD(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	t.Log("Mark a DGD as a pool, and refuse every ServiceExport create, as an admission policy would")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{Name: "llama", Namespace: "dynamo", UID: uuid.NewUUID()}
	dgd.Annotations = map[string]string{consts.KubeAnnotationGlobalRouterPool: "true"}
	refused := interceptor.Funcs{
		Create: func(context.Context, client.WithWatch, client.Object, ...client.CreateOption) error {
			return apierrors.NewForbidden(
				schema.GroupResource{Group: multicluster.ServiceExportGVK.Group, Resource: "serviceexports"},
				"llama-pool-relay", errors.New("denied by policy"))
		},
	}
	recorder := events.NewFakeRecorder(10)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:   newServiceExportTestClient(scheme, refused, dgd),
		Recorder: recorder,
		Config: &configv1alpha1.OperatorConfiguration{
			MultiCluster: configv1alpha1.MultiClusterConfiguration{
				MCSImplementation: configv1alpha1.MCSImplementationStandard,
			},
		},
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{MultiClusterExport: true}},
	}

	t.Log("The failure asks for a retry and is recorded as a warning, without an error that would fail the DGD")
	g.Expect(reconciler.reconcilePoolExports(ctx, dgd)).To(gomega.BeTrue())
	g.Expect(recorder.Events).To(gomega.HaveLen(1))
	g.Expect(<-recorder.Events).To(gomega.HavePrefix(corev1.EventTypeWarning + " PoolExportFailed"))

	t.Log("Exports that succeed ask for no retry")
	reconciler.Client = newServiceExportTestClient(scheme, interceptor.Funcs{}, dgd)
	g.Expect(reconciler.reconcilePoolExports(ctx, dgd)).To(gomega.BeFalse())
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
