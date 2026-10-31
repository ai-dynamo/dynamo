/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package multicluster

import (
	"context"
	"errors"
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestMaxServiceNameLength(t *testing.T) {
	for _, tc := range []struct {
		implementation configv1alpha1.MCSImplementation
		want           int
	}{
		{implementation: configv1alpha1.MCSImplementationStandard, want: 63},
		{implementation: configv1alpha1.MCSImplementationKarmada, want: 55},
	} {
		t.Run(string(tc.implementation), func(t *testing.T) {
			t.Log("Look up the longest Service name the implementation exports with its cluster")
			if got := MaxServiceNameLength(tc.implementation); got != tc.want {
				t.Errorf("MaxServiceNameLength(%q) = %d, want %d", tc.implementation, got, tc.want)
			}
		})
	}
}

func TestServiceNames(t *testing.T) {
	component := func(name string, componentType v1beta1.ComponentType) v1beta1.DynamoComponentDeploymentSharedSpec {
		return v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: name, ComponentType: componentType}
	}
	for _, tc := range []struct {
		name          string
		dgdName       string
		components    []v1beta1.DynamoComponentDeploymentSharedSpec
		wantFrontends []string
	}{
		{
			name:    "a frontend named Frontend gets the Service the Global Discovery Service expects by default",
			dgdName: "qwen",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				component("Frontend", v1beta1.ComponentTypeFrontend),
				component("VllmDecodeWorker", v1beta1.ComponentTypeDecode),
			},
			wantFrontends: []string{"qwen-frontend"},
		},
		{
			name:    "every frontend keeps its component's name, lowercased, with dots replaced",
			dgdName: "qwen3-0.6b",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				component("Web.v2", v1beta1.ComponentTypeFrontend),
				component("Frontend", v1beta1.ComponentTypeFrontend),
			},
			wantFrontends: []string{"qwen3-0-6b-web-v2", "qwen3-0-6b-frontend"},
		},
		{
			name:    "frontends whose Services share a name are listed once",
			dgdName: "qwen",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				component("Frontend", v1beta1.ComponentTypeFrontend),
				component("frontend", v1beta1.ComponentTypeFrontend),
			},
			wantFrontends: []string{"qwen-frontend"},
		},
		{
			name:       "a DGD without a frontend has only its PoolRelay Service",
			dgdName:    "qwen",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{component("Worker", v1beta1.ComponentTypeWorker)},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dgd := &v1beta1.DynamoGraphDeployment{Name: tc.dgdName, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: tc.components}}

			t.Log("Name the frontend Services as the operator names its own")
			if got := FrontendServiceNames(dgd); !slices.Equal(got, tc.wantFrontends) {
				t.Errorf("FrontendServiceNames = %v, want %v", got, tc.wantFrontends)
			}

			t.Log("The pool's Services are its PoolRelay Service, then its frontend Services")
			want := append([]string{tc.dgdName + "-pool-relay"}, tc.wantFrontends...)
			if got := ServiceNames(dgd); !slices.Equal(got, want) {
				t.Errorf("ServiceNames = %v, want %v", got, want)
			}
		})
	}
}

func TestRenderServiceExport(t *testing.T) {
	t.Log("Render the export of pool qwen's PoolRelay Service at the version the gate resolved")
	export := RenderServiceExport("v1alpha1", "dynamo", "qwen-pool-relay", "qwen")
	if export.GetAPIVersion() != "multicluster.x-k8s.io/v1alpha1" || export.GetKind() != "ServiceExport" {
		t.Errorf("export type = %s %s, want multicluster.x-k8s.io/v1alpha1 ServiceExport", export.GetAPIVersion(), export.GetKind())
	}
	if export.GetNamespace() != "dynamo" || export.GetName() != "qwen-pool-relay" {
		t.Errorf("export = %s/%s, want dynamo/qwen-pool-relay", export.GetNamespace(), export.GetName())
	}
	if got := export.GetLabels()[commonconsts.KubeLabelDynamoGraphDeploymentName]; got != "qwen" {
		t.Errorf("DGD label = %q, want qwen", got)
	}
}

// exportWithConditions returns a ServiceExport whose status holds conditions,
// each given as type, status, reason and message.
func exportWithConditions(conditions ...[4]string) *unstructured.Unstructured {
	export := RenderServiceExport("v1alpha1", "dynamo", "qwen-pool-relay", "qwen")
	items := make([]any, 0, len(conditions))
	for _, condition := range conditions {
		items = append(items, map[string]any{
			"type": condition[0], "status": condition[1], "reason": condition[2], "message": condition[3],
		})
	}
	if len(items) > 0 {
		export.Object["status"] = map[string]any{"conditions": items}
	}
	return export
}

func TestReadExportState(t *testing.T) {
	const (
		standard = configv1alpha1.MCSImplementationStandard
		karmada  = configv1alpha1.MCSImplementationKarmada
	)
	for _, tc := range []struct {
		name           string
		implementation configv1alpha1.MCSImplementation
		export         *unstructured.Unstructured
		want           ExportState
		wantDetail     string
	}{
		{name: "Karmada writes no status", implementation: karmada, export: exportWithConditions(), want: ExportAccepted},
		{
			name:           "a standard implementation has not reported yet",
			implementation: standard,
			export:         exportWithConditions(),
			want:           ExportPending,
			wantDetail:     "no conditions yet",
		},
		{
			name:           "valid, as an implementation without Ready reports it",
			implementation: standard,
			export:         exportWithConditions([4]string{"Valid", "True", "Valid", ""}),
			want:           ExportAccepted,
		},
		{
			name:           "valid and imported",
			implementation: standard,
			export: exportWithConditions(
				[4]string{"Valid", "True", "Valid", ""},
				[4]string{"Ready", "True", "Exported", ""},
				[4]string{"Conflict", "False", "NoConflicts", ""}),
			want: ExportAccepted,
		},
		{
			name:           "a missing Service is invalid",
			implementation: standard,
			export:         exportWithConditions([4]string{"Valid", "False", "NoService", "Service qwen-pool-relay not found"}),
			want:           ExportRejected,
			wantDetail:     "Valid is False (NoService): Service qwen-pool-relay not found",
		},
		{
			name:           "an export the registry refused",
			implementation: standard,
			export:         exportWithConditions([4]string{"Ready", "False", "Failed", ""}),
			want:           ExportRejected,
			wantDetail:     "Ready is False (Failed)",
		},
		{
			name:           "an import that has not finished, reported as Unknown",
			implementation: standard,
			export:         exportWithConditions([4]string{"Valid", "True", "", ""}, [4]string{"Ready", "Unknown", "Pending", ""}),
			want:           ExportPending,
			wantDetail:     "Ready is Unknown (Pending)",
		},
		{
			name:           "an import that has not finished, reported as False",
			implementation: standard,
			export:         exportWithConditions([4]string{"Valid", "True", "", ""}, [4]string{"Ready", "False", "Pending", ""}),
			want:           ExportPending,
			wantDetail:     "Ready is False (Pending)",
		},
		{
			name:           "a conflict that the implementation resolved",
			implementation: standard,
			export: exportWithConditions(
				[4]string{"Valid", "True", "", ""},
				[4]string{"Ready", "True", "Exported", ""},
				[4]string{"Conflict", "True", "PortConflict", "ports differ"}),
			want:       ExportConflicting,
			wantDetail: "Conflict is True (PortConflict): ports differ",
		},
		{
			name:           "a conflict while the import is not finished",
			implementation: standard,
			export: exportWithConditions(
				[4]string{"Ready", "Unknown", "Pending", ""},
				[4]string{"Conflict", "True", "PortConflict", ""}),
			want:       ExportPending,
			wantDetail: "Ready is Unknown (Pending)",
		},
		{
			name:           "a status that cannot be read counts as no report",
			implementation: standard,
			export: func() *unstructured.Unstructured {
				export := exportWithConditions()
				export.Object["status"] = map[string]any{"conditions": "not a list"}
				return export
			}(),
			want:       ExportPending,
			wantDetail: "no conditions yet",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Read what the MCS implementation reports on the export")
			got, detail := ReadExportState(tc.implementation, tc.export)
			if got != tc.want || detail != tc.wantDetail {
				t.Errorf("ReadExportState = %v, %q, want %v, %q", got, detail, tc.want, tc.wantDetail)
			}
		})
	}
}

func TestCheckServiceName(t *testing.T) {
	for _, tc := range []struct {
		name           string
		implementation configv1alpha1.MCSImplementation
		service        string
		want           NameProblem
	}{
		{name: "a Service name", implementation: configv1alpha1.MCSImplementationKarmada, service: "qwen-pool-relay", want: NameExportable},
		{name: "a dot", implementation: configv1alpha1.MCSImplementationStandard, service: "qwen3-0.6b-pool-relay", want: NameInvalid},
		{name: "a leading digit", implementation: configv1alpha1.MCSImplementationStandard, service: "3b-frontend", want: NameInvalid},
		{name: "longer than Karmada imports", implementation: configv1alpha1.MCSImplementationKarmada, service: strings.Repeat("a", 56), want: NameTooLong},
		{name: "as long as a Service name gets", implementation: configv1alpha1.MCSImplementationStandard, service: strings.Repeat("a", 63), want: NameExportable},
		{name: "longer than any Service name", implementation: configv1alpha1.MCSImplementationStandard, service: strings.Repeat("a", 64), want: NameInvalid},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Check whether the implementation can export a Service of the name")
			got, detail := CheckServiceName(tc.implementation, tc.service)
			if got != tc.want || (got == NameExportable) != (detail == "") {
				t.Errorf("CheckServiceName(%q) = %v, %q, want %v", tc.service, got, detail, tc.want)
			}
		})
	}
}

func TestListExports(t *testing.T) {
	t.Log("Store an export controlled by a DGD that is being deleted, and one a user created that Karmada reports nothing on")
	owned := exportWithConditions([4]string{"Valid", "False", "NoService", ""})
	owned.SetUID("export-1")
	owned.SetOwnerReferences([]metav1.OwnerReference{
		{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: "qwen", UID: "dgd-1", Controller: new(true)},
	})
	owned.SetFinalizers([]string{"example.com/hold"})
	owned.SetDeletionTimestamp(&metav1.Time{Time: time.Now()})
	foreign := RenderServiceExport("v1alpha1", "dynamo", "qwen-frontend", "qwen")
	foreign.SetUID("export-2")
	elsewhere := RenderServiceExport("v1alpha1", "other", "qwen-frontend", "qwen")
	mapper := meta.NewDefaultRESTMapper(nil)
	mapper.Add(ServiceExportGroupKind.WithVersion("v1alpha1"), meta.RESTScopeNamespace)
	reader := fake.NewClientBuilder().
		WithScheme(runtime.NewScheme()).
		WithRESTMapper(mapper).
		WithObjects(owned, foreign, elsewhere).
		Build()

	t.Log("List the exports in namespace dynamo, as Karmada sees them")
	got, err := ListExports(context.Background(), reader, "v1alpha1", "dynamo", configv1alpha1.MCSImplementationKarmada)
	if err != nil {
		t.Fatalf("ListExports: %v", err)
	}
	want := map[string]Export{
		"qwen-pool-relay": {
			Name: "qwen-pool-relay", UID: "export-1", ControllerUID: "dgd-1", Deleting: true,
			State: ExportRejected, Detail: "Valid is False (NoService)",
		},
		"qwen-frontend": {Name: "qwen-frontend", UID: "export-2", State: ExportAccepted},
	}
	if !maps.Equal(got, want) {
		t.Errorf("ListExports = %+v, want %+v", got, want)
	}
}

func TestReadClusterID(t *testing.T) {
	property := &unstructured.Unstructured{Object: map[string]any{"spec": map[string]any{"value": "west"}}}
	property.SetGroupVersionKind(ClusterPropertyGVK)
	property.SetName(ClusterIDProperty)
	otherProperty := property.DeepCopy()
	otherProperty.SetName("clusterset.k8s.io")
	failure := errors.New("connection refused")

	for _, tc := range []struct {
		name      string
		aboutAPI  bool
		objects   []client.Object
		getError  error
		want      string
		wantError bool
	}{
		{name: "the cluster's ID in its ClusterSet", aboutAPI: true, objects: []client.Object{property, otherProperty}, want: "west"},
		{name: "no cluster ID property", aboutAPI: true, objects: []client.Object{otherProperty}},
		{name: "no About API"},
		{
			name: "no permission, as in a namespace-restricted install", aboutAPI: true,
			getError: apierrors.NewForbidden(schema.GroupResource{Group: "about.k8s.io", Resource: "clusterproperties"}, ClusterIDProperty, failure),
		},
		{name: "another failure", aboutAPI: true, getError: failure, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Serve the About API, or not, and fail reads as the case needs")
			mapper := meta.NewDefaultRESTMapper(nil)
			if tc.aboutAPI {
				mapper.Add(ClusterPropertyGVK, meta.RESTScopeRoot)
			}
			funcs := interceptor.Funcs{}
			if tc.getError != nil {
				funcs.Get = func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
					return tc.getError
				}
			}
			reader := fake.NewClientBuilder().
				WithScheme(runtime.NewScheme()).
				WithRESTMapper(mapper).
				WithObjects(tc.objects...).
				WithInterceptorFuncs(funcs).
				Build()

			t.Log("Read the cluster's ID in its ClusterSet")
			got, err := ReadClusterID(context.Background(), reader)
			if (err != nil) != tc.wantError || got != tc.want {
				t.Errorf("ReadClusterID = %q, %v, want %q with error %v", got, err, tc.want, tc.wantError)
			}
			if tc.wantError && !strings.Contains(err.Error(), ClusterIDProperty) {
				t.Errorf("error %q does not name the property", err)
			}
		})
	}
}
