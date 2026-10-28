/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const defaultingAdmissionOperatorVersion = "1.6.0"

var defaultingAdmissionHardware = &defaultingHardwareDiscoverer{}

var defaultingAdmissionEnv = operatorenv.New(operatorenv.Options{
	Admission: operatorenv.AdmissionWebhooks{Mutating: true},
	SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
		return webhooksetup.Setup(mgr, webhooksetup.Options{
			Config:            opts.OperatorConfig,
			RuntimeConfig:     opts.RuntimeConfig,
			OperatorVersion:   opts.OperatorVersion,
			OperatorPrincipal: opts.OperatorPrincipal,
			Gate:              features.Gates{Grove: true, GPUDiscovery: true},
			DGDRV1beta2: webhookdefaulting.DGDRV1beta2Options{
				Hardware: defaultingAdmissionHardware,
			},
		})
	},
	OperatorVersion: defaultingAdmissionOperatorVersion,
})

type defaultingHardwareDiscoverer struct {
	mu    sync.Mutex
	found gpu.GPUInfo
	err   error
}

func (f *defaultingHardwareDiscoverer) set(found gpu.GPUInfo, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.found, f.err = found, err
}

func (f *defaultingHardwareDiscoverer) Discover(context.Context, nvidiacomv1beta1.GPUSKUType) (*gpu.GPUInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	found := f.found
	return &found, f.err
}

func TestMain(m *testing.M) {
	os.Exit(defaultingAdmissionEnv.RunM(m))
}

func TestDynamoGraphDeploymentDefaultingAdmission(t *testing.T) {
	tests := []struct {
		name                 string
		annotations          map[string]string
		component            nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		wantProvider         string
		wantReplicas         int32
		wantRoleReplicas     []int32
		addComponentOnUpdate bool
	}{
		{
			name: "defaults the complete DGD type tree",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "submitted-value",
				consts.KubeAnnotationWorkloadProvider:            "submitted-value",
			},
			component: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader},
					{Name: nvidiacomv1beta1.ComponentRoleWorker},
				},
			},
			wantProvider:         consts.WorkloadProviderGrove,
			wantReplicas:         1,
			wantRoleReplicas:     []int32{1, 3},
			addComponentOnUpdate: true,
		},
		{
			name: "preserves an explicit zero replica count",
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: consts.KubeLabelValueFalse,
			},
			component: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				Replicas:      new(int32(0)),
			},
			wantProvider: consts.WorkloadProviderComponent,
			wantReplicas: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := defaultingAdmissionEnv.ForTest(t)
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				Name: "graph", Namespace: env.Namespace(), Annotations: tt.annotations,
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{tt.component},
				},
			}

			t.Log("Create the DGD through the API server and production mutating handler")
			if err := env.Client().Create(t.Context(), dgd); err != nil {
				t.Fatalf("create DGD: %v", err)
			}
			if got := dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; got != defaultingAdmissionOperatorVersion {
				t.Fatalf("origin version = %q, want %q", got, defaultingAdmissionOperatorVersion)
			}
			if got := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; got != tt.wantProvider {
				t.Fatalf("workload provider = %q, want %q", got, tt.wantProvider)
			}
			component := &dgd.Spec.Components[0]
			if got := ptr.Deref(component.Replicas, -1); got != tt.wantReplicas {
				t.Fatalf("component replicas = %d, want %d", got, tt.wantReplicas)
			}
			for i, want := range tt.wantRoleReplicas {
				if got := ptr.Deref(component.Roles[i].Replicas, -1); got != want {
					t.Fatalf("role %d replicas = %d, want %d", i, got, want)
				}
			}
			assertCreateDefaultingIdempotent(t, env.Client(), dgd)

			if tt.addComponentOnUpdate {
				t.Log("Add a component on UPDATE and let structural traversal default the new element")
				dgd.Spec.Components = append(dgd.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "decode",
				})
				if err := env.Client().Update(t.Context(), dgd); err != nil {
					t.Fatalf("update DGD: %v", err)
				}
				if got := ptr.Deref(dgd.Spec.Components[1].Replicas, -1); got != 1 {
					t.Fatalf("new component replicas = %d, want 1", got)
				}
			}
		})
	}
}

func TestDynamoComponentDeploymentDefaultingAdmission(t *testing.T) {
	tests := []struct {
		name              string
		componentName     string
		roles             []nvidiacomv1beta1.ComponentRoleSpec
		wantComponentName string
		wantRoleReplicas  []int32
	}{
		{
			name: "defaults the standalone component name and roles",
			roles: []nvidiacomv1beta1.ComponentRoleSpec{
				{Name: nvidiacomv1beta1.ComponentRoleLeader},
				{Name: nvidiacomv1beta1.ComponentRoleWorker},
			},
			wantComponentName: "worker",
			wantRoleReplicas:  []int32{1, 3},
		},
		{
			name:              "preserves explicit component and role values",
			componentName:     "custom",
			roles:             []nvidiacomv1beta1.ComponentRoleSpec{{Name: nvidiacomv1beta1.ComponentRoleLeader, Replicas: new(int32(2))}, {Name: nvidiacomv1beta1.ComponentRoleWorker, Replicas: new(int32(2))}},
			wantComponentName: "custom",
			wantRoleReplicas:  []int32{2, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := defaultingAdmissionEnv.ForTest(t)
			dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
				Name: "worker", Namespace: env.Namespace(),
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					ComponentName: tt.componentName,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4},
					Roles:         tt.roles,
				},
			}

			t.Log("Create the DCD through the API server and production mutating handler")
			if err := env.Client().Create(t.Context(), dcd); err != nil {
				t.Fatalf("create DCD: %v", err)
			}
			if dcd.Spec.ComponentName != tt.wantComponentName {
				t.Fatalf("spec.name = %q, want %q", dcd.Spec.ComponentName, tt.wantComponentName)
			}
			for i, want := range tt.wantRoleReplicas {
				if got := ptr.Deref(dcd.Spec.Roles[i].Replicas, -1); got != want {
					t.Fatalf("role %d replicas = %d, want %d", i, got, want)
				}
			}
			assertCreateDefaultingIdempotent(t, env.Client(), dcd)
		})
	}
}

func TestDynamoGraphDeploymentRequestV1beta1DefaultingAdmission(t *testing.T) {
	wantDefaultSpec := nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
		Model:          "Qwen/Qwen3-8B",
		Backend:        nvidiacomv1beta1.BackendTypeAuto,
		Image:          "nvcr.io/nvidia/ai-dynamo/dynamo-planner:" + defaultingAdmissionOperatorVersion,
		SearchStrategy: nvidiacomv1beta1.SearchStrategyRapid,
		AutoApply:      new(true),
	}
	tests := []struct {
		name               string
		spec               nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec
		want               nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec
		checkIdempotence   bool
		clearImageOnUpdate bool
	}{
		{
			name: "combines schema and webhook defaults",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "Qwen/Qwen3-8B",
			},
			want:               wantDefaultSpec,
			checkIdempotence:   true,
			clearImageOnUpdate: true,
		},
		{
			name: "treats explicit schema defaults identically",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model:          "Qwen/Qwen3-8B",
				Backend:        nvidiacomv1beta1.BackendTypeAuto,
				SearchStrategy: nvidiacomv1beta1.SearchStrategyRapid,
				AutoApply:      new(true),
			},
			want: wantDefaultSpec,
		},
		{
			name:             "preserves an explicit profiler image",
			checkIdempotence: true,
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "Qwen/Qwen3-8B",
				Image: "registry.example/planner:custom",
			},
			want: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model:          "Qwen/Qwen3-8B",
				Backend:        nvidiacomv1beta1.BackendTypeAuto,
				Image:          "registry.example/planner:custom",
				SearchStrategy: nvidiacomv1beta1.SearchStrategyRapid,
				AutoApply:      new(true),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := defaultingAdmissionEnv.ForTest(t)
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Name: "request", Namespace: env.Namespace(),
				Spec: tt.spec,
			}

			t.Log("Create the DGDR through the API server and production mutating handler")
			if err := env.Client().Create(t.Context(), dgdr); err != nil {
				t.Fatalf("create DGDR: %v", err)
			}
			if !reflect.DeepEqual(dgdr.Spec, tt.want) {
				t.Fatalf("spec = %#v, want %#v", dgdr.Spec, tt.want)
			}
			if tt.checkIdempotence {
				assertCreateDefaultingIdempotent(t, env.Client(), dgdr)
			}

			if tt.clearImageOnUpdate {
				t.Log("Remove the image on UPDATE and verify the CREATE-only default stays absent")
				dgdr.Spec.Image = ""
				if err := env.Client().Update(t.Context(), dgdr); err != nil {
					t.Fatalf("update DGDR: %v", err)
				}
				if dgdr.Spec.Image != "" {
					t.Fatalf("image after UPDATE = %q, want empty", dgdr.Spec.Image)
				}
			}
		})
	}
}

func TestDynamoGraphDeploymentRequestV1beta2DefaultingAdmission(t *testing.T) {
	discoveredH200 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1, TotalGPUs: 8}
	v1beta2Tests := []struct {
		name          string
		request       *nvidiacomv1beta2.DynamoGraphDeploymentRequest
		discovered    gpu.GPUInfo
		discoveryErr  error
		wantCreateErr string
		check         func(*testing.T, *nvidiacomv1beta2.DynamoGraphDeploymentRequest)
		update        func(*nvidiacomv1beta2.DynamoGraphDeploymentRequest)
		beforeUpdate  func()
		checkUpdate   func(*testing.T, *nvidiacomv1beta2.DynamoGraphDeploymentRequest)
		idempotent    bool
	}{
		{
			name:       "v1beta2 model-only request combines all admission defaults",
			request:    v1beta2DefaultingRequest(),
			discovered: discoveredH200,
			check:      expectV1beta2ModelOnlyDefaults,
			idempotent: true,
		},
		{
			name: "v1beta2 explicit empty static workload and SLA opt out",
			request: v1beta2DefaultingRequest(func(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
				spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{}}
				spec.Objective = &nvidiacomv1beta2.ObjectiveSpec{SLA: &nvidiacomv1beta2.ObjectiveSLASpec{}}
			}),
			discovered: discoveredH200,
			check:      expectV1beta2ExplicitEmptyOptOuts,
		},
		{
			name: "v1beta2 explicit empty parent objects receive their defaults",
			request: v1beta2DefaultingRequest(func(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
				spec.Workload = &nvidiacomv1beta2.WorkloadSpec{}
				spec.Objective = &nvidiacomv1beta2.ObjectiveSpec{}
				spec.Recommendation = &nvidiacomv1beta2.RecommendationSpec{}
			}),
			discovered: discoveredH200,
			check:      expectV1beta2ModelOnlyDefaults,
		},
		{
			name: "v1beta2 explicit values and public input overrides are preserved",
			request: v1beta2DefaultingRequest(func(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
				spec.Backends = []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM}
				spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{InputTokens: new(int32(10))}}
				metric := nvidiacomv1beta2.ObjectiveMetric("TTFT")
				spec.Objective = &nvidiacomv1beta2.ObjectiveSpec{Metric: &metric}
				spec.Recommendation = &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 3}
				spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{InputOverrides: &runtime.RawExtension{
					Raw: []byte(`{"engine":{"hardware":"h100_sxm"},"custom":"kept"}`),
				}}
			}),
			discovered: discoveredH200,
			check:      expectV1beta2ExplicitValues,
		},
		{
			name: "v1beta2 unresolved explicit hardware reports its public field path",
			request: v1beta2DefaultingRequest(func(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
				spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
					SKU: new("h100_sxm"), Budget: new(int32(4)),
				}}
			}),
			discoveryErr:  errors.New("no matching GPUs found"),
			wantCreateErr: `spec.hardware.gpu.sku: Invalid value: "h100_sxm": could not be verified against the cluster's GPUs (no matching GPUs found)`,
		},
		{
			name:       "v1beta2 metadata-only update is idempotent and does not rediscover hardware",
			request:    v1beta2DefaultingRequest(),
			discovered: discoveredH200,
			update: func(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
				request.Labels = map[string]string{"updated": "true"}
			},
			beforeUpdate: func() {
				defaultingAdmissionHardware.set(gpu.GPUInfo{}, errors.New("cluster unreadable"))
			},
			checkUpdate: expectV1beta2MetadataOnlyUpdate,
		},
		{
			name: "v1beta2 removing hardware on update rediscovers both fields",
			request: v1beta2DefaultingRequest(func(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
				spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
					SKU: new("h200_sxm"), Budget: new(int32(4)),
				}}
			}),
			discovered: discoveredH200,
			update: func(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
				request.Spec.Hardware = nil
			},
			checkUpdate: expectV1beta2HardwareUpdate,
		},
	}

	for _, tt := range v1beta2Tests {
		t.Run(tt.name, func(t *testing.T) {
			defaultingAdmissionHardware.set(tt.discovered, tt.discoveryErr)
			t.Cleanup(func() { defaultingAdmissionHardware.set(gpu.GPUInfo{}, nil) })
			env := defaultingAdmissionEnv.ForTest(t)
			request := tt.request.DeepCopy()
			request.Name, request.Namespace = "request", env.Namespace()

			t.Log("Create the v1beta2 DGDR through the API server and production mutating handler")
			err := env.Client().Create(t.Context(), request)
			if tt.wantCreateErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantCreateErr) {
					t.Fatalf("create v1beta2 DGDR error = %v, want it to contain %q", err, tt.wantCreateErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("create v1beta2 DGDR: %v", err)
			}
			if tt.check != nil {
				tt.check(t, request)
			}
			if tt.idempotent {
				assertCreateDefaultingIdempotent(t, env.Client(), request)
			}

			if tt.update != nil {
				tt.update(request)
				if tt.beforeUpdate != nil {
					tt.beforeUpdate()
				}
				t.Log("Update the v1beta2 DGDR through the same admission chain")
				if err := env.Client().Update(t.Context(), request); err != nil {
					t.Fatalf("update v1beta2 DGDR: %v", err)
				}
				tt.checkUpdate(t, request)
			}
		})
	}
}

func v1beta2DefaultingRequest(mutate ...func(*nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec)) *nvidiacomv1beta2.DynamoGraphDeploymentRequest {
	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
			DynamoGraphDeploymentRequestNativeSpec: nvidiacomv1beta2.DynamoGraphDeploymentRequestNativeSpec{
				NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
					Model: &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-8B"},
				},
			},
		},
	}
	for _, apply := range mutate {
		apply(&request.Spec)
	}
	return request
}

func expectV1beta2ModelOnlyDefaults(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	t.Helper()
	if want := []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendSGLang, nvidiacomv1beta2.BackendTRTLLM}; !reflect.DeepEqual(request.Spec.Backends, want) {
		t.Fatalf("backends = %v, want %v", request.Spec.Backends, want)
	}
	expectV1beta2H200x8Hardware(t, request)
	if request.Spec.Workload == nil {
		t.Fatal("workload is nil, want model-only defaults")
	}
	static := request.Spec.Workload.Static
	if static == nil || ptr.Deref(static.InputTokens, 0) != 1024 || ptr.Deref(static.OutputTokens, 0) != 128 || static.Load == nil || ptr.Deref(static.Load.Concurrency, 0) != 10 {
		t.Fatalf("workload = %#v, want 1024/128 tokens at concurrency 10", request.Spec.Workload)
	}
	objective := request.Spec.Objective
	if objective == nil || objective.Mode != nvidiacomv1beta2.ObjectiveModeOptimize || ptr.Deref(objective.Metric, "") != nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU") || objective.SLA == nil || ptr.Deref(objective.SLA.TTFTMs, 0) != 2000 || ptr.Deref(objective.SLA.ITLMs, 0) != 30 {
		t.Fatalf("objective = %#v, want Optimize/ThroughputPerGPU with 2000/30 ms SLA", objective)
	}
	if request.Spec.Recommendation == nil || request.Spec.Recommendation.MaxCandidates != 1 {
		t.Fatalf("recommendation = %#v, want maxCandidates 1", request.Spec.Recommendation)
	}
}

func expectV1beta2ExplicitEmptyOptOuts(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	t.Helper()
	if got := request.Spec.Workload.Static; got == nil || got.InputTokens != nil || got.OutputTokens != nil || got.Load != nil {
		t.Fatalf("workload.static = %#v, want explicit empty opt-out", got)
	}
	if got := request.Spec.Objective.SLA; got == nil || got.TTFTMs != nil || got.ITLMs != nil || got.E2EMs != nil {
		t.Fatalf("objective.sla = %#v, want explicit empty opt-out", got)
	}
	if request.Spec.Objective.Mode != nvidiacomv1beta2.ObjectiveModeOptimize || request.Spec.Objective.Metric == nil {
		t.Fatalf("objective = %#v, want mode and metric defaults beside empty SLA", request.Spec.Objective)
	}
}

func expectV1beta2ExplicitValues(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	t.Helper()
	if want := []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM}; !reflect.DeepEqual(request.Spec.Backends, want) {
		t.Fatalf("backends = %v, want %v", request.Spec.Backends, want)
	}
	if request.Spec.Workload == nil || request.Spec.Workload.Static == nil {
		t.Fatalf("workload = %#v, want the partial static workload", request.Spec.Workload)
	}
	static := request.Spec.Workload.Static
	if ptr.Deref(static.InputTokens, 0) != 10 || ptr.Deref(static.OutputTokens, 0) != 128 || static.Load == nil || ptr.Deref(static.Load.Concurrency, 0) != 10 {
		t.Fatalf("workload.static = %#v, want explicit inputTokens and omitted-member defaults", static)
	}
	if request.Spec.Objective.Mode != nvidiacomv1beta2.ObjectiveModeOptimize || ptr.Deref(request.Spec.Objective.Metric, "") != nvidiacomv1beta2.ObjectiveMetric("TTFT") {
		t.Fatalf("objective = %#v, want explicit TTFT metric with Optimize default", request.Spec.Objective)
	}
	if request.Spec.Recommendation.MaxCandidates != 3 {
		t.Fatalf("maxCandidates = %d, want 3", request.Spec.Recommendation.MaxCandidates)
	}
	if request.Spec.AISimulate == nil || request.Spec.AISimulate.InputOverrides == nil {
		t.Fatalf("aisimulate = %#v, want input overrides", request.Spec.AISimulate)
	}
	var overrides map[string]any
	if err := json.Unmarshal(request.Spec.AISimulate.InputOverrides.Raw, &overrides); err != nil {
		t.Fatalf("decode input overrides: %v", err)
	}
	if overrides["custom"] != "kept" {
		t.Fatalf("inputOverrides = %#v, want custom value untouched", overrides)
	}
	engine, _ := overrides["engine"].(map[string]any)
	if engine["hardware"] != "h100_sxm" {
		t.Fatalf("inputOverrides = %#v, want engine.hardware untouched", overrides)
	}
	expectV1beta2H200x8Hardware(t, request)
}

func expectV1beta2MetadataOnlyUpdate(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	t.Helper()
	expectV1beta2H200x8Hardware(t, request)
	if request.Generation != 1 {
		t.Fatalf("generation = %d, want 1 after metadata-only update", request.Generation)
	}
}

func expectV1beta2HardwareUpdate(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	t.Helper()
	expectV1beta2H200x8Hardware(t, request)
	if request.Generation != 2 {
		t.Fatalf("generation = %d, want 2 after hardware update", request.Generation)
	}
}

func expectV1beta2H200x8Hardware(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	t.Helper()
	hardware := request.Spec.Hardware
	if hardware == nil || hardware.GPU == nil || ptr.Deref(hardware.GPU.SKU, "") != "h200_sxm" || ptr.Deref(hardware.GPU.Budget, 0) != 8 {
		t.Fatalf("hardware = %#v, want SKU h200_sxm and budget 8", hardware)
	}
}

func assertCreateDefaultingIdempotent(t *testing.T, kubeClient client.Client, object client.Object) {
	t.Helper()
	before := defaultingAdmissionState(t, object)
	repeated, ok := object.DeepCopyObject().(client.Object)
	if !ok {
		t.Fatalf("deep copy of %T is not a client.Object", object)
	}
	repeated.SetName(object.GetName() + "-defaulted")
	repeated.SetResourceVersion("")
	repeated.SetUID("")
	repeated.SetGeneration(0)
	repeated.SetCreationTimestamp(metav1.Time{})
	repeated.SetManagedFields(nil)
	if err := kubeClient.Create(t.Context(), repeated); err != nil {
		t.Fatalf("repeat CREATE defaulting: %v", err)
	}
	after := defaultingAdmissionState(t, repeated)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("defaulting is not idempotent:\nbefore: %#v\nafter: %#v", before, after)
	}
}

func defaultingAdmissionState(t *testing.T, object runtime.Object) map[string]any {
	t.Helper()
	value, err := runtime.DefaultUnstructuredConverter.ToUnstructured(object)
	if err != nil {
		t.Fatalf("convert %T to unstructured: %v", object, err)
	}
	metadata, _ := value["metadata"].(map[string]any)
	return map[string]any{
		"annotations": metadata["annotations"],
		"spec":        value["spec"],
	}
}
