package cuda

import (
	"context"
	"errors"
	"net"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	testGPUUUIDA = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
	testGPUUUIDB = "GPU-bbbbbbbb-5555-6666-7777-888888888888"
	testGPUUUIDC = "GPU-cccccccc-9999-aaaa-bbbb-cccccccccccc"
	testGPUUUIDD = "GPU-dddddddd-eeee-ffff-0000-111111111111"
)

func TestBuildDeviceMap(t *testing.T) {
	tests := []struct {
		name          string
		source        []string
		target        []string
		want          string
		wantPlacement []types.VMMPlacement
		wantErr       bool
	}{
		{
			name:   "single GPU",
			source: []string{testGPUUUIDA},
			target: []string{testGPUUUIDB},
			want:   testGPUUUIDA + "=" + testGPUUUIDB,
			wantPlacement: []types.VMMPlacement{{
				SourceGPUUUID: strings.TrimPrefix(testGPUUUIDA, "GPU-"),
				TargetGPUUUID: strings.TrimPrefix(testGPUUUIDB, "GPU-"),
				TargetOrdinal: 0,
			}},
		},
		{
			name:   "single GPU identity returns no map",
			source: []string{testGPUUUIDA},
			target: []string{testGPUUUIDA},
			want:   "",
			wantPlacement: []types.VMMPlacement{{
				SourceGPUUUID: strings.TrimPrefix(testGPUUUIDA, "GPU-"),
				TargetGPUUUID: strings.TrimPrefix(testGPUUUIDA, "GPU-"),
				TargetOrdinal: 0,
			}},
		},
		{
			name:   "multiple GPUs",
			source: []string{testGPUUUIDA, testGPUUUIDB},
			target: []string{testGPUUUIDC, testGPUUUIDD},
			want: testGPUUUIDA + "=" + testGPUUUIDC + "," +
				testGPUUUIDB + "=" + testGPUUUIDD,
			wantPlacement: []types.VMMPlacement{
				{
					SourceGPUUUID: strings.TrimPrefix(testGPUUUIDA, "GPU-"),
					TargetGPUUUID: strings.TrimPrefix(testGPUUUIDC, "GPU-"),
					TargetOrdinal: 0,
				},
				{
					SourceGPUUUID: strings.TrimPrefix(testGPUUUIDB, "GPU-"),
					TargetGPUUUID: strings.TrimPrefix(testGPUUUIDD, "GPU-"),
					TargetOrdinal: 1,
				},
			},
		},
		{
			name:   "multiple GPU identity returns no map",
			source: []string{testGPUUUIDA, testGPUUUIDB},
			target: []string{testGPUUUIDB, testGPUUUIDA},
			want:   "",
			wantPlacement: []types.VMMPlacement{
				{
					SourceGPUUUID: strings.TrimPrefix(testGPUUUIDA, "GPU-"),
					TargetGPUUUID: strings.TrimPrefix(testGPUUUIDA, "GPU-"),
					TargetOrdinal: 1,
				},
				{
					SourceGPUUUID: strings.TrimPrefix(testGPUUUIDB, "GPU-"),
					TargetGPUUUID: strings.TrimPrefix(testGPUUUIDB, "GPU-"),
					TargetOrdinal: 0,
				},
			},
		},
		{
			name:    "mismatched lengths",
			source:  []string{testGPUUUIDA, testGPUUUIDB},
			target:  []string{testGPUUUIDC},
			wantErr: true,
		},
		{
			name:    "both empty",
			source:  []string{},
			target:  []string{},
			wantErr: true,
		},
		{
			name:    "source empty target non-empty",
			source:  []string{},
			target:  []string{testGPUUUIDA},
			wantErr: true,
		},
		{
			name:    "malformed source",
			source:  []string{"GPU-not-a-uuid"},
			target:  []string{testGPUUUIDA},
			wantErr: true,
		},
		{
			name:    "duplicate source",
			source:  []string{testGPUUUIDA, testGPUUUIDA},
			target:  []string{testGPUUUIDB, testGPUUUIDC},
			wantErr: true,
		},
		{
			name:    "duplicate target",
			source:  []string{testGPUUUIDA, testGPUUUIDB},
			target:  []string{testGPUUUIDC, testGPUUUIDC},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := BuildDeviceMap(tc.source, tc.target, logr.Discard())
			placement, placementErr := BuildVMMPlacement(tc.source, tc.target)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error, got %q", got)
				}
				if placementErr == nil {
					t.Errorf("expected placement error, got %#v", placement)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if placementErr != nil {
				t.Fatalf("unexpected placement error: %v", placementErr)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
			if !slices.Equal(placement, tc.wantPlacement) {
				t.Errorf("placement = %#v, want %#v", placement, tc.wantPlacement)
			}
		})
	}
}

func TestValidateVMMPlacementRejectsInvalidPlans(t *testing.T) {
	valid := []types.VMMPlacement{
		{
			SourceGPUUUID: strings.TrimPrefix(testGPUUUIDA, "GPU-"),
			TargetGPUUUID: strings.TrimPrefix(testGPUUUIDC, "GPU-"),
			TargetOrdinal: 0,
		},
		{
			SourceGPUUUID: strings.TrimPrefix(testGPUUUIDB, "GPU-"),
			TargetGPUUUID: strings.TrimPrefix(testGPUUUIDD, "GPU-"),
			TargetOrdinal: 1,
		},
	}
	if err := ValidateVMMPlacement(
		[]string{testGPUUUIDA, testGPUUUIDB},
		valid,
	); err != nil {
		t.Fatalf("valid placement rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func([]types.VMMPlacement) []types.VMMPlacement
	}{
		{
			name: "count mismatch",
			mutate: func(plan []types.VMMPlacement) []types.VMMPlacement {
				return plan[:1]
			},
		},
		{
			name: "noncanonical source UUID",
			mutate: func(plan []types.VMMPlacement) []types.VMMPlacement {
				plan[0].SourceGPUUUID = strings.ToUpper(plan[0].SourceGPUUUID)
				return plan
			},
		},
		{
			name: "duplicate target ordinal",
			mutate: func(plan []types.VMMPlacement) []types.VMMPlacement {
				plan[1].TargetOrdinal = 0
				return plan
			},
		},
		{
			name: "source order mismatch",
			mutate: func(plan []types.VMMPlacement) []types.VMMPlacement {
				plan[0], plan[1] = plan[1], plan[0]
				return plan
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan := slices.Clone(valid)
			if err := ValidateVMMPlacement(
				[]string{testGPUUUIDA, testGPUUUIDB},
				test.mutate(plan),
			); err == nil {
				t.Fatal("invalid placement accepted")
			}
		})
	}
}

type testPodResourcesServer struct {
	podresourcesv1.UnimplementedPodResourcesListerServer
	resp *podresourcesv1.ListPodResourcesResponse
}

func (s *testPodResourcesServer) List(context.Context, *podresourcesv1.ListPodResourcesRequest) (*podresourcesv1.ListPodResourcesResponse, error) {
	return s.resp, nil
}

func (s *testPodResourcesServer) GetAllocatableResources(context.Context, *podresourcesv1.AllocatableResourcesRequest) (*podresourcesv1.AllocatableResourcesResponse, error) {
	return nil, status.Error(codes.Unimplemented, "not implemented in test")
}

func (s *testPodResourcesServer) Get(context.Context, *podresourcesv1.GetPodResourcesRequest) (*podresourcesv1.GetPodResourcesResponse, error) {
	return nil, status.Error(codes.Unimplemented, "not implemented in test")
}

func installTestPodResourcesServer(t *testing.T, resp *podresourcesv1.ListPodResourcesResponse) {
	socketDir := t.TempDir()
	socketPath := filepath.Join(socketDir, "kubelet.sock")

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen unix socket: %v", err)
	}

	server := grpc.NewServer()
	podresourcesv1.RegisterPodResourcesListerServer(server, &testPodResourcesServer{
		resp: resp,
	})

	go func() {
		if serveErr := server.Serve(listener); serveErr != nil {
			if errors.Is(serveErr, grpc.ErrServerStopped) || strings.Contains(serveErr.Error(), "use of closed network connection") {
				return
			}
			t.Errorf("serve test pod-resources gRPC server: %v", serveErr)
		}
	}()
	t.Cleanup(server.Stop)
	t.Cleanup(func() {
		_ = listener.Close()
	})

	previousSocketPath := podResourcesSocketPath
	podResourcesSocketPath = socketPath
	t.Cleanup(func() {
		podResourcesSocketPath = previousSocketPath
	})
}

func TestGetPodGPUUUIDs(t *testing.T) {
	installTestPodResourcesServer(t, &podresourcesv1.ListPodResourcesResponse{
		PodResources: []*podresourcesv1.PodResources{
			{
				Name:      "other-pod",
				Namespace: "default",
				Containers: []*podresourcesv1.ContainerResources{
					{
						Name: "main",
						Devices: []*podresourcesv1.ContainerDevices{
							{
								ResourceName: nvidiaGPUResource,
								DeviceIds:    []string{"GPU-ignore"},
							},
						},
					},
				},
			},
			{
				Name:      "test-pod",
				Namespace: "default",
				Containers: []*podresourcesv1.ContainerResources{
					{
						Name: "sidecar",
						Devices: []*podresourcesv1.ContainerDevices{
							{
								ResourceName: nvidiaGPUResource,
								DeviceIds:    []string{"GPU-sidecar"},
							},
						},
					},
					{
						Name: "main",
						Devices: []*podresourcesv1.ContainerDevices{
							{
								ResourceName: nvidiaGPUResource,
								DeviceIds:    []string{"GPU-a", "GPU-b"},
							},
							{
								ResourceName: "example.com/fpga",
								DeviceIds:    []string{"FPGA-ignore"},
							},
							{
								ResourceName: nvidiaGPUResource,
								DeviceIds:    []string{"GPU-c"},
							},
						},
					},
				},
			},
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	got, err := GetPodGPUUUIDs(ctx, "test-pod", "default", "main")
	if err != nil {
		t.Fatalf("GetPodGPUUUIDs: %v", err)
	}

	want := []string{"GPU-a", "GPU-b", "GPU-c"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestDiscoverGPUUUIDsUsesPodResourcesForClassicPod(t *testing.T) {
	installTestPodResourcesServer(t, &podresourcesv1.ListPodResourcesResponse{
		PodResources: []*podresourcesv1.PodResources{
			{
				Name:      "test-pod",
				Namespace: "default",
				Containers: []*podresourcesv1.ContainerResources{
					{
						Name: "main",
						Devices: []*podresourcesv1.ContainerDevices{
							{
								ResourceName: nvidiaGPUResource,
								DeviceIds:    []string{"GPU-a", "GPU-b"},
							},
						},
					},
				},
			},
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	got, err := DiscoverGPUUUIDs(
		ctx,
		nil,
		"test-pod",
		"default",
		"main",
		"/proc",
		123,
		logr.Discard(),
	)
	if err != nil {
		t.Fatalf("DiscoverGPUUUIDs: %v", err)
	}

	want := []string{"GPU-a", "GPU-b"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestDiscoverGPUUUIDsFallsBackToPodResourcesAfterDRAAPILookupError(t *testing.T) {
	installTestPodResourcesServer(t, &podresourcesv1.ListPodResourcesResponse{
		PodResources: []*podresourcesv1.PodResources{
			{
				Name:      "test-pod",
				Namespace: "default",
				Containers: []*podresourcesv1.ContainerResources{
					{
						Name: "main",
						Devices: []*podresourcesv1.ContainerDevices{
							{
								ResourceName: nvidiaGPUResource,
								DeviceIds:    []string{"GPU-a"},
							},
						},
					},
				},
			},
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	got, err := DiscoverGPUUUIDs(
		ctx,
		fake.NewSimpleClientset(),
		"test-pod",
		"default",
		"main",
		"/proc",
		123,
		logr.Discard(),
	)
	if err != nil {
		t.Fatalf("DiscoverGPUUUIDs: %v", err)
	}
	if len(got) != 1 || got[0] != "GPU-a" {
		t.Fatalf("got %v, want [GPU-a]", got)
	}
}

func TestDiscoverGPUUUIDsOrdersDRAPodByContainerOrdinal(t *testing.T) {
	previousSocketPath := podResourcesSocketPath
	podResourcesSocketPath = filepath.Join(t.TempDir(), "missing-kubelet.sock")
	t.Cleanup(func() {
		podResourcesSocketPath = previousSocketPath
	})

	nodeName := "node-1"
	poolName := "pool-node-1"
	namespace := "default"
	podName := "test-pod"
	claimName := "gpu-claim"
	uuid0 := "GPU-aaaaaaaa-1111-2222-3333-444444444444"
	uuid1 := "GPU-bbbbbbbb-5555-6666-7777-888888888888"

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: namespace},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{
				{
					Name: "main",
					Resources: corev1.ResourceRequirements{
						Claims: []corev1.ResourceClaim{{Name: "gpu"}},
					},
				},
			},
			ResourceClaims: []corev1.PodResourceClaim{
				{
					Name:              "gpu",
					ResourceClaimName: &claimName,
				},
			},
		},
	}
	claim := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: claimName, Namespace: namespace},
		Status: resourcev1.ResourceClaimStatus{
			Allocation: &resourcev1.AllocationResult{
				Devices: resourcev1.DeviceAllocationResult{
					Results: []resourcev1.DeviceRequestAllocationResult{
						{Driver: nvidiaGPUDRADriver, Pool: poolName, Device: "gpu-1", Request: "gpu"},
						{Driver: nvidiaGPUDRADriver, Pool: poolName, Device: "gpu-0", Request: "gpu"},
					},
				},
			},
		},
	}
	slice := &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: poolName + "-gpu.nvidia.com-xxx"},
		Spec: resourcev1.ResourceSliceSpec{
			Driver:   nvidiaGPUDRADriver,
			NodeName: &nodeName,
			Pool:     resourcev1.ResourcePool{Name: poolName},
			Devices: []resourcev1.Device{
				{
					Name: "gpu-0",
					Attributes: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
						resourcev1.QualifiedName("uuid"): {StringValue: &uuid0},
					},
				},
				{
					Name: "gpu-1",
					Attributes: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
						resourcev1.QualifiedName("uuid"): {StringValue: &uuid1},
					},
				},
			},
		},
	}

	client := fake.NewSimpleClientset(pod, claim, slice)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	got, err := discoverGPUUUIDs(
		ctx,
		client,
		podName,
		namespace,
		"main",
		"/proc",
		123,
		func(context.Context, string, int) ([]string, error) {
			return []string{uuid0, uuid1}, nil
		},
		logr.Discard(),
	)
	if err != nil {
		t.Fatalf("DiscoverGPUUUIDs: %v", err)
	}
	want := []string{uuid0, uuid1}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestOrderDRAUUIDsByRuntimeRejectsMismatches(t *testing.T) {
	uuid0 := "GPU-aaaaaaaa-1111-2222-3333-444444444444"
	uuid1 := "GPU-bbbbbbbb-5555-6666-7777-888888888888"
	uuid2 := "GPU-cccccccc-9999-aaaa-bbbb-cccccccccccc"

	tests := []struct {
		name      string
		allocated []string
		visible   []string
	}{
		{
			name:      "count mismatch",
			allocated: []string{uuid0, uuid1},
			visible:   []string{uuid0},
		},
		{
			name:      "different set",
			allocated: []string{uuid0, uuid1},
			visible:   []string{uuid0, uuid2},
		},
		{
			name:      "duplicate allocation",
			allocated: []string{uuid0, uuid0},
			visible:   []string{uuid0, uuid1},
		},
		{
			name:      "invalid allocation UUID",
			allocated: []string{uuid0, "not-a-gpu-uuid"},
			visible:   []string{uuid0, uuid1},
		},
		{
			name:      "duplicate visible",
			allocated: []string{uuid0, uuid1},
			visible:   []string{uuid0, uuid0},
		},
		{
			name:      "invalid visible UUID",
			allocated: []string{uuid0, uuid1},
			visible:   []string{uuid0, "not-a-gpu-uuid"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, err := orderDRAUUIDsByRuntime(tc.allocated, tc.visible); err == nil {
				t.Fatalf("expected error, got %v", got)
			}
		})
	}
}
