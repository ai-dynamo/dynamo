package cuda

import (
	"context"
	"errors"
	"net"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

func TestBuildDeviceMap(t *testing.T) {
	tests := []struct {
		name    string
		source  []string
		target  []string
		want    string
		wantErr bool
	}{
		{
			name:   "single GPU",
			source: []string{"GPU-aaa"},
			target: []string{"GPU-bbb"},
			want:   "GPU-aaa=GPU-bbb",
		},
		{
			name:   "multiple GPUs",
			source: []string{"GPU-aaa", "GPU-bbb"},
			target: []string{"GPU-ccc", "GPU-ddd"},
			want:   "GPU-aaa=GPU-ccc,GPU-bbb=GPU-ddd",
		},
		{
			name:    "mismatched lengths",
			source:  []string{"GPU-aaa", "GPU-bbb"},
			target:  []string{"GPU-ccc"},
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
			target:  []string{"GPU-aaa"},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := BuildDeviceMap(tc.source, tc.target, logr.Discard())
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
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

func TestExtractGPUUUIDFromCDIDevices(t *testing.T) {
	tests := []struct {
		name string
		cdis []*podresourcesv1.CDIDevice
		want string
	}{
		{
			name: "valid UUID in CDI name",
			cdis: []*podresourcesv1.CDIDevice{
				{Name: "gpu.nvidia.com/gpu=GPU-551720f0-caf0-22b7-f525-2a51a6ab478d"},
			},
			want: "GPU-551720f0-caf0-22b7-f525-2a51a6ab478d",
		},
		{
			name: "multiple CDI devices, second has UUID",
			cdis: []*podresourcesv1.CDIDevice{
				{Name: "gpu.nvidia.com/mps=some-partition"},
				{Name: "gpu.nvidia.com/gpu=GPU-aabbccdd-1122-3344-5566-778899aabbcc"},
			},
			want: "GPU-aabbccdd-1122-3344-5566-778899aabbcc",
		},
		{
			name: "no valid UUID",
			cdis: []*podresourcesv1.CDIDevice{
				{Name: "gpu.nvidia.com/gpu=gpu-0"},
			},
			want: "",
		},
		{
			name: "empty CDI list",
			cdis: nil,
			want: "",
		},
		{
			name: "malformed CDI name without equals sign",
			cdis: []*podresourcesv1.CDIDevice{
				{Name: "gpu.nvidia.com/gpu"},
			},
			want: "",
		},
		{
			name: "non-NVIDIA CDI name ignored",
			cdis: []*podresourcesv1.CDIDevice{
				{Name: "other.vendor.com/device=not-a-gpu-uuid"},
			},
			want: "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := extractGPUUUIDFromCDIDevices(tc.cdis)
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestGetPodGPUUUIDs(t *testing.T) {
	socketDir := t.TempDir()
	socketPath := filepath.Join(socketDir, "kubelet.sock")

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen unix socket: %v", err)
	}
	defer listener.Close()

	server := grpc.NewServer()
	podresourcesv1.RegisterPodResourcesListerServer(server, &testPodResourcesServer{
		resp: &podresourcesv1.ListPodResourcesResponse{
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
		},
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

	previousSocketPath := podResourcesSocketPath
	podResourcesSocketPath = socketPath
	t.Cleanup(func() {
		podResourcesSocketPath = previousSocketPath
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	got, err := GetPodGPUUUIDs(ctx, "test-pod", "default", "main", logr.Discard())
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

func TestGetPodGPUUUIDs_DRA(t *testing.T) {
	tests := []struct {
		name string
		resp *podresourcesv1.ListPodResourcesResponse
		want []string
	}{
		{
			name: "valid UUID in CDI device name",
			resp: &podresourcesv1.ListPodResourcesResponse{
				PodResources: []*podresourcesv1.PodResources{{
					Name: "dra-pod", Namespace: "default",
					Containers: []*podresourcesv1.ContainerResources{{
						Name: "main",
						DynamicResources: []*podresourcesv1.DynamicResource{{
							ClaimName: "gpu-claim", ClaimNamespace: "default",
							ClaimResources: []*podresourcesv1.ClaimResource{
								{
									DriverName: nvidiaGPUDRADriver,
									DeviceName: "gpu-0",
									CdiDevices: []*podresourcesv1.CDIDevice{
										{Name: "gpu.nvidia.com/gpu=GPU-aabbccdd-1122-3344-5566-778899aabbcc"},
									},
								},
								{
									DriverName: nvidiaGPUDRADriver,
									DeviceName: "gpu-1",
									CdiDevices: []*podresourcesv1.CDIDevice{
										{Name: "gpu.nvidia.com/gpu=GPU-11223344-aabb-ccdd-eeff-556677889900"},
									},
								},
							},
						}},
					}},
				}},
			},
			want: []string{
				"GPU-aabbccdd-1122-3344-5566-778899aabbcc",
				"GPU-11223344-aabb-ccdd-eeff-556677889900",
			},
		},
		{
			name: "opaque CDI names yield empty result for fallback",
			resp: &podresourcesv1.ListPodResourcesResponse{
				PodResources: []*podresourcesv1.PodResources{{
					Name: "dra-pod", Namespace: "default",
					Containers: []*podresourcesv1.ContainerResources{{
						Name: "main",
						DynamicResources: []*podresourcesv1.DynamicResource{{
							ClaimName: "gpu-claim", ClaimNamespace: "default",
							ClaimResources: []*podresourcesv1.ClaimResource{{
								DriverName: nvidiaGPUDRADriver,
								DeviceName: "gpu-0",
								CdiDevices: []*podresourcesv1.CDIDevice{
									{Name: "gpu.nvidia.com/gpu=gpu-0"},
								},
							}},
						}},
					}},
				}},
			},
			want: nil,
		},
		{
			name: "no CDI devices yields empty result for fallback",
			resp: &podresourcesv1.ListPodResourcesResponse{
				PodResources: []*podresourcesv1.PodResources{{
					Name: "dra-pod", Namespace: "default",
					Containers: []*podresourcesv1.ContainerResources{{
						Name: "main",
						DynamicResources: []*podresourcesv1.DynamicResource{{
							ClaimName: "gpu-claim", ClaimNamespace: "default",
							ClaimResources: []*podresourcesv1.ClaimResource{{
								DriverName: nvidiaGPUDRADriver,
								DeviceName: "gpu-0",
							}},
						}},
					}},
				}},
			},
			want: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			socketDir := t.TempDir()
			socketPath := filepath.Join(socketDir, "kubelet.sock")
			listener, err := net.Listen("unix", socketPath)
			if err != nil {
				t.Fatalf("listen unix socket: %v", err)
			}
			defer listener.Close()

			srv := grpc.NewServer()
			podresourcesv1.RegisterPodResourcesListerServer(srv, &testPodResourcesServer{resp: tc.resp})
			go func() {
				if serveErr := srv.Serve(listener); serveErr != nil {
					if errors.Is(serveErr, grpc.ErrServerStopped) || strings.Contains(serveErr.Error(), "use of closed network connection") {
						return
					}
					t.Errorf("serve: %v", serveErr)
				}
			}()
			t.Cleanup(srv.Stop)

			prev := podResourcesSocketPath
			podResourcesSocketPath = socketPath
			t.Cleanup(func() { podResourcesSocketPath = prev })

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			got, err := GetPodGPUUUIDs(ctx, "dra-pod", "default", "main", logr.Discard())
			if err != nil {
				t.Fatalf("GetPodGPUUUIDs: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v (len %d), want %v (len %d)", got, len(got), tc.want, len(tc.want))
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("got[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}
