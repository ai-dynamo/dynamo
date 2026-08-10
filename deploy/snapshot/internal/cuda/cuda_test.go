package cuda

import (
	"context"
	"errors"
	"fmt"
	"net"
	"path/filepath"
	"strings"
	"sync/atomic"
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
			name:   "single GPU identity returns no map",
			source: []string{"GPU-aaa"},
			target: []string{"GPU-aaa"},
			want:   "",
		},
		{
			name:   "multiple GPUs",
			source: []string{"GPU-aaa", "GPU-bbb"},
			target: []string{"GPU-ccc", "GPU-ddd"},
			want:   "GPU-aaa=GPU-ccc,GPU-bbb=GPU-ddd",
		},
		{
			name:   "multiple GPU identity returns no map",
			source: []string{"GPU-aaa", "GPU-bbb"},
			target: []string{"GPU-bbb", "GPU-aaa"},
			want:   "",
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

func TestRestoreAndUnlockProcessTreeConcurrentPhaseBarrier(t *testing.T) {
	pids := []int{101, 202, 303}
	restoreStarted := make(chan int, len(pids))
	releaseRestore := make(chan struct{})
	unlockStarted := make(chan int, len(pids))
	errCh := make(chan error, 1)

	go func() {
		_, err := restoreAndUnlockProcessTree(
			context.Background(),
			pids,
			true,
			func(ctx context.Context, pid int) error {
				restoreStarted <- pid
				select {
				case <-releaseRestore:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			},
			func(_ context.Context, pid int) error {
				unlockStarted <- pid
				return nil
			},
			func(context.Context, int) (string, error) {
				return "", errors.New("unexpected state check")
			},
			logr.Discard(),
		)
		errCh <- err
	}()

	for range pids {
		select {
		case <-restoreStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("restore operations did not all start concurrently")
		}
	}
	select {
	case pid := <-unlockStarted:
		t.Fatalf("unlock for pid %d started before all restores completed", pid)
	default:
	}

	close(releaseRestore)
	for range pids {
		select {
		case <-unlockStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("unlock operations did not all start")
		}
	}

	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("restoreAndUnlockProcessTree: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("restoreAndUnlockProcessTree did not complete")
	}
}

func TestRestoreAndUnlockProcessTreeSerialOrdering(t *testing.T) {
	pids := []int{303, 101, 202}
	var calls []string

	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		pids,
		false,
		func(_ context.Context, pid int) error {
			calls = append(calls, fmt.Sprintf("restore:%d", pid))
			return nil
		},
		func(_ context.Context, pid int) error {
			calls = append(calls, fmt.Sprintf("unlock:%d", pid))
			return nil
		},
		func(context.Context, int) (string, error) {
			return "", errors.New("unexpected state check")
		},
		logr.Discard(),
	)
	if err != nil {
		t.Fatalf("restoreAndUnlockProcessTree: %v", err)
	}

	want := []string{
		"restore:303",
		"restore:101",
		"restore:202",
		"unlock:303",
		"unlock:101",
		"unlock:202",
	}
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Fatalf("calls %v, want %v", calls, want)
	}
}

func TestRestoreAndUnlockProcessTreeWaitsAndOrdersErrors(t *testing.T) {
	pids := []int{303, 101, 202}
	var completed atomic.Int32
	var unlockCalls atomic.Int32

	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		pids,
		true,
		func(_ context.Context, pid int) error {
			defer completed.Add(1)
			if pid == 202 {
				time.Sleep(20 * time.Millisecond)
				return nil
			}
			return fmt.Errorf("restore error %d", pid)
		},
		func(context.Context, int) error {
			unlockCalls.Add(1)
			return nil
		},
		func(context.Context, int) (string, error) {
			return "", errors.New("unexpected state check")
		},
		logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected restore error")
	}
	if completed.Load() != int32(len(pids)) {
		t.Fatalf("completed %d restores, want %d", completed.Load(), len(pids))
	}
	if unlockCalls.Load() != 0 {
		t.Fatalf("unlock called %d times after restore failure", unlockCalls.Load())
	}
	errText := err.Error()
	first := strings.Index(errText, "pid 303")
	second := strings.Index(errText, "pid 101")
	if first < 0 || second < 0 || first >= second {
		t.Fatalf("errors are not in PID input order: %q", errText)
	}
}

func TestRestoreAndUnlockProcessTreeAcceptsAlreadyRunning(t *testing.T) {
	pids := []int{101, 202}

	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		pids,
		false,
		func(context.Context, int) error {
			return nil
		},
		func(_ context.Context, pid int) error {
			return fmt.Errorf("unlock error %d", pid)
		},
		func(_ context.Context, pid int) (string, error) {
			if pid == 101 {
				return "running", nil
			}
			return "checkpointed", nil
		},
		logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected unlock error for non-running process")
	}
	if strings.Contains(err.Error(), "pid 101") {
		t.Fatalf("already-running process returned an error: %v", err)
	}
	if !strings.Contains(err.Error(), "pid 202") {
		t.Fatalf("missing PID attribution: %v", err)
	}
}

func TestHasJobFile(t *testing.T) {
	tests := []struct {
		name           string
		entrypointArgs []string
		want           bool
		reason         string
	}{
		{
			name:           "launch-job wrapped entrypoint",
			entrypointArgs: []string{"cuda-checkpoint", "--launch-job", "python3"},
			want:           true,
			reason:         "is wrapped with cuda-checkpoint --launch-job",
		},
		{
			name:           "launch-job wrapped entrypoint with absolute path",
			entrypointArgs: []string{"/usr/local/bin/cuda-checkpoint", "--launch-job", "python3"},
			want:           true,
			reason:         "is wrapped with cuda-checkpoint --launch-job",
		},
		{
			name:   "nil entrypoint args",
			want:   true,
			reason: "unknown",
		},
		{
			name:           "empty entrypoint args",
			entrypointArgs: []string{},
			want:           true,
			reason:         "unknown",
		},
		{
			name:           "ordinary entrypoint",
			entrypointArgs: []string{"python3", "-m", "dynamo.vllm"},
			reason:         "is not wrapped with cuda-checkpoint --launch-job",
		},
		{
			name:           "launch-job flag without cuda-checkpoint entrypoint",
			entrypointArgs: []string{"python3", "--launch-job"},
			reason:         "is not wrapped with cuda-checkpoint --launch-job",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, reason := HasJobFile(tc.entrypointArgs)
			if got != tc.want {
				t.Fatalf("HasJobFile() = %t, want %t (reason %q)", got, tc.want, reason)
			}
			if !strings.Contains(reason, tc.reason) {
				t.Fatalf("reason %q does not contain %q", reason, tc.reason)
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
