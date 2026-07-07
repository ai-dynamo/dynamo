package cuda

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
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

func TestRestoreAndUnlockProcessTreePreservesPhaseBarrier(t *testing.T) {
	pids := []int{101, 202, 303}
	procRoot := fakeProcRoot(t, map[int]string{
		101: "PATH=/bin",
		202: "PATH=/bin",
		303: "PATH=/bin",
	})
	restoreStarted := make(chan int, len(pids))
	releaseRestore := make(chan struct{})
	unlockStarted := make(chan int, len(pids))
	releaseUnlock := make(chan struct{})
	type result struct {
		timings RestorePhaseTimings
		err     error
	}
	resultCh := make(chan result, 1)

	go func() {
		timings, err := restoreAndUnlockProcessTree(
			context.Background(),
			pids,
			procRoot,
			func(ctx context.Context, pid int) error {
				restoreStarted <- pid
				select {
				case <-releaseRestore:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			},
			func(ctx context.Context, pid int) error {
				unlockStarted <- pid
				select {
				case <-releaseUnlock:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			},
			func(context.Context, int) (string, error) {
				return "", errors.New("unexpected state check")
			},
			logr.Discard(),
		)
		resultCh <- result{timings: timings, err: err}
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
			t.Fatal("unlock operations did not all start concurrently")
		}
	}
	close(releaseUnlock)

	select {
	case got := <-resultCh:
		if got.err != nil {
			t.Fatalf("restoreAndUnlockProcessTree: %v", got.err)
		}
		if got.timings.RestoreDuration <= 0 ||
			got.timings.UnlockDuration <= 0 ||
			got.timings.TotalDuration <= 0 {
			t.Fatalf("expected positive phase timings, got %+v", got.timings)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("restoreAndUnlockProcessTree did not complete")
	}
}

func TestRestoreAndUnlockProcessTreePreservesSerialPhaseBarrier(t *testing.T) {
	pids := []int{303, 101, 202}
	procRoot := fakeProcRoot(t, map[int]string{
		303: "CUDA_CHECKPOINT_JOB_FILE=/tmp/job",
		101: "PATH=/bin",
		202: "PATH=/bin",
	})
	var calls []string

	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		pids,
		procRoot,
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
	procRoot := fakeProcRoot(t, map[int]string{
		303: "PATH=/bin",
		101: "PATH=/bin",
		202: "PATH=/bin",
	})
	var completed atomic.Int32
	var unlockCalls atomic.Int32

	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		pids,
		procRoot,
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
	procRoot := fakeProcRoot(t, map[int]string{
		101: "PATH=/bin",
		202: "PATH=/bin",
	})
	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		pids,
		procRoot,
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

func TestCanRestoreConcurrently(t *testing.T) {
	tests := []struct {
		name     string
		environs map[int]string
		pids     []int
		mutate   func(t *testing.T, procRoot string)
		want     bool
		reason   string
	}{
		{
			name: "all independent",
			environs: map[int]string{
				101: "PATH=/bin",
				202: "OTHER=value",
			},
			pids:   []int{101, 202},
			want:   true,
			reason: "independent",
		},
		{
			name: "unrelated bare entry",
			environs: map[int]string{
				101: "FLASHINFER_PYNCCL_NO_MODEL_ALLGATHER_GUARD_STANDBY_SKIP",
			},
			pids:   []int{101},
			want:   true,
			reason: "independent",
		},
		{
			name: "bare job file",
			environs: map[int]string{
				101: "CUDA_CHECKPOINT_JOB_FILE",
			},
			pids:   []int{101},
			reason: "pid 101 has CUDA_CHECKPOINT_JOB_FILE set",
		},
		{
			name: "shared job file",
			environs: map[int]string{
				101: "CUDA_CHECKPOINT_JOB_FILE=/tmp/job",
				202: "PATH=/bin",
			},
			pids:   []int{101, 202},
			reason: "pid 101 has CUDA_CHECKPOINT_JOB_FILE set",
		},
		{
			name: "mixed processes",
			environs: map[int]string{
				101: "PATH=/bin",
				202: "CUDA_CHECKPOINT_JOB_FILE=/tmp/job",
			},
			pids:   []int{101, 202},
			reason: "pid 202 has CUDA_CHECKPOINT_JOB_FILE set",
		},
		{
			name: "read failure",
			environs: map[int]string{
				101: "PATH=/bin",
			},
			pids:   []int{101, 202},
			reason: "could not inspect target pid 202 environment",
		},
		{
			name: "malformed environment",
			environs: map[int]string{
				101: "PATH=/bin",
			},
			pids: []int{101},
			mutate: func(t *testing.T, procRoot string) {
				t.Helper()
				path := filepath.Join(procRoot, "101", "environ")
				if err := os.WriteFile(path, []byte("PATH=/bin"), 0o600); err != nil {
					t.Fatalf("write malformed environment: %v", err)
				}
			},
			reason: "could not parse target pid 101 environment",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			procRoot := fakeProcRoot(t, tc.environs)
			if tc.mutate != nil {
				tc.mutate(t, procRoot)
			}

			got, reason := canRestoreConcurrently(procRoot, tc.pids)
			if got != tc.want {
				t.Fatalf("canRestoreConcurrently() = %t, want %t", got, tc.want)
			}
			if !strings.Contains(reason, tc.reason) {
				t.Fatalf("reason %q does not contain %q", reason, tc.reason)
			}
		})
	}
}

func fakeProcRoot(t *testing.T, environs map[int]string) string {
	t.Helper()
	procRoot := t.TempDir()
	for pid, environ := range environs {
		dir := filepath.Join(procRoot, strconv.Itoa(pid))
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatalf("create fake proc pid directory: %v", err)
		}
		data := []byte{}
		if environ != "" {
			data = append([]byte(environ), 0)
		}
		if err := os.WriteFile(filepath.Join(dir, "environ"), data, 0o600); err != nil {
			t.Fatalf("write fake process environment: %v", err)
		}
	}
	return procRoot
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

func TestDiscoverGPUUUIDsPrefersDRAForDRAPod(t *testing.T) {
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
	uuid := "GPU-ffffffff-1111-2222-3333-444444444444"

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: namespace},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
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
						resourcev1.QualifiedName("uuid"): {StringValue: &uuid},
					},
				},
			},
		},
	}

	client := fake.NewSimpleClientset(pod, claim, slice)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	got, err := DiscoverGPUUUIDs(
		ctx,
		client,
		podName,
		namespace,
		"main",
		"/proc",
		123,
		logr.Discard(),
	)
	if err != nil {
		t.Fatalf("DiscoverGPUUUIDs: %v", err)
	}
	if len(got) != 1 || got[0] != uuid {
		t.Fatalf("got %v, want [%s]", got, uuid)
	}
}
