package cuda

import (
	"context"
	"errors"
	"fmt"
	"net"
	"path/filepath"
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

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestCustomStorageProcessDir(t *testing.T) {
	got := customStorageProcessDir("/checkpoints/tmp/staged", 7)
	want := "/checkpoints/tmp/staged/cuda-custom-storage/process-0007"
	if got != want {
		t.Fatalf("customStorageProcessDir() = %q, want %q", got, want)
	}
}

func TestParseCustomStorageTelemetry(t *testing.T) {
	tests := []struct {
		name         string
		output       string
		processWall  time.Duration
		wantStatus   string
		wantDuration time.Duration
	}{
		{
			name:         "valid payload",
			output:       "diagnostic\n" + `{"event":"cuda_custom_storage_transfer","helper_main_to_telemetry_seconds":1.25}`,
			processWall:  2 * time.Second,
			wantStatus:   "valid",
			wantDuration: 1250 * time.Millisecond,
		},
		{
			name:        "absent event",
			output:      `{"event":"other","helper_main_to_telemetry_seconds":1}`,
			processWall: 2 * time.Second,
			wantStatus:  "event-absent",
		},
		{
			name:        "malformed JSON",
			output:      `{"event":"cuda_custom_storage_transfer",`,
			processWall: 2 * time.Second,
			wantStatus:  "malformed-json",
		},
		{
			name:        "missing field",
			output:      `{"event":"cuda_custom_storage_transfer"}`,
			processWall: 2 * time.Second,
			wantStatus:  "missing-duration",
		},
		{
			name:        "negative",
			output:      `{"event":"cuda_custom_storage_transfer","helper_main_to_telemetry_seconds":-1}`,
			processWall: 2 * time.Second,
			wantStatus:  "invalid-duration",
		},
		{
			name:        "nonfinite extreme",
			output:      `{"event":"cuda_custom_storage_transfer","helper_main_to_telemetry_seconds":1e309}`,
			processWall: 2 * time.Second,
			wantStatus:  "invalid-duration",
		},
		{
			name:        "greater than wall",
			output:      `{"event":"cuda_custom_storage_transfer","helper_main_to_telemetry_seconds":2.1}`,
			processWall: 2 * time.Second,
			wantStatus:  "duration-exceeds-process-wall",
		},
		{
			name:         "rounding tolerance",
			output:       `{"event":"cuda_custom_storage_transfer","helper_main_to_telemetry_seconds":2.000001}`,
			processWall:  2 * time.Second,
			wantStatus:   "valid",
			wantDuration: 2 * time.Second,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := parseCustomStorageTelemetry(test.output, test.processWall)
			if got.status != test.wantStatus {
				t.Fatalf("status = %q, want %q (error: %s)", got.status, test.wantStatus, got.err)
			}
			if got.helperMainDuration != test.wantDuration {
				t.Fatalf("helper main duration = %v, want %v", got.helperMainDuration, test.wantDuration)
			}
		})
	}
}

func TestCustomStorageSuccessLogValues(t *testing.T) {
	toMap := func(values []any) map[string]any {
		result := make(map[string]any, len(values)/2)
		for index := 0; index < len(values); index += 2 {
			result[values[index].(string)] = values[index+1]
		}
		return result
	}

	processWall := 2 * time.Second
	invalid := toMap(customStorageSuccessLogValues(
		42,
		actionRestore,
		processWall,
		"output",
		parseCustomStorageTelemetry(`{"event":"cuda_custom_storage_transfer"}`, processWall),
	))
	if invalid["duration"] != processWall || invalid["helper_process_wall_duration"] != processWall {
		t.Fatalf("invalid telemetry log fields lost process wall duration: %#v", invalid)
	}
	if _, found := invalid["helper_main_to_telemetry_duration"]; found {
		t.Fatalf("invalid telemetry fabricated helper duration: %#v", invalid)
	}
	if _, found := invalid["helper_process_overhead_duration"]; found {
		t.Fatalf("invalid telemetry fabricated process overhead: %#v", invalid)
	}

	valid := toMap(customStorageSuccessLogValues(
		42,
		actionRestore,
		processWall,
		"output",
		parseCustomStorageTelemetry(
			`{"event":"cuda_custom_storage_transfer","helper_main_to_telemetry_seconds":1.25}`,
			processWall,
		),
	))
	if valid["helper_main_to_telemetry_duration"] != 1250*time.Millisecond ||
		valid["helper_process_overhead_duration"] != 750*time.Millisecond {
		t.Fatalf("valid telemetry derived durations are wrong: %#v", valid)
	}
}

type helperActionCall struct {
	pid         int
	action      string
	deviceMap   string
	storageMode string
	storageDir  string
}

type recordingHelperActionRunner struct {
	calls      []helperActionCall
	failAction string
	failPID    int
}

func (r *recordingHelperActionRunner) run(
	_ context.Context,
	pid int,
	action,
	deviceMap,
	storageMode,
	storageDir string,
	_ types.CUDATransferSettings,
	_ snapshotruntime.ProcessDetails,
	_ logr.Logger,
) error {
	r.calls = append(r.calls, helperActionCall{
		pid:         pid,
		action:      action,
		deviceMap:   deviceMap,
		storageMode: storageMode,
		storageDir:  storageDir,
	})
	if action == r.failAction && pid == r.failPID {
		return errors.New("injected helper failure")
	}
	return nil
}

func TestRestoreAndUnlockProcessTreeRestoresAllBeforeFirstUnlock(t *testing.T) {
	runner := &recordingHelperActionRunner{}
	pids := []int{41, 7, 99}

	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		pids,
		"GPU-old=GPU-new",
		"posix",
		"/checkpoints/example",
		types.CUDATransferSettings{BufferCount: 1, ChunkBytes: 64 * 1024 * 1024},
		runner,
		logr.Discard(),
	)
	if err != nil {
		t.Fatalf("restoreAndUnlockProcessTree() error = %v", err)
	}
	if len(runner.calls) != 2*len(pids) {
		t.Fatalf("helper calls = %v, want %d restore/unlock calls", runner.calls, 2*len(pids))
	}
	for index, pid := range pids {
		call := runner.calls[index]
		wantDir := filepath.Join(
			"/checkpoints/example",
			"cuda-custom-storage",
			fmt.Sprintf("process-%04d", index),
		)
		if call.action != actionRestore || call.pid != pid || call.storageDir != wantDir {
			t.Fatalf(
				"helper call %d = %+v, want restore for PID %d and directory %q",
				index,
				call,
				pid,
				wantDir,
			)
		}
	}
	if runner.calls[len(pids)].action != actionUnlock {
		t.Fatalf("first post-restore call = %+v, want unlock", runner.calls[len(pids)])
	}
}

func TestLockAndCheckpointProcessTreePreservesPIDOrderInStorageDirectories(t *testing.T) {
	runner := &recordingHelperActionRunner{}
	pids := []int{300, 100, 200}

	_, err := lockAndCheckpointProcessTree(
		context.Background(),
		pids,
		"posix",
		"/checkpoints/example",
		types.CUDATransferSettings{BufferCount: 1, ChunkBytes: 64 * 1024 * 1024},
		runner,
		logr.Discard(),
	)
	if err != nil {
		t.Fatalf("lockAndCheckpointProcessTree() error = %v", err)
	}

	checkpointCalls := runner.calls[len(pids):]
	for index, pid := range pids {
		call := checkpointCalls[index]
		wantDir := filepath.Join(
			"/checkpoints/example",
			"cuda-custom-storage",
			fmt.Sprintf("process-%04d", index),
		)
		if call.action != actionCheckpoint || call.pid != pid || call.storageDir != wantDir {
			t.Fatalf(
				"checkpoint call %d = %+v, want PID %d and directory %q",
				index,
				call,
				pid,
				wantDir,
			)
		}
	}
}

func TestRestoreAndUnlockProcessTreeRestoreFailureSkipsAllUnlocks(t *testing.T) {
	runner := &recordingHelperActionRunner{
		failAction: actionRestore,
		failPID:    22,
	}

	_, err := restoreAndUnlockProcessTree(
		context.Background(),
		[]int{11, 22, 33},
		"",
		"posix",
		"/checkpoints/example",
		types.CUDATransferSettings{BufferCount: 1, ChunkBytes: 64 * 1024 * 1024},
		runner,
		logr.Discard(),
	)
	if err == nil {
		t.Fatal("restoreAndUnlockProcessTree() error = nil, want restore failure")
	}
	for _, call := range runner.calls {
		if call.action == actionUnlock {
			t.Fatalf("unlock called after restore failure: %+v", runner.calls)
		}
	}
}

func TestDaemonUnlockFailureIsReturned(t *testing.T) {
	runner := &recordingHelperActionRunner{
		failAction: actionUnlock,
		failPID:    11,
	}
	settings := types.CUDATransferSettings{
		BufferCount: 1,
		ChunkBytes:  64 * 1024 * 1024,
	}
	_, err := restoreAndUnlockProcessTree(
		context.Background(), []int{11}, "", types.CUDAStorageModePOSIX,
		"/checkpoints/example", settings, runner, logr.Discard(),
	)
	if err == nil {
		t.Fatal("restoreAndUnlockProcessTree() masked daemon unlock failure")
	}
}

func TestValidatedProcessIdentitiesRejectsDuplicateOrIncompleteIdentity(t *testing.T) {
	valid := snapshotruntime.ProcessDetails{
		OutermostPID:   42,
		InnermostPID:   7,
		StartTimeTicks: 123,
		Cgroup:         "0::/kubepods/test\n",
	}
	if _, _, err := validatedProcessIdentities([]snapshotruntime.ProcessDetails{valid, valid}); err == nil {
		t.Fatal("validatedProcessIdentities() accepted duplicate host PID")
	}
	incomplete := valid
	incomplete.Cgroup = ""
	if _, _, err := validatedProcessIdentities([]snapshotruntime.ProcessDetails{incomplete}); err == nil {
		t.Fatal("validatedProcessIdentities() accepted incomplete identity")
	}
}

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
