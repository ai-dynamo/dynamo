package cuda

import (
	"context"
	"testing"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestDeviceUUIDFromAttributes(t *testing.T) {
	uuidVal := "GPU-f8ddcf75-4014-85da-28da-9dc4de19d997"
	tests := []struct {
		name  string
		attrs map[resourcev1.QualifiedName]resourcev1.DeviceAttribute
		want  string
	}{
		{
			name:  "nil map",
			attrs: nil,
			want:  "",
		},
		{
			name:  "empty map",
			attrs: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{},
			want:  "",
		},
		{
			name: "uuid present",
			attrs: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
				resourcev1.QualifiedName("uuid"): {StringValue: &uuidVal},
			},
			want: uuidVal,
		},
		{
			name: "uuid missing, other attr present",
			attrs: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
				resourcev1.QualifiedName("productName"): {StringValue: ptr("NVIDIA A100")},
			},
			want: "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := deviceUUIDFromAttributes(tc.attrs)
			if got != tc.want {
				t.Errorf("deviceUUIDFromAttributes() = %q, want %q", got, tc.want)
			}
		})
	}
}

func ptr(s string) *string { return &s }

func TestGetGPUUUIDsViaDRAAPI(t *testing.T) {
	ctx := context.Background()
	log := logr.Discard()

	t.Run("nil clientset returns nil without error", func(t *testing.T) {
		got, usesDRAGPU, err := GetGPUUUIDsViaDRAAPI(ctx, nil, "pod", "ns", log)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if usesDRAGPU {
			t.Fatal("expected usesDRAGPU to be false")
		}
		if got != nil {
			t.Errorf("got %v, want nil", got)
		}
	})

	t.Run("empty pod name returns nil", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		got, usesDRAGPU, err := GetGPUUUIDsViaDRAAPI(ctx, client, "", "ns", log)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if usesDRAGPU {
			t.Fatal("expected usesDRAGPU to be false")
		}
		if got != nil {
			t.Errorf("got %v, want nil", got)
		}
	})

	t.Run("pod not found returns error", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		_, _, err := GetGPUUUIDsViaDRAAPI(ctx, client, "missing", "default", log)
		if err == nil {
			t.Fatal("expected error when pod not found")
		}
	})

	t.Run("pod with DRA claims resolves UUIDs", func(t *testing.T) {
		nodeName := "node-1"
		poolName := "pool-node-1"
		claimName := "gpu-claim"
		namespace := "default"
		podName := "test-pod"
		uuid1 := "GPU-aaaaaaaa-1111-2222-3333-444444444444"
		uuid2 := "GPU-bbbbbbbb-5555-6666-7777-888888888888"

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
							{Driver: nvidiaGPUDRADriver, Pool: poolName, Device: "gpu-1", Request: "gpu"},
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
							resourcev1.QualifiedName("uuid"): {StringValue: &uuid1},
						},
					},
					{
						Name: "gpu-1",
						Attributes: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
							resourcev1.QualifiedName("uuid"): {StringValue: &uuid2},
						},
					},
				},
			},
		}

		client := fake.NewSimpleClientset(pod, claim, slice)
		got, usesDRAGPU, err := GetGPUUUIDsViaDRAAPI(ctx, client, podName, namespace, log)
		if err != nil {
			t.Fatalf("GetGPUUUIDsViaDRAAPI: %v", err)
		}
		if !usesDRAGPU {
			t.Fatal("expected usesDRAGPU to be true")
		}
		want := []string{uuid1, uuid2}
		if len(got) != len(want) {
			t.Fatalf("got %v (len %d), want %v (len %d)", got, len(got), want, len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("got[%d] = %q, want %q", i, got[i], want[i])
			}
		}
	})

	t.Run("pod with template-backed DRA claims resolves UUIDs via pod status", func(t *testing.T) {
		nodeName := "node-1"
		poolName := "pool-node-1"
		namespace := "default"
		podName := "test-pod"
		generatedClaimName := "generated-gpu-claim"
		uuid1 := "GPU-cccccccc-1111-2222-3333-444444444444"

		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: namespace},
			Spec: corev1.PodSpec{
				NodeName: nodeName,
				ResourceClaims: []corev1.PodResourceClaim{
					{
						Name: "gpu",
					},
				},
			},
			Status: corev1.PodStatus{
				ResourceClaimStatuses: []corev1.PodResourceClaimStatus{
					{
						Name:              "gpu",
						ResourceClaimName: ptr(generatedClaimName),
					},
				},
			},
		}
		claim := &resourcev1.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{Name: generatedClaimName, Namespace: namespace},
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
							resourcev1.QualifiedName("uuid"): {StringValue: &uuid1},
						},
					},
				},
			},
		}

		client := fake.NewSimpleClientset(pod, claim, slice)
		got, usesDRAGPU, err := GetGPUUUIDsViaDRAAPI(ctx, client, podName, namespace, log)
		if err != nil {
			t.Fatalf("GetGPUUUIDsViaDRAAPI: %v", err)
		}
		if !usesDRAGPU {
			t.Fatal("expected usesDRAGPU to be true")
		}
		want := []string{uuid1}
		if len(got) != len(want) {
			t.Fatalf("got %v (len %d), want %v (len %d)", got, len(got), want, len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("got[%d] = %q, want %q", i, got[i], want[i])
			}
		}
	})

	t.Run("pod with unresolved resource claim returns nil", func(t *testing.T) {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "default"},
			Spec: corev1.PodSpec{
				NodeName: "node-1",
				ResourceClaims: []corev1.PodResourceClaim{
					{
						Name: "gpu",
					},
				},
			},
		}

		client := fake.NewSimpleClientset(pod)
		got, usesDRAGPU, err := GetGPUUUIDsViaDRAAPI(ctx, client, "pod", "default", log)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if usesDRAGPU {
			t.Fatal("expected usesDRAGPU to be false")
		}
		if got != nil {
			t.Errorf("got %v, want nil", got)
		}
	})

	t.Run("pod with direct and template-backed claims resolves UUIDs from both", func(t *testing.T) {
		nodeName := "node-1"
		poolName := "pool-node-1"
		namespace := "default"
		podName := "test-pod"
		directClaimName := "direct-gpu-claim"
		generatedClaimName := "generated-gpu-claim"
		uuid1 := "GPU-dddddddd-1111-2222-3333-444444444444"
		uuid2 := "GPU-eeeeeeee-5555-6666-7777-888888888888"

		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: namespace},
			Spec: corev1.PodSpec{
				NodeName: nodeName,
				ResourceClaims: []corev1.PodResourceClaim{
					{
						Name:              "gpu-direct",
						ResourceClaimName: ptr(directClaimName),
					},
					{
						Name: "gpu-template",
					},
				},
			},
			Status: corev1.PodStatus{
				ResourceClaimStatuses: []corev1.PodResourceClaimStatus{
					{
						Name:              "gpu-template",
						ResourceClaimName: ptr(generatedClaimName),
					},
				},
			},
		}
		directClaim := &resourcev1.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{Name: directClaimName, Namespace: namespace},
			Status: resourcev1.ResourceClaimStatus{
				Allocation: &resourcev1.AllocationResult{
					Devices: resourcev1.DeviceAllocationResult{
						Results: []resourcev1.DeviceRequestAllocationResult{
							{Driver: nvidiaGPUDRADriver, Pool: poolName, Device: "gpu-0", Request: "gpu-direct"},
						},
					},
				},
			},
		}
		generatedClaim := &resourcev1.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{Name: generatedClaimName, Namespace: namespace},
			Status: resourcev1.ResourceClaimStatus{
				Allocation: &resourcev1.AllocationResult{
					Devices: resourcev1.DeviceAllocationResult{
						Results: []resourcev1.DeviceRequestAllocationResult{
							{Driver: nvidiaGPUDRADriver, Pool: poolName, Device: "gpu-1", Request: "gpu-template"},
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
							resourcev1.QualifiedName("uuid"): {StringValue: &uuid1},
						},
					},
					{
						Name: "gpu-1",
						Attributes: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
							resourcev1.QualifiedName("uuid"): {StringValue: &uuid2},
						},
					},
				},
			},
		}

		client := fake.NewSimpleClientset(pod, directClaim, generatedClaim, slice)
		got, usesDRAGPU, err := GetGPUUUIDsViaDRAAPI(ctx, client, podName, namespace, log)
		if err != nil {
			t.Fatalf("GetGPUUUIDsViaDRAAPI: %v", err)
		}
		if !usesDRAGPU {
			t.Fatal("expected usesDRAGPU to be true")
		}
		want := []string{uuid1, uuid2}
		if len(got) != len(want) {
			t.Fatalf("got %v (len %d), want %v (len %d)", got, len(got), want, len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("got[%d] = %q, want %q", i, got[i], want[i])
			}
		}
	})

	t.Run("pod with no resource claims returns nil", func(t *testing.T) {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "default"},
			Spec:       corev1.PodSpec{NodeName: "node-1"},
		}
		client := fake.NewSimpleClientset(pod)
		got, usesDRAGPU, err := GetGPUUUIDsViaDRAAPI(ctx, client, "pod", "default", log)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if usesDRAGPU {
			t.Fatal("expected usesDRAGPU to be false")
		}
		if got != nil {
			t.Errorf("got %v, want nil", got)
		}
	})
}
