package cuda

import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	resourceAttributeUUID = "uuid"
)

type allocatedDRADevice struct {
	pool   string
	device string
}

func getAllocatedNVIDIADRADevices(ctx context.Context, clientset kubernetes.Interface, podName, podNamespace string, log logr.Logger) ([]allocatedDRADevice, string, bool, error) {
	if clientset == nil {
		return nil, "", false, nil
	}
	if podName == "" || podNamespace == "" {
		return nil, "", false, nil
	}

	pod, err := clientset.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return nil, "", false, fmt.Errorf("get pod %s/%s: %w", podNamespace, podName, err)
	}
	if len(pod.Spec.ResourceClaims) == 0 {
		return nil, pod.Spec.NodeName, false, nil
	}
	if pod.Spec.NodeName == "" {
		log.V(1).Info("pod has no node name, skipping DRA API lookup")
		return nil, "", false, nil
	}

	claimNamesByPodRef := make(map[string]string, len(pod.Spec.ResourceClaims))
	for _, ref := range pod.Spec.ResourceClaims {
		if ref.ResourceClaimName != nil && *ref.ResourceClaimName != "" {
			claimNamesByPodRef[ref.Name] = *ref.ResourceClaimName
		}
	}
	for _, status := range pod.Status.ResourceClaimStatuses {
		if status.ResourceClaimName == nil || *status.ResourceClaimName == "" {
			continue
		}
		if _, exists := claimNamesByPodRef[status.Name]; !exists {
			claimNamesByPodRef[status.Name] = *status.ResourceClaimName
		}
	}

	var allocated []allocatedDRADevice
	usesDRAGPU := false
	for _, ref := range pod.Spec.ResourceClaims {
		claimName := claimNamesByPodRef[ref.Name]
		if claimName == "" {
			log.V(1).Info("pod resource claim has no resolved claim name", "pod_claim", ref.Name)
			continue
		}
		claim, err := clientset.ResourceV1().ResourceClaims(podNamespace).Get(ctx, claimName, metav1.GetOptions{})
		if err != nil {
			return nil, pod.Spec.NodeName, usesDRAGPU, fmt.Errorf("get resource claim %s/%s: %w", podNamespace, claimName, err)
		}
		if claim.Status.Allocation == nil || len(claim.Status.Allocation.Devices.Results) == 0 {
			continue
		}
		for _, result := range claim.Status.Allocation.Devices.Results {
			if result.Driver != nvidiaGPUDRADriver {
				continue
			}
			usesDRAGPU = true
			allocated = append(allocated, allocatedDRADevice{
				pool:   result.Pool,
				device: result.Device,
			})
		}
	}

	return allocated, pod.Spec.NodeName, usesDRAGPU, nil
}

// GetGPUUUIDsViaDRAAPI resolves GPU UUIDs for a pod by querying the Kubernetes API:
// Pod (resource claim refs) -> ResourceClaim (allocation results) -> ResourceSlice (device attributes).
// It also reports whether the pod is using NVIDIA DRA GPU allocations at all.
func GetGPUUUIDsViaDRAAPI(ctx context.Context, clientset kubernetes.Interface, podName, podNamespace string, log logr.Logger) ([]string, bool, error) {
	allocated, nodeName, usesDRAGPU, err := getAllocatedNVIDIADRADevices(ctx, clientset, podName, podNamespace, log)
	if err != nil {
		return nil, usesDRAGPU, err
	}
	if !usesDRAGPU || len(allocated) == 0 {
		return nil, usesDRAGPU, nil
	}

	slices, err := clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.driver=%s,spec.nodeName=%s", nvidiaGPUDRADriver, nodeName),
	})
	if err != nil {
		return nil, true, fmt.Errorf("list resource slices for node %s: %w", nodeName, err)
	}

	poolDeviceToUUID := make(map[string]map[string]string)
	for i := range slices.Items {
		s := &slices.Items[i]
		poolName := s.Spec.Pool.Name
		if poolDeviceToUUID[poolName] == nil {
			poolDeviceToUUID[poolName] = make(map[string]string)
		}
		for _, dev := range s.Spec.Devices {
			uuid := deviceUUIDFromAttributes(dev.Attributes)
			if uuid != "" && gpuUUIDPattern.MatchString(uuid) {
				poolDeviceToUUID[poolName][dev.Name] = uuid
			}
		}
	}

	var uuids []string
	for _, device := range allocated {
		devMap := poolDeviceToUUID[device.pool]
		if devMap == nil {
			log.V(1).Info("no ResourceSlice found for pool", "pool", device.pool, "device", device.device)
			continue
		}
		uuid, ok := devMap[device.device]
		if !ok || uuid == "" {
			log.V(1).Info("device has no UUID in ResourceSlice", "pool", device.pool, "device", device.device)
			continue
		}
		uuids = append(uuids, uuid)
	}
	if len(uuids) > 0 {
		log.Info("resolved GPU UUIDs via DRA API", "uuids", uuids)
	}
	return uuids, true, nil
}

func deviceUUIDFromAttributes(attrs map[resourcev1.QualifiedName]resourcev1.DeviceAttribute) string {
	a, ok := attrs[resourcev1.QualifiedName(resourceAttributeUUID)]
	if !ok || a.StringValue == nil {
		return ""
	}
	return *a.StringValue
}
