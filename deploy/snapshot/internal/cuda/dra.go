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

// GetGPUUUIDsViaDRAAPI resolves GPU UUIDs for a pod by querying the Kubernetes API:
// Pod (resource claim refs) -> ResourceClaim (allocation results) -> ResourceSlice (device attributes).
// Returns nil without error if the pod has no DRA claims or the driver is not gpu.nvidia.com.
func GetGPUUUIDsViaDRAAPI(ctx context.Context, clientset kubernetes.Interface, podName, podNamespace string, log logr.Logger) ([]string, error) {
	if clientset == nil {
		return nil, nil
	}
	if podName == "" || podNamespace == "" {
		return nil, nil
	}

	pod, err := clientset.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get pod %s/%s: %w", podNamespace, podName, err)
	}
	if len(pod.Spec.ResourceClaims) == 0 {
		return nil, nil
	}
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		log.V(1).Info("pod has no node name, skipping DRA API lookup")
		return nil, nil
	}

	var allocated []struct {
		driver string
		pool   string
		device string
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
	for _, ref := range pod.Spec.ResourceClaims {
		claimName := claimNamesByPodRef[ref.Name]
		if claimName == "" {
			log.V(1).Info("pod resource claim has no resolved claim name", "pod_claim", ref.Name)
			continue
		}
		claim, err := clientset.ResourceV1().ResourceClaims(podNamespace).Get(ctx, claimName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("get resource claim %s/%s: %w", podNamespace, claimName, err)
		}
		if claim.Status.Allocation == nil || len(claim.Status.Allocation.Devices.Results) == 0 {
			continue
		}
		for _, r := range claim.Status.Allocation.Devices.Results {
			if r.Driver == nvidiaGPUDRADriver {
				allocated = append(allocated, struct {
					driver string
					pool   string
					device string
				}{r.Driver, r.Pool, r.Device})
			}
		}
	}
	if len(allocated) == 0 {
		return nil, nil
	}

	slices, err := clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.driver=%s,spec.nodeName=%s", nvidiaGPUDRADriver, nodeName),
	})
	if err != nil {
		return nil, fmt.Errorf("list resource slices for node %s: %w", nodeName, err)
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
	for _, a := range allocated {
		devMap := poolDeviceToUUID[a.pool]
		if devMap == nil {
			log.V(1).Info("no ResourceSlice found for pool", "pool", a.pool, "device", a.device)
			continue
		}
		uuid, ok := devMap[a.device]
		if !ok || uuid == "" {
			log.V(1).Info("device has no UUID in ResourceSlice", "pool", a.pool, "device", a.device)
			continue
		}
		uuids = append(uuids, uuid)
	}
	if len(uuids) > 0 {
		log.Info("resolved GPU UUIDs via DRA API", "uuids", uuids)
	}
	return uuids, nil
}

func deviceUUIDFromAttributes(attrs map[resourcev1.QualifiedName]resourcev1.DeviceAttribute) string {
	a, ok := attrs[resourcev1.QualifiedName(resourceAttributeUUID)]
	if !ok || a.StringValue == nil {
		return ""
	}
	return *a.StringValue
}
