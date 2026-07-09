package cuda

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
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

// containerResourceClaimRefs returns the pod-level resource-claim references
// exposed to the named container via container.resources.claims, mapped to
// the set of claim request names the container is restricted to. A nil inner
// set exposes every request in the claim. An empty containerName returns a
// nil map, meaning the pod-wide claim view applies.
func containerResourceClaimRefs(pod *corev1.Pod, containerName string) (map[string]map[string]struct{}, error) {
	if containerName == "" {
		return nil, nil
	}
	for _, containers := range [][]corev1.Container{pod.Spec.Containers, pod.Spec.InitContainers} {
		for i := range containers {
			if containers[i].Name != containerName {
				continue
			}
			refs := make(map[string]map[string]struct{}, len(containers[i].Resources.Claims))
			for _, claimRef := range containers[i].Resources.Claims {
				requests, seen := refs[claimRef.Name]
				if seen && requests == nil {
					// Already exposed without a request restriction.
					continue
				}
				if claimRef.Request == "" {
					refs[claimRef.Name] = nil
					continue
				}
				if requests == nil {
					requests = make(map[string]struct{})
					refs[claimRef.Name] = requests
				}
				requests[claimRef.Request] = struct{}{}
			}
			return refs, nil
		}
	}
	return nil, fmt.Errorf("container %q not found in pod %s/%s spec", containerName, pod.Namespace, pod.Name)
}

// resultMatchesRequests reports whether an allocation result belongs to one of
// the claim request names a container is restricted to. Allocation results
// for subrequests use the "<main request>/<subrequest>" form, so a container
// reference to the main request matches its subrequest results too.
func resultMatchesRequests(resultRequest string, requests map[string]struct{}) bool {
	if requests == nil {
		return true
	}
	if _, ok := requests[resultRequest]; ok {
		return true
	}
	mainRequest, _, found := strings.Cut(resultRequest, "/")
	if !found {
		return false
	}
	_, ok := requests[mainRequest]
	return ok
}

func getAllocatedNVIDIADRADevices(ctx context.Context, clientset kubernetes.Interface, podName, podNamespace, containerName string, log logr.Logger) ([]allocatedDRADevice, string, bool, error) {
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

	containerClaimRefs, err := containerResourceClaimRefs(pod, containerName)
	if err != nil {
		// The pod defines resource claims but the target container cannot be
		// resolved; fail rather than fall back to unverified discovery.
		return nil, pod.Spec.NodeName, true, err
	}
	if containerClaimRefs != nil && len(containerClaimRefs) == 0 {
		// The target container references no resource claims.
		return nil, pod.Spec.NodeName, false, nil
	}
	// Probe state: the target container (or the whole pod, when containerName
	// is empty) references resource claims. Every error past this point
	// returns true so the caller fails closed instead of falling back to
	// unverified discovery.

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
	hasNVIDIADRAAllocation := false
	for _, ref := range pod.Spec.ResourceClaims {
		var containerRequests map[string]struct{}
		if containerClaimRefs != nil {
			requests, exposed := containerClaimRefs[ref.Name]
			if !exposed {
				continue
			}
			containerRequests = requests
		}
		claimName := claimNamesByPodRef[ref.Name]
		if claimName == "" {
			log.V(1).Info("pod resource claim has no resolved claim name", "pod_claim", ref.Name)
			continue
		}
		claim, err := clientset.ResourceV1().ResourceClaims(podNamespace).Get(ctx, claimName, metav1.GetOptions{})
		if err != nil {
			return nil, pod.Spec.NodeName, true, fmt.Errorf("get resource claim %s/%s: %w", podNamespace, claimName, err)
		}
		if claim.Status.Allocation == nil || len(claim.Status.Allocation.Devices.Results) == 0 {
			continue
		}
		for _, result := range claim.Status.Allocation.Devices.Results {
			if result.Driver != nvidiaGPUDRADriver {
				continue
			}
			if !resultMatchesRequests(result.Request, containerRequests) {
				continue
			}
			hasNVIDIADRAAllocation = true
			allocated = append(allocated, allocatedDRADevice{
				pool:   result.Pool,
				device: result.Device,
			})
		}
	}

	return allocated, pod.Spec.NodeName, hasNVIDIADRAAllocation, nil
}

// GetGPUUUIDsViaDRAAPI resolves GPU UUIDs for a pod by querying the Kubernetes API:
// Pod (resource claim refs) -> ResourceClaim (allocation results) -> ResourceSlice (device attributes).
// A non-empty containerName restricts resolution to the claims (and claim
// requests) that container references via resources.claims.
// On success, the returned bool reports whether NVIDIA DRA GPU allocations
// were found for that container. With a non-nil error, it reports whether the
// container was established to reference resource claims, in which case the
// caller must fail instead of falling back to other discovery paths.
func GetGPUUUIDsViaDRAAPI(ctx context.Context, clientset kubernetes.Interface, podName, podNamespace, containerName string, log logr.Logger) ([]string, bool, error) {
	allocated, nodeName, hasNVIDIADRAAllocation, err := getAllocatedNVIDIADRADevices(ctx, clientset, podName, podNamespace, containerName, log)
	if err != nil {
		return nil, hasNVIDIADRAAllocation, err
	}
	if !hasNVIDIADRAAllocation || len(allocated) == 0 {
		return nil, hasNVIDIADRAAllocation, nil
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
