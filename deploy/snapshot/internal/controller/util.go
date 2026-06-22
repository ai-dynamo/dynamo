package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ktypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	checkpointLeaseDuration      = 30 * time.Second
	checkpointLeaseRenewInterval = 10 * time.Second
)

func checkpointLeaseExpired(lease *coordinationv1.Lease, now time.Time) bool {
	if lease == nil || lease.Spec.LeaseDurationSeconds == nil {
		return true
	}
	last := lease.Spec.RenewTime
	if last == nil {
		last = lease.Spec.AcquireTime
	}
	if last == nil {
		return true
	}
	return now.After(last.Time.Add(time.Duration(*lease.Spec.LeaseDurationSeconds) * time.Second))
}

func podFromInformerObj(obj interface{}) (*corev1.Pod, bool) {
	if pod, ok := obj.(*corev1.Pod); ok {
		return pod, true
	}
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		return nil, false
	}
	pod, ok := tombstone.Obj.(*corev1.Pod)
	return pod, ok
}

func isContainerReady(pod *corev1.Pod, containerName string) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == containerName {
			return status.Ready
		}
	}
	return false
}

func annotatePod(ctx context.Context, clientset kubernetes.Interface, log logr.Logger, pod *corev1.Pod, annotations map[string]string) error {
	patchBytes, err := json.Marshal(map[string]any{
		"metadata": map[string]any{
			"annotations": annotations,
		},
	})
	if err != nil {
		return fmt.Errorf("failed to build annotation patch payload: %w", err)
	}

	_, err = clientset.CoreV1().Pods(pod.Namespace).Patch(
		ctx, pod.Name, ktypes.MergePatchType, patchBytes, metav1.PatchOptions{},
	)
	if err != nil {
		log.Error(err, "Failed to annotate pod",
			"pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name),
			"annotations", annotations,
		)
	}
	return err
}

// acquireLease acquires or renews a checkpoint lease at an arbitrary namespace/name key,
// returning false when another live holder owns it.
func (w *NodeController) acquireLease(ctx context.Context, key client.ObjectKey) (bool, error) {
	now := metav1.NewMicroTime(time.Now())
	leaseDurationSeconds := int32(checkpointLeaseDuration.Seconds())

	leaseClient := w.clientset.CoordinationV1().Leases(key.Namespace)
	existing, err := leaseClient.Get(ctx, key.Name, metav1.GetOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return false, fmt.Errorf("get checkpoint lease %s: %w", key.String(), err)
		}
		lease := &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity:       &w.holderID,
				LeaseDurationSeconds: &leaseDurationSeconds,
				AcquireTime:          &now,
				RenewTime:            &now,
			},
		}
		if _, err := leaseClient.Create(ctx, lease, metav1.CreateOptions{}); err != nil {
			if apierrors.IsAlreadyExists(err) {
				return false, nil
			}
			return false, fmt.Errorf("create checkpoint lease %s: %w", key.String(), err)
		}
		return true, nil
	}

	if !checkpointLeaseExpired(existing, now.Time) &&
		existing.Spec.HolderIdentity != nil &&
		*existing.Spec.HolderIdentity != w.holderID {
		return false, nil
	}
	existing.Spec.HolderIdentity = &w.holderID
	existing.Spec.LeaseDurationSeconds = &leaseDurationSeconds
	if existing.Spec.AcquireTime == nil || checkpointLeaseExpired(existing, now.Time) {
		existing.Spec.AcquireTime = &now
	}
	existing.Spec.RenewTime = &now
	if _, err := leaseClient.Update(ctx, existing, metav1.UpdateOptions{}); err != nil {
		if apierrors.IsConflict(err) {
			return false, nil
		}
		return false, fmt.Errorf("update checkpoint lease %s: %w", key.String(), err)
	}
	return true, nil
}

// renewLease periodically renews the lease until ctx is cancelled.
func (w *NodeController) renewLease(ctx context.Context, key client.ObjectKey) {
	ticker := time.NewTicker(checkpointLeaseRenewInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := w.renewLeaseOnce(ctx, key); err != nil {
				log.FromContext(ctx).Error(err, "Failed to renew checkpoint lease", "lease", key.String())
				return
			}
		}
	}
}

// renewLeaseOnce bumps the lease renew time, failing if this holder no longer owns it.
func (w *NodeController) renewLeaseOnce(ctx context.Context, key client.ObjectKey) error {
	leaseClient := w.clientset.CoordinationV1().Leases(key.Namespace)
	lease, err := leaseClient.Get(ctx, key.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get checkpoint lease %s for renewal: %w", key.String(), err)
	}
	if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != w.holderID {
		return fmt.Errorf("checkpoint lease %s is no longer held by %q", key.String(), w.holderID)
	}
	now := metav1.NewMicroTime(time.Now())
	leaseDurationSeconds := int32(checkpointLeaseDuration.Seconds())
	lease.Spec.LeaseDurationSeconds = &leaseDurationSeconds
	lease.Spec.RenewTime = &now
	if _, err := leaseClient.Update(ctx, lease, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("renew checkpoint lease %s: %w", key.String(), err)
	}
	return nil
}

// releaseLease deletes the lease if this holder owns it.
func (w *NodeController) releaseLease(ctx context.Context, key client.ObjectKey) error {
	leaseClient := w.clientset.CoordinationV1().Leases(key.Namespace)
	lease, err := leaseClient.Get(ctx, key.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("get checkpoint lease %s for release: %w", key.String(), err)
	}
	if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != w.holderID {
		return nil
	}
	if err := leaseClient.Delete(ctx, key.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("delete checkpoint lease %s: %w", key.String(), err)
	}
	return nil
}
func emitPodEvent(ctx context.Context, clientset kubernetes.Interface, log logr.Logger, pod *corev1.Pod, component, eventType, reason, message string) {
	event := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("%s-", pod.Name),
			Namespace:    pod.Namespace,
		},
		InvolvedObject: corev1.ObjectReference{
			Kind:       "Pod",
			Namespace:  pod.Namespace,
			Name:       pod.Name,
			UID:        pod.UID,
			APIVersion: "v1",
		},
		Type:    eventType,
		Reason:  reason,
		Message: message,
		Source: corev1.EventSource{
			Component: component,
		},
		Count:          1,
		FirstTimestamp: metav1.Now(),
		LastTimestamp:  metav1.Now(),
	}

	if _, err := clientset.CoreV1().Events(pod.Namespace).Create(ctx, event, metav1.CreateOptions{}); err != nil {
		log.Error(err, "Failed to create event",
			"pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name),
			"reason", reason,
			"message", message,
		)
	}
}
