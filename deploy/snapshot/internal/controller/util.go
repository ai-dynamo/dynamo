package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ktypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
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

func resolveMainContainerName(pod *corev1.Pod) string {
	containerName := ""
	for _, c := range pod.Spec.Containers {
		if c.Name == "main" {
			return c.Name
		}
		if containerName == "" {
			containerName = c.Name
		}
	}
	return containerName
}

func isPodReady(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
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

func getCheckpointJob(ctx context.Context, clientset kubernetes.Interface, pod *corev1.Pod) (*batchv1.Job, error) {
	jobName := pod.Labels["batch.kubernetes.io/job-name"]
	if jobName == "" {
		return nil, fmt.Errorf("pod %s/%s has no batch.kubernetes.io/job-name label", pod.Namespace, pod.Name)
	}

	job, err := clientset.BatchV1().Jobs(pod.Namespace).Get(ctx, jobName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get checkpoint job %s/%s: %w", pod.Namespace, jobName, err)
	}
	return job, nil
}

func acquireCheckpointLease(ctx context.Context, clientset kubernetes.Interface, log logr.Logger, job *batchv1.Job, holderIdentity string) (bool, error) {
	leaseName := job.Name
	now := metav1.NewMicroTime(time.Now())
	leaseDurationSeconds := int32(checkpointLeaseDuration.Seconds())

	leaseClient := clientset.CoordinationV1().Leases(job.Namespace)
	existingLease, err := leaseClient.Get(ctx, leaseName, metav1.GetOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return false, fmt.Errorf("failed to get checkpoint lease %s/%s: %w", job.Namespace, leaseName, err)
		}

		lease := &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      leaseName,
				Namespace: job.Namespace,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity:       &holderIdentity,
				LeaseDurationSeconds: &leaseDurationSeconds,
				AcquireTime:          &now,
				RenewTime:            &now,
			},
		}

		if _, err := leaseClient.Create(ctx, lease, metav1.CreateOptions{}); err != nil {
			if apierrors.IsAlreadyExists(err) {
				return false, nil
			}
			return false, fmt.Errorf("failed to create checkpoint lease %s/%s: %w", job.Namespace, leaseName, err)
		}
		return true, nil
	}

	if !checkpointLeaseExpired(existingLease, now.Time) &&
		existingLease.Spec.HolderIdentity != nil &&
		*existingLease.Spec.HolderIdentity != holderIdentity {
		return false, nil
	}

	existingLease.Spec.HolderIdentity = &holderIdentity
	existingLease.Spec.LeaseDurationSeconds = &leaseDurationSeconds
	if existingLease.Spec.AcquireTime == nil || checkpointLeaseExpired(existingLease, now.Time) {
		existingLease.Spec.AcquireTime = &now
	}
	existingLease.Spec.RenewTime = &now

	if _, err := leaseClient.Update(ctx, existingLease, metav1.UpdateOptions{}); err != nil {
		if apierrors.IsConflict(err) {
			log.V(1).Info("Checkpoint lease update conflicted", "lease", fmt.Sprintf("%s/%s", job.Namespace, leaseName))
			return false, nil
		}
		return false, fmt.Errorf("failed to update checkpoint lease %s/%s: %w", job.Namespace, leaseName, err)
	}

	return true, nil
}

func renewCheckpointLease(ctx context.Context, clientset kubernetes.Interface, job *batchv1.Job, holderIdentity string) error {
	leaseName := job.Name
	leaseClient := clientset.CoordinationV1().Leases(job.Namespace)
	lease, err := leaseClient.Get(ctx, leaseName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get checkpoint lease %s/%s for renewal: %w", job.Namespace, leaseName, err)
	}
	if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != holderIdentity {
		return fmt.Errorf("checkpoint lease %s/%s is no longer held by %q", job.Namespace, leaseName, holderIdentity)
	}

	now := metav1.NewMicroTime(time.Now())
	leaseDurationSeconds := int32(checkpointLeaseDuration.Seconds())
	lease.Spec.LeaseDurationSeconds = &leaseDurationSeconds
	lease.Spec.RenewTime = &now

	if _, err := leaseClient.Update(ctx, lease, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("failed to renew checkpoint lease %s/%s: %w", job.Namespace, leaseName, err)
	}
	return nil
}

func releaseCheckpointLease(ctx context.Context, clientset kubernetes.Interface, log logr.Logger, job *batchv1.Job, holderIdentity string) error {
	leaseName := job.Name
	leaseClient := clientset.CoordinationV1().Leases(job.Namespace)
	lease, err := leaseClient.Get(ctx, leaseName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get checkpoint lease %s/%s for release: %w", job.Namespace, leaseName, err)
	}

	if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != holderIdentity {
		log.V(1).Info("Skipping checkpoint lease release because another holder owns it",
			"lease", fmt.Sprintf("%s/%s", job.Namespace, leaseName),
			"holder", holderIdentity,
		)
		return nil
	}

	if err := leaseClient.Delete(ctx, leaseName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to delete checkpoint lease %s/%s: %w", job.Namespace, leaseName, err)
	}
	return nil
}

func (w *NodeController) renewCheckpointLease(ctx context.Context, log logr.Logger, job *batchv1.Job, stopLease context.CancelCauseFunc) {
	ticker := time.NewTicker(checkpointLeaseRenewInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := renewCheckpointLease(ctx, w.clientset, job, w.holderID); err != nil {
				log.Error(err, "Failed to renew checkpoint lease")
				stopLease(fmt.Errorf("checkpoint lease renewal failed: %w", err))
				return
			}
		}
	}
}

func annotateJob(ctx context.Context, clientset kubernetes.Interface, log logr.Logger, job *batchv1.Job, annotations map[string]string) error {
	patchBytes, err := json.Marshal(map[string]any{
		"metadata": map[string]any{
			"annotations": annotations,
		},
	})
	if err != nil {
		return fmt.Errorf("failed to build job annotation patch payload: %w", err)
	}

	_, err = clientset.BatchV1().Jobs(job.Namespace).Patch(
		ctx, job.Name, ktypes.MergePatchType, patchBytes, metav1.PatchOptions{},
	)
	if err != nil {
		log.Error(err, "Failed to annotate checkpoint job",
			"job", fmt.Sprintf("%s/%s", job.Namespace, job.Name),
			"annotations", annotations,
		)
	}
	return err
}

func waitForPodReady(ctx context.Context, clientset kubernetes.Interface, namespace, podName, containerName string) error {
	lastPhase := ""

	for {
		pod, err := clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get pod %s/%s: %w", namespace, podName, err)
		}

		lastPhase = string(pod.Status.Phase)
		for _, condition := range pod.Status.Conditions {
			if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
				return nil
			}
		}

		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Name != containerName {
				continue
			}
			if cs.State.Terminated != nil {
				return fmt.Errorf(
					"pod %s/%s container %s terminated: reason=%s exitCode=%d",
					namespace, podName, containerName,
					cs.State.Terminated.Reason, cs.State.Terminated.ExitCode,
				)
			}
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("pod %s/%s did not become Ready (last phase: %s): %w", namespace, podName, lastPhase, ctx.Err())
		case <-time.After(1 * time.Second):
		}
	}
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
