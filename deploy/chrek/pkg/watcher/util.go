package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ktypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

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
