package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const defaultGeneratedCheckpointIDPrefix = "manual-snapshot"

type checkpointOptions struct {
	ManifestPath                 string
	Namespace                    string
	KubeContext                  string
	CheckpointID                 string
	DisableCudaCheckpointJobFile bool
	Timeout                      time.Duration
}

type result struct {
	Name               string
	Namespace          string
	CheckpointID       string
	CheckpointLocation string
	CheckpointJob      string
	RestorePod         string
	Status             string
}

func runCheckpointFlow(ctx context.Context, opts checkpointOptions) (*result, error) {
	if strings.TrimSpace(opts.ManifestPath) == "" {
		return nil, fmt.Errorf("missing required flags: --manifest")
	}
	if opts.Timeout <= 0 {
		return nil, fmt.Errorf("--timeout must be greater than zero")
	}

	pod, clientset, namespace, storage, err := loadRunContext(ctx, opts.ManifestPath, opts.Namespace, opts.KubeContext)
	if err != nil {
		return nil, err
	}

	checkpointID := strings.TrimSpace(opts.CheckpointID)
	if checkpointID == "" {
		checkpointID = fmt.Sprintf("%s-%d", defaultGeneratedCheckpointIDPrefix, time.Now().UTC().UnixNano())
	}
	resolvedStorage, err := snapshotprotocol.ResolveCheckpointStorage(checkpointID, "", snapshotprotocol.Storage{
		Type:     snapshotprotocol.StorageTypePVC,
		PVCName:  storage.PVCName,
		BasePath: storage.BasePath,
	})
	if err != nil {
		return nil, err
	}

	checkpointJobName := pod.Name + "-checkpoint"
	job, err := snapshotprotocol.NewCheckpointJob(&corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      pod.Labels,
			Annotations: pod.Annotations,
		},
		Spec: *pod.Spec.DeepCopy(),
	}, snapshotprotocol.CheckpointJobOptions{
		Namespace:       namespace,
		CheckpointID:    checkpointID,
		ArtifactVersion: snapshotprotocol.DefaultCheckpointArtifactVersion,
		SeccompProfile:  snapshotprotocol.DefaultSeccompLocalhostProfile,
		Name:            checkpointJobName,
		WrapLaunchJob:   !opts.DisableCudaCheckpointJobFile,
	})
	if err != nil {
		return nil, err
	}
	_, err = clientset.BatchV1().Jobs(namespace).Create(ctx, job, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("checkpoint job %s/%s already exists", namespace, checkpointJobName)
	}
	if err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForCheckpoint(waitCtx, clientset, namespace, checkpointJobName)
	if err != nil {
		return nil, err
	}

	return &result{
		Name:               pod.Name,
		Namespace:          namespace,
		CheckpointID:       checkpointID,
		CheckpointLocation: resolvedStorage.Location,
		CheckpointJob:      checkpointJobName,
		Status:             status,
	}, nil
}

func waitForCheckpoint(ctx context.Context, clientset kubernetes.Interface, namespace string, jobName string) (string, error) {
	var status string
	if err := wait.PollUntilContextCancel(ctx, 2*time.Second, true, func(ctx context.Context) (bool, error) {
		job, err := clientset.BatchV1().Jobs(namespace).Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, fmt.Errorf("get checkpoint job %s/%s: %w", namespace, jobName, err)
		}

		status = strings.TrimSpace(job.Annotations[snapshotprotocol.CheckpointStatusAnnotation])
		if status == snapshotprotocol.CheckpointStatusCompleted {
			return true, nil
		}
		if status == snapshotprotocol.CheckpointStatusFailed {
			return false, fmt.Errorf("checkpoint job %s/%s failed", namespace, jobName)
		}
		if job.Status.Failed > 0 {
			return false, fmt.Errorf("checkpoint job %s/%s failed", namespace, jobName)
		}
		for _, condition := range job.Status.Conditions {
			if condition.Status != corev1.ConditionTrue {
				continue
			}
			if condition.Type == batchv1.JobFailed {
				return false, fmt.Errorf("checkpoint job %s/%s failed: %s", namespace, jobName, strings.TrimSpace(condition.Message))
			}
		}
		return false, nil
	}); err != nil {
		if !wait.Interrupted(err) {
			return "", err
		}

		summaryCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		pods, err := clientset.CoreV1().Pods(namespace).List(summaryCtx, metav1.ListOptions{
			LabelSelector: "batch.kubernetes.io/job-name=" + jobName,
		})
		summary := "no checkpoint pod created yet"
		if err != nil {
			summary = "unable to list checkpoint pod: " + err.Error()
		} else if len(pods.Items) != 0 {
			pod := pods.Items[0]
			parts := []string{
				fmt.Sprintf("job_status=%q", status),
				fmt.Sprintf("pod=%s phase=%s", pod.Name, pod.Status.Phase),
			}
			for _, condition := range pod.Status.Conditions {
				if condition.Status == corev1.ConditionTrue || condition.Status == corev1.ConditionFalse {
					parts = append(parts, fmt.Sprintf("%s=%s", condition.Type, condition.Status))
				}
			}
			for _, status := range pod.Status.ContainerStatuses {
				if status.State.Waiting != nil {
					parts = append(parts, fmt.Sprintf("container=%s waiting=%s", status.Name, status.State.Waiting.Reason))
				}
				if status.State.Terminated != nil {
					parts = append(parts, fmt.Sprintf("container=%s terminated=%s", status.Name, status.State.Terminated.Reason))
				}
			}
			summary = strings.Join(parts, " ")
		}
		return "", fmt.Errorf("checkpoint job %s/%s timed out: %s", namespace, jobName, summary)
	}
	return status, nil
}
