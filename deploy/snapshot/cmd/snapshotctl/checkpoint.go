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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
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
	err := watchNamedObject(
		ctx,
		jobName,
		&batchv1.Job{},
		func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
			return clientset.BatchV1().Jobs(namespace).List(ctx, options)
		},
		func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
			return clientset.BatchV1().Jobs(namespace).Watch(ctx, options)
		},
		func(event watch.Event) (bool, error) {
			if event.Type == watch.Error {
				return false, apierrors.FromObject(event.Object)
			}

			job, ok := event.Object.(*batchv1.Job)
			if !ok {
				return false, fmt.Errorf("unexpected checkpoint watch object %T", event.Object)
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
		},
	)
	if err != nil {
		if err == context.DeadlineExceeded {
			return "", fmt.Errorf("checkpoint job %s/%s timed out: status=%q", namespace, jobName, status)
		}
		return "", err
	}
	return status, nil
}
