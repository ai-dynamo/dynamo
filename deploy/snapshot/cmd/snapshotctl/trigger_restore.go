package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

type triggerRestoreOptions struct {
	PodName     string
	Namespace   string
	KubeContext string
}

func runTriggerRestore(args []string) error {
	flags := flag.NewFlagSet("trigger-restore", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)

	podName := flags.String("pod", "", "Restore target pod name")
	namespace := flags.String("namespace", "", "Namespace override; defaults to the current kube context namespace")
	kubeContext := flags.String("kube-context", "", "Kubernetes context override")

	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(flags.Args()) != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	if strings.TrimSpace(*podName) == "" {
		return fmt.Errorf("--pod is required")
	}

	result, err := runTriggerRestoreFlow(context.Background(), triggerRestoreOptions{
		PodName:     *podName,
		Namespace:   *namespace,
		KubeContext: *kubeContext,
	})
	if err != nil {
		return err
	}

	snapshotctlLog.Info("Restore trigger submitted", "pod", result.RestorePod, "trigger", result.RestoreTrigger)
	fmt.Printf("status=triggered\n")
	fmt.Printf("namespace=%s\n", result.Namespace)
	fmt.Printf("restore_pod=%s\n", result.RestorePod)
	fmt.Printf("restore_trigger=%s\n", result.RestoreTrigger)
	return nil
}

func runTriggerRestoreFlow(ctx context.Context, opts triggerRestoreOptions) (*result, error) {
	clientset, currentNamespace, err := loadClientset(opts.KubeContext)
	if err != nil {
		return nil, err
	}
	namespace := currentNamespace
	if namespace == "" {
		namespace = corev1.NamespaceDefault
	}
	if strings.TrimSpace(opts.Namespace) != "" {
		namespace = strings.TrimSpace(opts.Namespace)
	}

	podName := strings.TrimSpace(opts.PodName)
	pod, err := clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get restore target pod %s/%s: %w", namespace, podName, err)
	}
	if pod.Labels[snapshotprotocol.RestoreTargetLabel] != "true" {
		return nil, fmt.Errorf("pod %s/%s is not marked as a restore target", namespace, podName)
	}
	if strings.TrimSpace(pod.Annotations[snapshotprotocol.RestoreModeAnnotation]) != snapshotprotocol.RestoreModeManual {
		return nil, fmt.Errorf("pod %s/%s is not configured for manual restore", namespace, podName)
	}

	trigger := fmt.Sprintf("%d", time.Now().UTC().UnixNano())
	patch, err := json.Marshal(map[string]any{
		"metadata": map[string]any{
			"annotations": map[string]string{
				snapshotprotocol.RestoreTriggerAnnotation: trigger,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("encode restore trigger patch: %w", err)
	}
	if _, err := clientset.CoreV1().Pods(namespace).Patch(ctx, podName, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil {
		return nil, fmt.Errorf("patch restore trigger on pod %s/%s: %w", namespace, podName, err)
	}

	return &result{
		Name:           podName,
		Namespace:      namespace,
		RestorePod:     podName,
		RestoreTrigger: trigger,
		Status:         "triggered",
	}, nil
}
