package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	snapshotv1alpha1 "github.com/ai-dynamo/dynamo/deploy/snapshot/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/tools/clientcmd"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

const defaultGeneratedSnapshotIDPrefix = "manual-snapshot"

var dnsLabelPattern = regexp.MustCompile(`[^a-z0-9-]+`)

type podManifest struct {
	Name      string
	Namespace string
	Template  corev1.PodTemplateSpec
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage()
		return nil
	}

	switch args[0] {
	case "checkpoint":
		return runCheckpoint(args[1:])
	case "restore":
		return runRestore(args[1:])
	case "help", "-h", "--help":
		printUsage()
		return nil
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func runCheckpoint(args []string) error {
	flags := flag.NewFlagSet("checkpoint", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)

	manifestPath := flags.String("manifest", "", "Path to a worker Pod manifest")
	namespaceOverride := flags.String("namespace", "", "Namespace override; defaults to the manifest namespace or current kube context namespace")
	snapshotID := flags.String("checkpoint-hash", "", "Snapshot identifier; defaults to a generated value")
	timeout := flags.Duration("timeout", 45*time.Minute, "Maximum time to wait for checkpoint completion")

	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(flags.Args()) != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	if strings.TrimSpace(*manifestPath) == "" {
		return fmt.Errorf("missing required flags: --manifest")
	}
	if *timeout <= 0 {
		return fmt.Errorf("--timeout must be greater than zero")
	}

	kubeClient, currentNamespace, err := loadKubeClient()
	if err != nil {
		return err
	}
	manifest, err := loadPodManifest(*manifestPath)
	if err != nil {
		return err
	}
	namespace := resolveNamespace(*namespaceOverride, manifest.Namespace, currentNamespace)
	if strings.TrimSpace(*snapshotID) == "" {
		*snapshotID = fmt.Sprintf("%s-%d", defaultGeneratedSnapshotIDPrefix, time.Now().UTC().UnixNano())
	}

	request := &snapshotv1alpha1.SnapshotRequest{
		ObjectMeta: metav1ObjectMetaForRequest("checkpoint", namespace, manifest.Name, *snapshotID),
		Spec: snapshotv1alpha1.SnapshotRequestSpec{
			Phase:       snapshotv1alpha1.SnapshotRequestPhaseCheckpoint,
			SnapshotID:  *snapshotID,
			PodTemplate: manifest.Template.DeepCopy(),
		},
	}
	if err := kubeClient.Create(context.Background(), request); err != nil {
		return fmt.Errorf("create SnapshotRequest %s/%s: %w", request.Namespace, request.Name, err)
	}

	request, err = waitForSnapshotRequest(kubeClient, request.Namespace, request.Name, *timeout)
	if err != nil {
		return err
	}

	fmt.Printf("status=%s\n", request.Status.State)
	fmt.Printf("namespace=%s\n", request.Namespace)
	fmt.Printf("name=%s\n", request.Name)
	fmt.Printf("checkpoint_job=%s\n", request.Status.JobName)
	fmt.Printf("checkpoint_hash=%s\n", request.Spec.SnapshotID)
	fmt.Printf("checkpoint_location=%s\n", request.Status.Location)
	return nil
}

func runRestore(args []string) error {
	flags := flag.NewFlagSet("restore", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)

	manifestPath := flags.String("manifest", "", "Path to a worker Pod manifest")
	namespaceOverride := flags.String("namespace", "", "Namespace override; defaults to the manifest namespace or current kube context namespace")
	snapshotID := flags.String("checkpoint-hash", "", "Snapshot identifier to restore")
	timeout := flags.Duration("timeout", 45*time.Minute, "Maximum time to wait for restore completion")

	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(flags.Args()) != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	if strings.TrimSpace(*manifestPath) == "" || strings.TrimSpace(*snapshotID) == "" {
		return fmt.Errorf("missing required flags: --manifest, --checkpoint-hash")
	}
	if *timeout <= 0 {
		return fmt.Errorf("--timeout must be greater than zero")
	}

	kubeClient, currentNamespace, err := loadKubeClient()
	if err != nil {
		return err
	}
	manifest, err := loadPodManifest(*manifestPath)
	if err != nil {
		return err
	}
	namespace := resolveNamespace(*namespaceOverride, manifest.Namespace, currentNamespace)

	request := &snapshotv1alpha1.SnapshotRequest{
		ObjectMeta: metav1ObjectMetaForRequest("restore", namespace, manifest.Name, *snapshotID),
		Spec: snapshotv1alpha1.SnapshotRequestSpec{
			Phase:       snapshotv1alpha1.SnapshotRequestPhaseRestore,
			SnapshotID:  *snapshotID,
			PodTemplate: manifest.Template.DeepCopy(),
		},
	}
	if err := kubeClient.Create(context.Background(), request); err != nil {
		return fmt.Errorf("create SnapshotRequest %s/%s: %w", request.Namespace, request.Name, err)
	}

	request, err = waitForSnapshotRequest(kubeClient, request.Namespace, request.Name, *timeout)
	if err != nil {
		return err
	}

	fmt.Printf("status=%s\n", request.Status.State)
	fmt.Printf("namespace=%s\n", request.Namespace)
	fmt.Printf("name=%s\n", request.Name)
	fmt.Printf("restore_pod=%s\n", request.Status.PodName)
	fmt.Printf("checkpoint_hash=%s\n", request.Spec.SnapshotID)
	fmt.Printf("checkpoint_location=%s\n", request.Status.Location)
	return nil
}

func loadKubeClient() (ctrlclient.Client, string, error) {
	configLoader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	)
	restConfig, err := configLoader.ClientConfig()
	if err != nil {
		return nil, "", fmt.Errorf("load kubeconfig: %w", err)
	}
	namespace, _, err := configLoader.Namespace()
	if err != nil {
		return nil, "", fmt.Errorf("resolve current namespace: %w", err)
	}
	if strings.TrimSpace(namespace) == "" {
		namespace = corev1.NamespaceDefault
	}

	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		return nil, "", err
	}
	if err := snapshotv1alpha1.AddToScheme(scheme); err != nil {
		return nil, "", err
	}

	kubeClient, err := ctrlclient.New(restConfig, ctrlclient.Options{Scheme: scheme})
	if err != nil {
		return nil, "", fmt.Errorf("create kubernetes client: %w", err)
	}
	return kubeClient, namespace, nil
}

func loadPodManifest(manifestPath string) (podManifest, error) {
	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return podManifest{}, fmt.Errorf("read manifest %s: %w", manifestPath, err)
	}

	var pod corev1.Pod
	jsonContent, err := utilyaml.ToJSON(content)
	if err != nil {
		return podManifest{}, fmt.Errorf("convert manifest %s to json: %w", manifestPath, err)
	}
	if err := json.Unmarshal(jsonContent, &pod); err != nil {
		return podManifest{}, fmt.Errorf("parse manifest %s: %w", manifestPath, err)
	}
	if kind := strings.TrimSpace(pod.Kind); kind != "" && kind != "Pod" {
		return podManifest{}, fmt.Errorf("manifest %s is kind %q, expected Pod", manifestPath, kind)
	}
	if len(pod.Spec.Containers) != 1 {
		return podManifest{}, fmt.Errorf("manifest %s has %d containers; snapshotctl requires exactly one worker container", manifestPath, len(pod.Spec.Containers))
	}
	if strings.TrimSpace(pod.Spec.Containers[0].Image) == "" {
		return podManifest{}, fmt.Errorf("manifest %s: worker container image is required", manifestPath)
	}

	return podManifest{
		Name:      sanitizeName(firstNonEmpty(pod.Name, "snapshot-worker")),
		Namespace: strings.TrimSpace(pod.Namespace),
		Template: corev1.PodTemplateSpec{
			ObjectMeta: pod.ObjectMeta,
			Spec:       *pod.Spec.DeepCopy(),
		},
	}, nil
}

func waitForSnapshotRequest(kubeClient ctrlclient.Client, namespace string, name string, timeout time.Duration) (*snapshotv1alpha1.SnapshotRequest, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		request := &snapshotv1alpha1.SnapshotRequest{}
		if err := kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, request); err != nil {
			return nil, fmt.Errorf("get SnapshotRequest %s/%s: %w", namespace, name, err)
		}
		switch request.Status.State {
		case snapshotv1alpha1.SnapshotRequestStateSucceeded:
			return request, nil
		case snapshotv1alpha1.SnapshotRequestStateFailed:
			if strings.TrimSpace(request.Status.Message) == "" {
				return nil, fmt.Errorf("SnapshotRequest %s/%s failed", namespace, name)
			}
			return nil, fmt.Errorf("SnapshotRequest %s/%s failed: %s", namespace, name, request.Status.Message)
		}

		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("timed out waiting for SnapshotRequest %s/%s after %s", namespace, name, timeout)
		case <-ticker.C:
		}
	}
}

func resolveNamespace(explicitNamespace string, manifestNamespace string, currentNamespace string) string {
	if namespace := strings.TrimSpace(explicitNamespace); namespace != "" {
		return namespace
	}
	if namespace := strings.TrimSpace(manifestNamespace); namespace != "" {
		return namespace
	}
	if namespace := strings.TrimSpace(currentNamespace); namespace != "" {
		return namespace
	}
	return corev1.NamespaceDefault
}

func sanitizeName(value string) string {
	value = strings.ToLower(value)
	value = dnsLabelPattern.ReplaceAllString(value, "-")
	value = strings.Trim(value, "-")
	if value == "" {
		return "snapshot"
	}
	if len(value) > 63 {
		value = strings.Trim(value[:63], "-")
		if value == "" {
			return "snapshot"
		}
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func metav1ObjectMetaForRequest(phase string, namespace string, manifestName string, snapshotID string) metav1.ObjectMeta {
	prefix := sanitizeName(fmt.Sprintf("snapshot-%s-%s-%s", phase, manifestName, snapshotID))
	if len(prefix) > 57 {
		prefix = prefix[:57]
		prefix = strings.TrimRight(prefix, "-")
	}
	if prefix == "" {
		prefix = "snapshot-request"
	}
	return metav1.ObjectMeta{
		Namespace:    namespace,
		GenerateName: prefix + "-",
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `snapshotctl submits SnapshotRequest resources from a worker Pod manifest.

Subcommands:
  checkpoint
  restore

Examples:
  snapshotctl checkpoint --manifest /tmp/vllm-worker-pod.yaml
  snapshotctl restore --manifest /tmp/sglang-worker-pod.yaml --checkpoint-hash manual-snapshot-123
`)
}
