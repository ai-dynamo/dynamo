package secrets

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type DockerSecretIndexer struct {
	// maps for a namespace, a docker registry server to a list of secret names
	secrets   map[string]map[string][]string
	client    client.Reader
	namespace string
	mu        sync.RWMutex
}

func NewDockerSecretIndexer(client client.Reader) *DockerSecretIndexer {
	return &DockerSecretIndexer{
		secrets: make(map[string]map[string][]string),
		client:  client,
	}
}

func NewNamespacedDockerSecretIndexer(client client.Reader, namespace string) *DockerSecretIndexer {
	indexer := NewDockerSecretIndexer(client)
	indexer.namespace = namespace
	return indexer
}

func (i *DockerSecretIndexer) RefreshIndex(ctx context.Context) error {
	// scan for all secrets in the namespace
	secrets := &corev1.SecretList{}
	var opts []client.ListOption
	if i.namespace != "" {
		opts = append(opts, client.InNamespace(i.namespace))
	}
	if err := i.client.List(ctx, secrets, opts...); err != nil {
		return fmt.Errorf("unable to list secrets: %w", err)
	}
	slices.SortFunc(secrets.Items, func(a, b corev1.Secret) int {
		if a.Namespace != b.Namespace {
			if a.Namespace < b.Namespace {
				return -1
			}
			return 1
		}
		if a.Name < b.Name {
			return -1
		}
		if a.Name > b.Name {
			return 1
		}
		return 0
	})
	tmpSecrets := make(map[string]map[string][]string)
	for _, secret := range secrets.Items {
		if secret.Type == corev1.SecretTypeDockerConfigJson {
			// unmarshal the secret data
			dockerConfig := &struct {
				Auths map[string]any `json:"auths"`
			}{}
			if err := json.Unmarshal(secret.Data[corev1.DockerConfigJsonKey], dockerConfig); err != nil {
				return fmt.Errorf("unable to unmarshal docker config json for secret %s: %w", secret.Name, err)
			}
			namespace := secret.Namespace
			if _, ok := tmpSecrets[namespace]; !ok {
				tmpSecrets[namespace] = make(map[string][]string)
			}
			for auth := range dockerConfig.Auths {
				// retrieve the registry host
				registry, err := common.GetHost(auth)
				if err != nil {
					return fmt.Errorf("unable to get host for registry %s for secret %s: %w", auth, secret.Name, err)
				}
				tmpSecrets[namespace][registry] = append(tmpSecrets[namespace][registry], secret.Name)
			}
		}
	}
	for namespace := range tmpSecrets {
		for registry, secretNames := range tmpSecrets[namespace] {
			slices.Sort(secretNames)
			tmpSecrets[namespace][registry] = slices.Compact(secretNames)
		}
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	i.secrets = tmpSecrets
	return nil
}

func (i *DockerSecretIndexer) GetSecrets(namespace, registry string) ([]string, error) {
	registry, err := common.GetHost(registry)
	if err != nil {
		return nil, err
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return append([]string(nil), i.secrets[namespace][registry]...), nil
}
