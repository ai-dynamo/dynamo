package secrets

import (
	"context"
	"encoding/json"
	"errors"
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

// NewDockerSecretIndexer builds an indexer scoped to namespace. An empty namespace indexes all namespaces.
func NewDockerSecretIndexer(reader client.Reader, namespace string) *DockerSecretIndexer {
	return &DockerSecretIndexer{
		secrets:   make(map[string]map[string][]string),
		client:    reader,
		namespace: namespace,
	}
}

func (i *DockerSecretIndexer) RefreshIndex(ctx context.Context) error {
	// scan for all secrets in the namespace
	secrets := &corev1.SecretList{}
	if err := i.client.List(ctx, secrets, i.listOptions()...); err != nil {
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
	var refreshErrors []error
	for _, secret := range secrets.Items {
		if secret.Type == corev1.SecretTypeDockerConfigJson {
			// unmarshal the secret data
			dockerConfig := &struct {
				Auths map[string]any `json:"auths"`
			}{}
			if err := json.Unmarshal(secret.Data[corev1.DockerConfigJsonKey], dockerConfig); err != nil {
				refreshErrors = append(refreshErrors, fmt.Errorf("unable to unmarshal docker config json for secret %s/%s: %w", secret.Namespace, secret.Name, err))
				continue
			}
			namespace := secret.Namespace
			if _, ok := tmpSecrets[namespace]; !ok {
				tmpSecrets[namespace] = make(map[string][]string)
			}
			for auth := range dockerConfig.Auths {
				// retrieve the registry host
				registry, err := common.GetHost(auth)
				if err != nil {
					refreshErrors = append(refreshErrors, fmt.Errorf("unable to get host for registry %q for secret %s/%s: %w", auth, secret.Namespace, secret.Name, err))
					continue
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
	if len(refreshErrors) > 0 {
		return errors.Join(refreshErrors...)
	}
	return nil
}

func (i *DockerSecretIndexer) listOptions() []client.ListOption {
	if i.namespace == "" {
		return nil
	}
	return []client.ListOption{client.InNamespace(i.namespace)}
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
