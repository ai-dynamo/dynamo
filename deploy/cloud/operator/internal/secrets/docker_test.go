package secrets

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDockerSecretIndexer_RefreshIndex(t *testing.T) {

	// Create mock secrets
	mockSecrets := []corev1.Secret{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "secret1",
				Namespace: "default",
			},
			Type: corev1.SecretTypeDockerConfigJson,
			Data: map[string][]byte{
				".dockerconfigjson": []byte(`{"auths":{"docker.io":{}, "my-registry.com:5005/registry1":{}}}`),
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "secret2",
				Namespace: "default",
			},
			Type: corev1.SecretTypeDockerConfigJson,
			Data: map[string][]byte{
				".dockerconfigjson": []byte(`{"auths":{"my-registry.com:5005/registry2":{}}}`),
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "secret3",
				Namespace: "another-namespace",
			},
			Type: corev1.SecretTypeDockerConfigJson,
			Data: map[string][]byte{
				".dockerconfigjson": []byte(`{"auths":{"my-registry.com:5005/registry2":{}}}`),
			},
		},
	}

	// Create fake client with mock secrets
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(&mockSecrets[0], &mockSecrets[1], &mockSecrets[2]).
		Build()

	i := NewDockerSecretIndexer(fakeClient)
	if err := i.RefreshIndex(context.Background()); err != nil {
		t.Errorf("DockerSecretIndexer.RefreshIndex() error = %v, wantErr %v", err, nil)
	}

	secrets, err := i.GetSecrets("default", "docker.io")
	if err != nil {
		t.Errorf("DockerSecretIndexer.GetSecrets() error = %v, wantErr %v", err, nil)
	}
	if len(secrets) != 1 {
		t.Errorf("DockerSecretIndexer.GetSecrets() = %v, want %v", len(secrets), 1)
	}
	if secrets[0] != "secret1" {
		t.Errorf("DockerSecretIndexer.GetSecrets() = %v, want %v", secrets[0], "secret1")
	}

	secrets, err = i.GetSecrets("default", "my-registry.com:5005")
	if err != nil {
		t.Errorf("DockerSecretIndexer.GetSecrets() error = %v, wantErr %v", err, nil)
	}
	if len(secrets) != 2 {
		t.Errorf("DockerSecretIndexer.GetSecrets() = %v, want %v", len(secrets), 2)
	}
	if secrets[0] != "secret1" {
		t.Errorf("DockerSecretIndexer.GetSecrets() = %v, want %v", secrets[0], "secret1")
	}
	if secrets[1] != "secret2" {
		t.Errorf("DockerSecretIndexer.GetSecrets() = %v, want %v", secrets[1], "secret2")
	}

	secrets, err = i.GetSecrets("another-namespace", "my-registry.com:5005")
	if err != nil {
		t.Errorf("DockerSecretIndexer.GetSecrets() error = %v, wantErr %v", err, nil)
	}
	if len(secrets) != 1 {
		t.Errorf("DockerSecretIndexer.GetSecrets() = %v, want %v", len(secrets), 1)
	}
	if secrets[0] != "secret3" {
		t.Errorf("DockerSecretIndexer.GetSecrets() = %v, want %v", secrets[0], "secret3")
	}
}
