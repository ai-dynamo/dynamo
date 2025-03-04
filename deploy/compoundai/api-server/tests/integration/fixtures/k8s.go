package fixtures

import (
	"context"
	"fmt"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/models"

	"github.com/rs/zerolog/log"
	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	v1 "k8s.io/client-go/listers/core/v1"
)

type MockedK8sService struct{}

func (s *MockedK8sService) GetK8sClient(kubeConfig string) (kubernetes.Interface, error) {
	log.Info().Msgf("Using fake client.")
	return fake.NewClientset(), nil
}

func (s *MockedK8sService) ListPodsByDeployment(ctx context.Context, podLister v1.PodNamespaceLister, deployment *models.Deployment) ([]*apiv1.Pod, error) {
	log.Info().Msgf("Faking list by deployment")
	selector, err := labels.Parse(fmt.Sprintf("%s = %s", consts.KubeLabelCompoundNimVersionDeployment, deployment.Name))
	if err != nil {
		return nil, err
	}

	return s.ListPodsBySelector(ctx, podLister, selector)
}

func (s *MockedK8sService) ListPodsBySelector(ctx context.Context, podLister v1.PodNamespaceLister, selector labels.Selector) ([]*apiv1.Pod, error) {
	log.Info().Msgf("Faking list by selector")
	pods, err := podLister.List(selector)
	if err != nil {
		return nil, err
	}

	return pods, nil
}
