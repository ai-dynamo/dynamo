package dynamo

import (
	"maps"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestComponentsByNameNil(t *testing.T) {
	if got := ComponentsByName(nil); len(got) != 0 {
		t.Fatalf("ComponentsByName(nil) = %#v, want empty map", got)
	}
}

func TestGetDCDComponentNamePrefersSpecOverLegacyMetadata(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "metadata-name",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoComponent: "label-component",
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "spec-component",
			},
		},
	}

	if got, want := GetDCDComponentName(dcd), "spec-component"; got != want {
		t.Fatalf("GetDCDComponentName() = %q, want %q", got, want)
	}
}

func TestGetDCDComponentNameLegacyFallbacks(t *testing.T) {
	tests := []struct {
		name string
		dcd  *v1beta1.DynamoComponentDeployment
		want string
	}{
		{
			name: "nil",
			want: "",
		},
		{
			name: "label",
			dcd: &v1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "metadata-name",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoComponent: "label-component",
					},
				},
			},
			want: "label-component",
		},
		{
			name: "metadata name",
			dcd: &v1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "metadata-name"},
			},
			want: "metadata-name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetDCDComponentName(tt.dcd); got != tt.want {
				t.Fatalf("GetDCDComponentName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMergeLowPriorityMetadata(t *testing.T) {
	got := mergeLowPriorityMetadata(
		map[string]string{"existing": "kept", "shared": "winner"},
		map[string]string{"shared": "ignored", "new": "added"},
	)
	want := map[string]string{"existing": "kept", "shared": "winner", "new": "added"}
	if !maps.Equal(got, want) {
		t.Fatalf("mergeLowPriorityMetadata() = %#v, want %#v", got, want)
	}
}
