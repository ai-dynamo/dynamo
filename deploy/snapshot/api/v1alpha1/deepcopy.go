package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func (in *SnapshotRequestSpec) DeepCopyInto(out *SnapshotRequestSpec) {
	*out = *in
	if in.PodTemplate != nil {
		out.PodTemplate = new(corev1.PodTemplateSpec)
		in.PodTemplate.DeepCopyInto(out.PodTemplate)
	}
}

func (in *SnapshotRequestSpec) DeepCopy() *SnapshotRequestSpec {
	if in == nil {
		return nil
	}
	out := new(SnapshotRequestSpec)
	in.DeepCopyInto(out)
	return out
}

func (in *SnapshotRequestStatus) DeepCopyInto(out *SnapshotRequestStatus) {
	*out = *in
	if in.StartedAt != nil {
		out.StartedAt = in.StartedAt.DeepCopy()
	}
	if in.CompletedAt != nil {
		out.CompletedAt = in.CompletedAt.DeepCopy()
	}
}

func (in *SnapshotRequestStatus) DeepCopy() *SnapshotRequestStatus {
	if in == nil {
		return nil
	}
	out := new(SnapshotRequestStatus)
	in.DeepCopyInto(out)
	return out
}

func (in *SnapshotRequest) DeepCopyInto(out *SnapshotRequest) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	in.Spec.DeepCopyInto(&out.Spec)
	in.Status.DeepCopyInto(&out.Status)
}

func (in *SnapshotRequest) DeepCopy() *SnapshotRequest {
	if in == nil {
		return nil
	}
	out := new(SnapshotRequest)
	in.DeepCopyInto(out)
	return out
}

func (in *SnapshotRequest) DeepCopyObject() runtime.Object {
	if out := in.DeepCopy(); out != nil {
		return out
	}
	return nil
}

func (in *SnapshotRequestList) DeepCopyInto(out *SnapshotRequestList) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ListMeta.DeepCopyInto(&out.ListMeta)
	if in.Items != nil {
		out.Items = make([]SnapshotRequest, len(in.Items))
		for i := range in.Items {
			in.Items[i].DeepCopyInto(&out.Items[i])
		}
	}
}

func (in *SnapshotRequestList) DeepCopy() *SnapshotRequestList {
	if in == nil {
		return nil
	}
	out := new(SnapshotRequestList)
	in.DeepCopyInto(out)
	return out
}

func (in *SnapshotRequestList) DeepCopyObject() runtime.Object {
	if out := in.DeepCopy(); out != nil {
		return out
	}
	return nil
}
