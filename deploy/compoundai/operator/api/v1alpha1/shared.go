package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type BaseStatus struct {
	Version    string             `json:"version,omitempty"`
	State      string             `json:"state,omitempty"`
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`
}

func (b *BaseStatus) GetConditions() []metav1.Condition {
	return b.Conditions
}

func (b *BaseStatus) SetConditions(conditions []metav1.Condition) {
	b.Conditions = conditions
}

type BaseCRD struct {
	Status *BaseStatus `json:"status,omitempty"`
}

func (b *BaseCRD) GetStatusConditions() []metav1.Condition {
	if b.Status != nil {
		return b.Status.GetConditions()
	}
	return nil
}

func (b *BaseCRD) SetStatusConditions(conditions []metav1.Condition) {
	status := b.Status
	if status == nil {
		status = &BaseStatus{}
		b.Status = status
	}
	status.Conditions = conditions
}

func (b *BaseCRD) GetVersion() string {
	if b.Status != nil {
		return b.Status.Version
	}
	return ""
}

func (b *BaseCRD) SetVersion(version string) {
	status := b.Status
	if status == nil {
		status = &BaseStatus{}
		b.Status = status
	}
	status.Version = version
}

func (b *BaseCRD) SetState(state string) {
	status := b.Status
	if status == nil {
		status = &BaseStatus{}
		b.Status = status
	}
	status.State = state
}

func (n *BaseCRD) GetHelmVersionMatrix() map[string]string {
	return nil
}
