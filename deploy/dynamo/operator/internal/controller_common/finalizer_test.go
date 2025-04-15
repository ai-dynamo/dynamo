package controller_common

import (
	"context"
	"errors"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// finalize mock object
type FinalizerMock[T client.Object] struct {
	FinalizeResourceFunc func(ctx context.Context, obj T) error
}

func (f *FinalizerMock[T]) FinalizeResource(ctx context.Context, obj T) error {
	return f.FinalizeResourceFunc(ctx, obj)
}

// client.Object mock object
type ObjectMock struct {
	client.Object
	GetDeletionTimestampFunc func() *metav1.Time
	GetResourceVersionFunc   func() string
	GetFinalizersFunc        func() []string
}

func (o *ObjectMock) GetDeletionTimestamp() *metav1.Time {
	return o.GetDeletionTimestampFunc()
}

func (o *ObjectMock) GetFinalizers() []string {
	return o.GetFinalizersFunc()
}

func (o *ObjectMock) SetFinalizers(finalizers []string) {
}

func (o *ObjectMock) GetResourceVersion() string {
	return o.GetResourceVersionFunc()
}

// writer mock object
type WriterMock struct {
	client.Writer
	UpdateFunc func(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error
}

func (w *WriterMock) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	return w.UpdateFunc(ctx, obj, opts...)
}

func TestHandleFinalizer(t *testing.T) {
	type args struct {
		ctx       context.Context
		obj       client.Object
		writer    client.Writer
		finalizer Finalizer[client.Object]
	}
	tests := []struct {
		name    string
		args    args
		want    bool
		wantErr bool
	}{
		{
			name: "deleted object with finalizer - nominal case",
			args: args{
				ctx: context.Background(),
				obj: &ObjectMock{
					GetDeletionTimestampFunc: func() *metav1.Time {
						return &metav1.Time{Time: time.Now()}
					},
					GetFinalizersFunc: func() []string {
						return []string{finalizerName}
					},
					GetResourceVersionFunc: func() string {
						return "1"
					},
				},
				writer: &WriterMock{
					UpdateFunc: func(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
						return nil
					},
				},
				finalizer: &FinalizerMock[client.Object]{
					FinalizeResourceFunc: func(ctx context.Context, obj client.Object) error {
						return nil
					},
				},
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "deleted object with finalizer - object update error",
			args: args{
				ctx: context.Background(),
				obj: &ObjectMock{
					GetDeletionTimestampFunc: func() *metav1.Time {
						return &metav1.Time{Time: time.Now()}
					},
					GetFinalizersFunc: func() []string {
						return []string{finalizerName}
					},
					GetResourceVersionFunc: func() string {
						return "1"
					},
				},
				writer: &WriterMock{
					UpdateFunc: func(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
						return errors.New("update error")
					},
				},
				finalizer: &FinalizerMock[client.Object]{
					FinalizeResourceFunc: func(ctx context.Context, obj client.Object) error {
						return nil
					},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "deleted object with finalizer - finalize error",
			args: args{
				ctx: context.Background(),
				obj: &ObjectMock{
					GetDeletionTimestampFunc: func() *metav1.Time {
						return &metav1.Time{Time: time.Now()}
					},
					GetFinalizersFunc: func() []string {
						return []string{finalizerName}
					},
					GetResourceVersionFunc: func() string {
						return "1"
					},
				},
				finalizer: &FinalizerMock[client.Object]{
					FinalizeResourceFunc: func(ctx context.Context, obj client.Object) error {
						return errors.New("finalize error")
					},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "non deleted object without finalizer - nominal case",
			args: args{
				ctx: context.Background(),
				obj: &ObjectMock{
					GetDeletionTimestampFunc: func() *metav1.Time {
						return nil
					},
					GetFinalizersFunc: func() []string {
						return []string{}
					},
					GetResourceVersionFunc: func() string {
						return "1"
					},
				},
				writer: &WriterMock{
					UpdateFunc: func(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
						return nil
					},
				},
			},
			want:    false,
			wantErr: false,
		},
		{
			name: "non deleted object without finalizer - update error",
			args: args{
				ctx: context.Background(),
				obj: &ObjectMock{
					GetDeletionTimestampFunc: func() *metav1.Time {
						return nil
					},
					GetFinalizersFunc: func() []string {
						return []string{}
					},
					GetResourceVersionFunc: func() string {
						return "1"
					},
				},
				writer: &WriterMock{
					UpdateFunc: func(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
						return errors.New("update error")
					},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "non deleted object with finalizer - nominal case",
			args: args{
				ctx: context.Background(),
				obj: &ObjectMock{
					GetDeletionTimestampFunc: func() *metav1.Time {
						return nil
					},
					GetFinalizersFunc: func() []string {
						return []string{finalizerName}
					},
				},
			},
			want:    false,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := HandleFinalizer(tt.args.ctx, tt.args.obj, tt.args.writer, tt.args.finalizer)
			if (err != nil) != tt.wantErr {
				t.Errorf("HandleFinalizer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("HandleFinalizer() = %v, want %v", got, tt.want)
			}
		})
	}
}
