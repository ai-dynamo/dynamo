/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package podcache

import (
	"context"
	"strings"
	"sync"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	toolscache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/watchlist"
	"sigs.k8s.io/controller-runtime/pkg/cache"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// configureRetention filters the typed Pod informer before its store receives objects.
// The API list/watch remains unrestricted; this bounds retained objects, not API traffic.
func configureRetention(options *cache.Options) {
	newInformer := options.NewInformer
	if newInformer == nil {
		newInformer = toolscache.NewSharedIndexInformer
	}

	// Preserve custom informer construction and leave every other resource untouched.
	options.NewInformer = func(lw toolscache.ListerWatcher, obj runtime.Object, period time.Duration, indexes toolscache.Indexers) toolscache.SharedIndexInformer {
		if _, ok := obj.(*corev1.Pod); ok {
			if _, wrapped := lw.(*retainedPodListWatch); !wrapped {
				lw = &retainedPodListWatch{
					source:               toolscache.ToListerWatcherWithContext(lw),
					unsupportedWatchList: watchlist.DoesClientNotSupportWatchListSemantics(lw),
				}
			}
		}
		return newInformer(lw, obj, period, indexes)
	}
}

// interestedPod includes each existing watch/index contract, including older Jobs
// whose Pod templates have no Dynamo labels.
func interestedPod(pod *corev1.Pod) bool {
	labels := pod.Labels
	if labels[consts.KubeLabelDynamoGraphDeploymentName] != "" ||
		labels[consts.KubeLabelDynamoSelector] != "" ||
		(labels[consts.KubeLabelDynamoComponent] != "" && labels[consts.KubeLabelDynamoNamespace] != "") ||
		labels[consts.KubeLabelDynamoFailoverEngineGroupMember] == consts.KubeLabelValueTrue {
		return true
	}

	// Job labels are not inherited by Pods, so legacy Jobs require owner identity.
	owner := metav1.GetControllerOf(pod)
	return owner != nil && owner.Kind == "Job" && owner.APIVersion == batchv1.SchemeGroupVersion.String() &&
		(strings.HasPrefix(owner.Name, "profile-") || strings.HasSuffix(owner.Name, "-aisimulate"))
}

type retainedPodListWatch struct {
	unsupportedWatchList bool
	source               toolscache.ListerWatcherWithContext
	mu                   sync.Mutex
	admitted             map[string]*corev1.Pod
	pending              map[string]*corev1.Pod
	continuation         string
	active               int
}

func (lw *retainedPodListWatch) IsWatchListSemanticsUnSupported() bool {
	return lw.unsupportedWatchList
}

func podKey(pod *corev1.Pod) string { return pod.Namespace + "/" + pod.Name }

func (lw *retainedPodListWatch) List(options metav1.ListOptions) (runtime.Object, error) {
	return lw.ListWithContext(context.Background(), options)
}

func (lw *retainedPodListWatch) ListWithContext(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
	obj, err := lw.source.ListWithContext(ctx, options)
	if err != nil {
		return nil, err
	}
	list, ok := obj.(*corev1.PodList)
	if !ok {
		return obj, nil
	}

	// Preserve pagination and collection resource versions while removing unrelated Pods.
	filtered := &corev1.PodList{TypeMeta: list.TypeMeta, ListMeta: list.ListMeta}
	for i := range list.Items {
		if interestedPod(&list.Items[i]) {
			filtered.Items = append(filtered.Items, list.Items[i])
		}
	}

	// Consistency-check lists must never change the active watch's membership.
	lw.mu.Lock()
	defer lw.mu.Unlock()
	if lw.active != 0 {
		return filtered, nil
	}
	if options.Continue == "" {
		lw.pending = make(map[string]*corev1.Pod)
	} else if lw.pending == nil || options.Continue != lw.continuation {
		return filtered, nil
	}

	// Publish a replacement snapshot only after the last page of a successful list.
	for i := range filtered.Items {
		pod := &filtered.Items[i]
		lw.pending[podKey(pod)] = Project(pod.DeepCopy())
	}
	lw.continuation = list.Continue
	if list.Continue == "" {
		lw.admitted, lw.pending = lw.pending, nil
	}
	return filtered, nil
}

func (lw *retainedPodListWatch) Watch(options metav1.ListOptions) (watch.Interface, error) {
	return lw.WatchWithContext(context.Background(), options)
}

func (lw *retainedPodListWatch) WatchWithContext(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
	ctx, cancel := context.WithCancel(ctx)
	source, err := lw.source.WatchWithContext(ctx, options)
	if err != nil {
		cancel()
		return nil, err
	}

	// Every watch-list attempt builds a fresh snapshot; ordinary reconnects keep membership.
	initial := options.SendInitialEvents != nil && *options.SendInitialEvents
	lw.mu.Lock()
	if initial || lw.admitted == nil {
		lw.admitted = make(map[string]*corev1.Pod)
	}
	lw.active++
	lw.mu.Unlock()

	// An unbuffered channel keeps membership advancement coupled to reflector delivery.
	stream := &retainedPodWatch{cancel: cancel, events: make(chan watch.Event), done: make(chan struct{})}
	go lw.forward(ctx, source, stream, initial)
	return stream, nil
}

type retainedPodWatch struct {
	cancel context.CancelFunc
	events chan watch.Event
	done   chan struct{}
}

func (w *retainedPodWatch) Stop() {
	w.cancel()
	<-w.done
}
func (w *retainedPodWatch) ResultChan() <-chan watch.Event { return w.events }

func (lw *retainedPodListWatch) forward(ctx context.Context, source watch.Interface, stream *retainedPodWatch, initial bool) {
	defer close(stream.done)
	defer close(stream.events)
	defer source.Stop()
	defer stream.cancel()
	defer func() {
		lw.mu.Lock()
		lw.active--
		if initial {
			lw.admitted = nil
		}
		lw.mu.Unlock()
	}()

	// Forward API control events unchanged, including the watch-list completion bookmark.
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-source.ResultChan():
			if !ok {
				return
			}
			// Read membership without holding a lock while the reflector receives events.
			// It can synchronously run a consistency LIST between watch events.
			lw.mu.Lock()
			out, key, next, deliver := lw.filterEvent(event, initial)
			lw.mu.Unlock()
			if !deliver {
				continue
			}
			select {
			case <-ctx.Done():
				return
			case stream.events <- out:
				if out.Type == watch.Bookmark {
					if pod, ok := out.Object.(*corev1.Pod); ok && pod.Annotations[metav1.InitialEventsAnnotationKey] == "true" {
						initial = false
					}
				}

				// Commit before processing the next event; Stop joins this goroutine.
				lw.mu.Lock()
				if key != "" {
					if next == nil {
						delete(lw.admitted, key)
					} else {
						lw.admitted[key] = next
					}
				}
				lw.mu.Unlock()
			}
		}
	}
}

// filterEvent requires mu to be held. It returns a pending membership change
// that is committed only once the reflector has received the event.
func (lw *retainedPodListWatch) filterEvent(event watch.Event, initial bool) (watch.Event, string, *corev1.Pod, bool) {
	pod, ok := event.Object.(*corev1.Pod)
	if !ok || (event.Type != watch.Added && event.Type != watch.Modified && event.Type != watch.Deleted) {
		return event, "", nil, true
	}
	key := podKey(pod)
	previous := lw.admitted[key]

	// Keep the last admitted identity for removals even when current labels no longer match.
	if event.Type != watch.Deleted && interestedPod(pod) {
		if previous == nil {
			event.Type = watch.Added
		}
		return event, key, Project(pod.DeepCopy()), true
	}
	if previous != nil {
		deleted := previous.DeepCopy()
		deleted.ResourceVersion = pod.ResourceVersion
		return watch.Event{Type: watch.Deleted, Object: deleted}, key, nil, true
	}

	// Initial synthetic Adds have no resume RV until the real completion bookmark.
	if initial {
		return watch.Event{}, "", nil, false
	}
	return watch.Event{Type: watch.Bookmark, Object: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: pod.ResourceVersion}}}, "", nil, true
}
