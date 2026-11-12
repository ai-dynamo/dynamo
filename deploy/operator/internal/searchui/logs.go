/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"bytes"
	"context"
	"errors"
	"io"
	"sync"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui/history"
	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// LogContainers are the Run Job containers whose output the UI retains.
var LogContainers = []string{"sweeper", "publisher"}

const (
	logBufferMax   = 8 << 20
	logBufferTrim  = 6 << 20
	logFlushPeriod = 5 * time.Second
	podPollPeriod  = 3 * time.Second
	podWaitMax     = 30 * time.Minute
)

// LogManager streams the containers of each Run's Job pod while it exists, keeps the
// output in memory for live subscribers, and flushes it to the history store.
type LogManager struct {
	client kubernetes.Interface
	store  history.Store
	log    logr.Logger

	mu   sync.Mutex
	runs map[string]*runLogs // by run UID
}

type runLogs struct {
	namespace  string
	uid        string
	jobName    string
	containers map[string]*logBuffer
}

type logBuffer struct {
	mu    sync.Mutex
	data  []byte
	dirty bool
	done  bool
	subs  map[chan []byte]struct{}
}

func NewLogManager(client kubernetes.Interface, store history.Store, log logr.Logger) *LogManager {
	return &LogManager{client: client, store: store, log: log, runs: map[string]*runLogs{}}
}

// Ensure starts streaming for a Run once its Job is known. terminal says the Run is
// finished, so a missing pod means the logs are gone rather than not yet started.
func (m *LogManager) Ensure(ctx context.Context, namespace, uid, jobName string, terminal bool) {
	if jobName == "" || m.client == nil {
		return
	}
	m.mu.Lock()
	if _, ok := m.runs[uid]; ok {
		m.mu.Unlock()
		return
	}
	rl := &runLogs{namespace: namespace, uid: uid, jobName: jobName, containers: map[string]*logBuffer{}}
	for _, c := range LogContainers {
		rl.containers[c] = &logBuffer{subs: map[chan []byte]struct{}{}}
	}
	m.runs[uid] = rl
	m.mu.Unlock()
	go m.stream(ctx, rl, terminal)
}

func (m *LogManager) stream(ctx context.Context, rl *runLogs, terminal bool) {
	log := m.log.WithValues("namespace", rl.namespace, "job", rl.jobName)
	pod, err := m.findPod(ctx, rl, terminal)
	if err != nil {
		log.V(1).Info("no pod to stream logs from", "reason", err.Error())
		for _, c := range LogContainers {
			rl.containers[c].finish()
		}
		return
	}
	var wg sync.WaitGroup
	for _, c := range LogContainers {
		wg.Add(1)
		go func(container string) {
			defer wg.Done()
			m.streamContainer(ctx, rl, pod.Name, container)
		}(c)
	}
	wg.Wait()
}

// findPod waits for the Job's pod. Job pods carry the job-name label.
func (m *LogManager) findPod(ctx context.Context, rl *runLogs, terminal bool) (*corev1.Pod, error) {
	deadline := time.Now().Add(podWaitMax)
	attempts := 0
	for {
		attempts++
		pods, err := m.client.CoreV1().Pods(rl.namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + rl.jobName})
		if err != nil {
			return nil, err
		}
		if len(pods.Items) > 0 {
			// Prefer the newest pod: a retried Job may leave failed ones behind.
			newest := pods.Items[0]
			for _, p := range pods.Items[1:] {
				if p.CreationTimestamp.After(newest.CreationTimestamp.Time) {
					newest = p
				}
			}
			return &newest, nil
		}
		if terminal && attempts >= 2 {
			return nil, errors.New("run is terminal and its pod is gone")
		}
		if time.Now().After(deadline) {
			return nil, errors.New("timed out waiting for the job pod")
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(podPollPeriod):
		}
	}
}

func (m *LogManager) streamContainer(ctx context.Context, rl *runLogs, podName, container string) {
	buf := rl.containers[container]
	defer buf.finish()
	log := m.log.WithValues("namespace", rl.namespace, "pod", podName, "container", container)

	// Wait until the container has started; GetLogs fails before that.
	for {
		pod, err := m.client.CoreV1().Pods(rl.namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			log.V(1).Info("pod disappeared before logs could be read", "error", err.Error())
			return
		}
		state := containerState(pod, container)
		if state != nil && (state.Running != nil || state.Terminated != nil) {
			break
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(podPollPeriod):
		}
	}

	stop := make(chan struct{})
	go m.flushLoop(ctx, rl, container, buf, stop)
	defer close(stop)

	stream, err := m.client.CoreV1().Pods(rl.namespace).GetLogs(podName, &corev1.PodLogOptions{Container: container, Follow: true}).Stream(ctx)
	if err != nil {
		log.V(1).Info("cannot open log stream", "error", err.Error())
		return
	}
	defer func() { _ = stream.Close() }()
	chunk := make([]byte, 32<<10)
	for {
		n, err := stream.Read(chunk)
		if n > 0 {
			buf.append(chunk[:n])
		}
		if err != nil {
			if !errors.Is(err, io.EOF) && ctx.Err() == nil {
				log.V(1).Info("log stream ended", "error", err.Error())
			}
			return
		}
	}
}

func containerState(pod *corev1.Pod, name string) *corev1.ContainerState {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == name {
			return &status.State
		}
	}
	return nil
}

func (m *LogManager) flushLoop(ctx context.Context, rl *runLogs, container string, buf *logBuffer, stop <-chan struct{}) {
	ticker := time.NewTicker(logFlushPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-stop:
			m.flush(context.Background(), rl, container, buf)
			return
		case <-ticker.C:
			m.flush(ctx, rl, container, buf)
		}
	}
}

func (m *LogManager) flush(ctx context.Context, rl *runLogs, container string, buf *logBuffer) {
	if m.store == nil {
		return
	}
	buf.mu.Lock()
	if !buf.dirty {
		buf.mu.Unlock()
		return
	}
	data := bytes.Clone(buf.data)
	buf.dirty = false
	buf.mu.Unlock()
	if err := m.store.PutLog(ctx, rl.namespace, rl.uid, container, data); err != nil {
		m.log.Error(err, "flush log to history", "run", rl.uid, "container", container)
		buf.mu.Lock()
		buf.dirty = true
		buf.mu.Unlock()
	}
}

func (b *logBuffer) append(p []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = append(b.data, p...)
	if len(b.data) > logBufferMax {
		cut := len(b.data) - logBufferTrim
		if nl := bytes.IndexByte(b.data[cut:], '\n'); nl >= 0 {
			cut += nl + 1
		}
		b.data = append([]byte(nil), b.data[cut:]...)
	}
	b.dirty = true
	for ch := range b.subs {
		select {
		case ch <- bytes.Clone(p):
		default:
		}
	}
}

func (b *logBuffer) finish() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.done = true
	for ch := range b.subs {
		close(ch)
		delete(b.subs, ch)
	}
}

// Get returns the retained log: the live buffer when streaming has started, otherwise the
// stored copy. The bool reports whether the stream is still open.
func (m *LogManager) Get(ctx context.Context, namespace, uid, container string) ([]byte, bool, error) {
	m.mu.Lock()
	rl, ok := m.runs[uid]
	m.mu.Unlock()
	if ok {
		if buf, ok := rl.containers[container]; ok {
			buf.mu.Lock()
			data, done := bytes.Clone(buf.data), buf.done
			buf.mu.Unlock()
			if len(data) > 0 || !done {
				return data, !done, nil
			}
		}
	}
	if m.store == nil {
		return nil, false, history.ErrNotFound
	}
	data, err := m.store.GetLog(ctx, namespace, uid, container)
	return data, false, err
}

// Subscribe returns the current buffer and a channel of further chunks. The channel is
// closed when the stream ends. ok is false when nothing is streaming for the run.
func (m *LogManager) Subscribe(uid, container string) (snapshot []byte, updates <-chan []byte, cancel func(), ok bool) {
	m.mu.Lock()
	rl, found := m.runs[uid]
	m.mu.Unlock()
	if !found {
		return nil, nil, nil, false
	}
	buf, found := rl.containers[container]
	if !found {
		return nil, nil, nil, false
	}
	ch := make(chan []byte, 64)
	buf.mu.Lock()
	snapshot = bytes.Clone(buf.data)
	if buf.done {
		buf.mu.Unlock()
		close(ch)
		return snapshot, ch, func() {}, true
	}
	buf.subs[ch] = struct{}{}
	buf.mu.Unlock()
	cancel = func() {
		buf.mu.Lock()
		if _, still := buf.subs[ch]; still {
			delete(buf.subs, ch)
			close(ch)
		}
		buf.mu.Unlock()
	}
	return snapshot, ch, cancel, true
}
