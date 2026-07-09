/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package config

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	metricsfilters "sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	configapi "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

// fromFile provides an alternative to the deprecated ctrl.ConfigFile().AtPath(path).OfKind(&cfg)
func fromFile(path string, scheme *runtime.Scheme, cfg *configapi.OperatorConfiguration) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)

	// Regardless of if the bytes are of any external version,
	// it will be read successfully and converted into the internal version
	return runtime.DecodeInto(codecs.UniversalDecoder(), content, cfg)
}

// addTo provides an alternative to the deprecated o.AndFrom(&cfg)
func addTo(o *ctrl.Options, cfg *configapi.OperatorConfiguration) {
	// if the enable-http2 flag is false (the default), http/2 should be disabled
	// due to its vulnerabilities. More specifically, disabling http/2 will
	// prevent from being vulnerable to the HTTP/2 Stream Cancellation and
	// Rapid Reset CVEs. For more information see:
	// - https://github.com/advisories/GHSA-qppj-fm5r-hxr3
	// - https://github.com/advisories/GHSA-4374-p667-p6c8
	disableHTTP2 := func(c *tls.Config) {
		c.NextProtos = []string{"http/1.1"}
	}

	tlsOpts := []func(*tls.Config){}
	if !cfg.Security.EnableHTTP2 {
		tlsOpts = append(tlsOpts, disableHTTP2)
	}

	webhookServer := webhook.NewServer(webhook.Options{
		Host:    cfg.Server.Webhook.Host,
		Port:    cfg.Server.Webhook.Port,
		CertDir: cfg.Server.Webhook.CertDir,
		TLSOpts: tlsOpts,
	})

	o.Metrics = metricsserver.Options{
		BindAddress:    fmt.Sprintf("%s:%d", cfg.Server.Metrics.BindAddress, cfg.Server.Metrics.Port),
		SecureServing:  ptr.Deref(cfg.Server.Metrics.Secure, true),
		FilterProvider: metricsfilters.WithAuthenticationAndAuthorization,
		TLSOpts:        tlsOpts,
	}

	o.WebhookServer = webhookServer
	o.HealthProbeBindAddress = fmt.Sprintf("%s:%d", cfg.Server.HealthProbe.BindAddress, cfg.Server.HealthProbe.Port)
	o.LeaderElection = cfg.LeaderElection.Enabled
	o.LeaderElectionID = cfg.LeaderElection.ID
	o.LeaderElectionNamespace = cfg.LeaderElection.Namespace

	restrictedNamespace := cfg.Namespace.Restricted
	if restrictedNamespace != "" {
		o.Cache.DefaultNamespaces = map[string]cache.Config{
			restrictedNamespace: {},
		}
		// PodSnapshotContent is cluster-scoped, so DefaultNamespaces does not cover it.
		// Register it cluster-wide explicitly so the PodSnapshotReconciler can watch it.
		o.Cache.ByObject = map[client.Object]cache.ByObject{
			&nvidiacomv1alpha1.PodSnapshotContent{}: {},
		}
	}
}

// Load returns a set of controller options and configuration from the given file, if the config file path is empty
// it used the default configapi values.
func Load(scheme *runtime.Scheme, configFile string) (ctrl.Options, configapi.OperatorConfiguration, error) {
	var err error
	options := ctrl.Options{
		Scheme: scheme,
	}

	cfg := configapi.OperatorConfiguration{}
	if configFile == "" {
		scheme.Default(&cfg)
	} else {
		err := fromFile(configFile, scheme, &cfg)
		if err != nil {
			return options, cfg, err
		}
	}
	addTo(&options, &cfg)
	return options, cfg, err
}

func Encode(scheme *runtime.Scheme, cfg *configapi.OperatorConfiguration) (string, error) {
	codecs := serializer.NewCodecFactory(scheme)
	const mediaType = runtime.ContentTypeYAML
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return "", fmt.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
	}

	encoder := codecs.EncoderForVersion(info.Serializer, configapi.SchemeGroupVersion)
	buf := new(bytes.Buffer)
	if err := encoder.Encode(cfg, buf); err != nil {
		return "", err
	}
	return buf.String(), nil
}
