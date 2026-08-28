---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Operator TLS
subtitle: Configure TLS once at the platform level and auto-inject it into every deployment
---

The Dynamo operator can inject TLS configuration into every
`DynamoGraphDeployment` (DGD) pod automatically, so you don't have to set the
`DYN_TCP_TLS_*` and `NATS_TLS_*` environment variables on each component. TLS
is configured once at the platform level via `InfrastructureConfiguration`, and
the operator propagates the corresponding env vars to all DGD pods it manages.

For the full list of TLS/mTLS environment variables and CLI flags, and for the
per-component configuration method, see the
[TLS reference](../../../../reference/components/tls-configuration.mdx).

## Operator-level TLS configuration

Set the values in the operator Helm chart. When installing the operator as
part of the platform chart, prefix them with `dynamo-operator.`:

```yaml
dynamo-operator:
  tcpTLSCertPath: /etc/certs/server/cert.pem
  tcpTLSKeyPath: /etc/certs/server/key.pem
  tcpTLSCAPath: /etc/certs/ca/ca.pem
  # Override the TLS SNI hostname when dialing by IP to a server whose
  # certificate has a DNS SAN (e.g. a *.svc.cluster.local cert).
  tcpTLSServerName: dynamo-worker.dynamo-system.svc.cluster.local
  natsTLSCAPath: /etc/certs/ca/ca.pem
  # NATS TLS requires a tls:// server address — see the note below.
  natsAddr: "tls://dynamo-platform-nats.dynamo-system.svc.cluster.local:4222"
```

Or pass them via `--set` during `helm install`/`helm upgrade` (platform chart
shown; drop the `dynamo-operator.` prefix if installing the subchart directly):

```bash
helm upgrade dynamo-platform ... \
  --set dynamo-operator.tcpTLSCertPath=/etc/certs/server/cert.pem \
  --set dynamo-operator.tcpTLSKeyPath=/etc/certs/server/key.pem \
  --set dynamo-operator.tcpTLSCAPath=/etc/certs/ca/ca.pem \
  --set dynamo-operator.tcpTLSServerName=dynamo-worker.dynamo-system.svc.cluster.local \
  --set dynamo-operator.natsTLSCAPath=/etc/certs/ca/ca.pem \
  --set dynamo-operator.natsAddr=tls://dynamo-platform-nats.dynamo-system.svc.cluster.local:4222
```

Per-component env vars in `podTemplate` take precedence over operator-level
values when both are set.

> [!NOTE]
> When any `natsTLS*` value is set, `natsAddr` **must** use the
> `tls://` scheme — the runtime fails closed at startup otherwise. If you are
> using the bundled NATS subchart, also enable TLS on the server side (see
> [Enabling TLS on the NATS server](../../../../reference/components/tls-configuration.mdx#enabling-tls-on-the-nats-server)).

## Operator-level mTLS configuration

mTLS certificate paths can also be configured at the operator level:

```yaml
dynamo-operator:
  tcpTLSClientCertPath: /etc/certs/client/cert.pem
  tcpTLSClientKeyPath: /etc/certs/client/key.pem
  tcpTLSClientCAPath: /etc/certs/client-ca/ca.pem
  natsTLSClientCertPath: /etc/certs/client/cert.pem
  natsTLSClientKeyPath: /etc/certs/client/key.pem
```

The certificates themselves are typically delivered by a certificate
management system (such as cert-manager) and mounted into the pods at the
paths referenced above. The operator injects the **paths** (via env vars),
not the volumes — the cert files must exist at those paths in every DGD pod.

A common setup is to issue a `Certificate` with cert-manager, store it in a
Kubernetes `Secret`, and mount that Secret as a volume in the pod template:

```yaml
spec:
  components:
  - name: Frontend
    podTemplate:
      spec:
        containers:
        - name: main
          volumeMounts:
          - name: tls-server-certs
            mountPath: /etc/certs/server
            readOnly: true
          - name: tls-ca-cert
            mountPath: /etc/certs/ca
            readOnly: true
        volumes:
        - name: tls-server-certs
          secret:
            secretName: dynamo-tls-server
            # cert-manager Secrets use tls.crt / tls.key; map them to the
            # filenames the operator's tcpTLSCertPath / tcpTLSKeyPath point at.
            items:
            - key: tls.crt
              path: cert.pem
            - key: tls.key
              path: key.pem
        - name: tls-ca-cert
          secret:
            secretName: dynamo-tls-ca
            # Mount the CA certificate at the path tcpTLSCAPath / natsTLSCAPath
            # point at (e.g. /etc/certs/ca/ca.pem).
            items:
            - key: ca.crt
              path: ca.pem
```

This example shows a single component (`Frontend`); every component that
receives the TLS env vars needs the same volume mounts. If you are using the
operator's auto-injection, apply these mounts in each component's
`podTemplate`.

> [!NOTE]
> For NATS TLS to work, the NATS server itself must also be
> configured to listen on TLS. The operator injects the **client-side** env
> vars, but enabling TLS on the NATS server subchart is a separate step — see
> [Enabling TLS on the NATS server](../../../../reference/components/tls-configuration.mdx#enabling-tls-on-the-nats-server)
> in the TLS reference.
