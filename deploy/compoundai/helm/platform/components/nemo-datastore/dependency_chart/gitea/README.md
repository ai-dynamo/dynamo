# Gitea Helm Chart <!-- omit from toc -->

- [Introduction](#introduction)
- [Update and versioning policy](#update-and-versioning-policy)
- [Dependencies](#dependencies)
  - [HA Dependencies](#ha-dependencies)
  - [Non-HA Dependencies](#non-ha-dependencies)
  - [Dependency Versioning](#dependency-versioning)
- [Installing](#installing)
- [High Availability](#high-availability)
- [Configuration](#configuration)
  - [Default Configuration](#default-configuration)
    - [Database defaults](#database-defaults)
    - [Server defaults](#server-defaults)
    - [Metrics defaults](#metrics-defaults)
    - [Rootless Defaults](#rootless-defaults)
    - [Session, Cache and Queue](#session-cache-and-queue)
  - [Single-Pod Configurations](#single-pod-configurations)
  - [Additional _app.ini_ settings](#additional-appini-settings)
    - [User defined environment variables in app.ini](#user-defined-environment-variables-in-appini)
  - [External Database](#external-database)
  - [Ports and external url](#ports-and-external-url)
  - [ClusterIP](#clusterip)
  - [SSH and Ingress](#ssh-and-ingress)
  - [SSH on crio based kubernetes cluster](#ssh-on-crio-based-kubernetes-cluster)
  - [Cache](#cache)
  - [Persistence](#persistence)
  - [Admin User](#admin-user)
  - [LDAP Settings](#ldap-settings)
  - [OAuth2 Settings](#oauth2-settings)
- [Configure commit signing](#configure-commit-signing)
- [Metrics and profiling](#metrics-and-profiling)
- [Pod annotations](#pod-annotations)
- [Themes](#themes)
- [Renovate](#renovate)
- [Parameters](#parameters)
  - [Global](#global)
  - [strategy](#strategy)
  - [Image](#image)
  - [Security](#security)
  - [Service](#service)
  - [Ingress](#ingress)
  - [deployment](#deployment)
  - [ServiceAccount](#serviceaccount)
  - [Persistence](#persistence-1)
  - [Init](#init)
  - [Signing](#signing)
  - [Gitea](#gitea)
  - [LivenessProbe](#livenessprobe)
  - [ReadinessProbe](#readinessprobe)
  - [StartupProbe](#startupprobe)
  - [redis-cluster](#redis-cluster)
  - [PostgreSQL-ha](#postgresql-ha)
  - [PostgreSQL](#postgresql)
  - [Advanced](#advanced)
- [Contributing](#contributing)
- [Upgrading](#upgrading)

[Gitea](https://gitea.com) is a community managed lightweight code hosting solution written in Go.
It is published under the MIT license.

## Introduction

This helm chart has taken some inspiration from [jfelten's helm chart](https://github.com/jfelten/gitea-helm-chart).
Yet it takes a completely different approach in providing a database and cache with dependencies.
Additionally, this chart allows to provide LDAP and admin user configuration with values.

## Update and versioning policy

The Gitea helm chart versioning does not follow Gitea's versioning.
The latest chart version can be looked up in [https://dl.gitea.com/charts](https://dl.gitea.com/charts) or in the [repository releases](https://gitea.com/gitea/helm-chart/releases).

The chart aims to follow Gitea's releases closely.
There might be times when the chart is behind the latest Gitea release.
This might be caused by different reasons, most often due to time constraints of the maintainers (remember, all work here is done voluntarily in the spare time of people).
If you're eager to use the latest Gitea version earlier than this chart catches up, then change the tag in `values.yaml` to the latest Gitea version.
Note that besides the exact Gitea version one can also use the `:1` tag to automatically follow the latest Gitea version.
This should be combined with `image.pullPolicy: "Always"`.
Important: Using the `:1` will also automatically jump to new minor release (e.g. from 1.13 to 1.14) which may eventually cause incompatibilities if major/breaking changes happened between these versions.
This is due to Gitea not strictly following [semantic versioning](https://semver.org/#summary) as breaking changes do not increase the major version.
I.e., "minor" version bumps are considered "major".
Yet most often no issues will be encountered and the chart maintainers aim to communicate early/upfront if this would be the case.

## Dependencies

Gitea is most performant when run with an external database and cache.
This chart provides those dependencies via sub-charts.
Users can also configure their own external providers via the configuration.

### HA Dependencies

These dependencies are enabled by default:

- PostgreSQL HA ([Bitnami PostgreSQL-HA](https://github.com/bitnami/charts/blob/main/bitnami/postgresql-ha/Chart.yaml))
- Redis-Cluster ([Bitnami Redis-Cluster](https://github.com/bitnami/charts/blob/main/bitnami/redis-cluster/Chart.yaml))

### Non-HA Dependencies

Alternatively, the following non-HA replacements are available:

- PostgreSQL ([Bitnami PostgreSQL](postgresql](https://github.com/bitnami/charts/blob/main/bitnami/postgresql/Chart.yaml)))

### Dependency Versioning

Updates of sub-charts will be incorporated into the Gitea chart as they are released.
The reasoning behind this is that new users of the chart will start with the most recent sub-chart dependency versions.

**Note** If you want to stay on an older appVersion of a sub-chart dependency (e.g. PostgreSQL), you need to override the image tag in your `values.yaml` file.
In fact, we recommend to do so right from the start to be independent of major sub-chart dependency changes as they are released.
There is no need to update to every new PostgreSQL major version - you can happily skip some and do larger updates when you are ready for them.

We recommend to use a rolling tag like `:<majorVersion>-debian-<debian major version>` to incorporate minor and patch updates for the respective major version as they are released.
Alternatively you can also use a versioning helper tool like [renovate](https://github.com/renovatebot/renovate).

Please double-check the image repository and available tags in the sub-chart:

- [PostgreSQL-HA](https://hub.docker.com/r/bitnami/postgresql-repmgr/tags)
- [PostgreSQL](https://hub.docker.com/r/bitnami/postgresql/tags)
- [Redis Cluster](https://hub.docker.com/r/bitnami/redis-cluster/tags)

and look up the image tag which fits your needs on Dockerhub.

## Installing

```sh
helm repo add gitea-charts https://dl.gitea.com/charts/
helm repo update
helm install gitea gitea-charts/gitea
```

Alternatively, the chart can also be installed from Dockerhub (since v9.6.0)

```sh
helm install gitea oci://registry-1.docker.io/giteacharts/gitea
```

When upgrading, please refer to the [Upgrading](#upgrading) section at the bottom of this document for major and breaking changes.

## High Availability

Since version 9.0.0 this chart supports running Gitea and it's dependencies in HA mode.
Care must be taken for production use as not all implementation details of Gitea core are officially HA-ready yet.

Deploying a HA-ready Gitea instance requires some effort including using HA-ready dependencies.
See the [HA Setup](docs/ha-setup.md) document for more details.

## Configuration

Gitea offers lots of configuration options.
This is fully described in the [Gitea Cheat Sheet](https://docs.gitea.com/administration/config-cheat-sheet).

```yaml
gitea:
  config:
    APP_NAME: "Gitea: With a cup of tea."
    repository:
      ROOT: "~/gitea-repositories"
    repository.pull-request:
      WORK_IN_PROGRESS_PREFIXES: "WIP:,[WIP]:"
```

### Default Configuration

This chart will set a few defaults in the Gitea configuration based on the service and ingress settings.
All defaults can be overwritten in `gitea.config`.

INSTALL_LOCK is always set to true, since we want to configure Gitea with this helm chart and everything is taken care of.

_All default settings are made directly in the generated `app.ini`, not in the Values._

#### Database defaults

If a builtIn database is enabled the database configuration is set automatically.
For example, PostgreSQL builtIn will appear in the `app.ini` as:

```ini
[database]
DB_TYPE = postgres
HOST = RELEASE-NAME-postgresql.default.svc.cluster.local:5432
NAME = gitea
PASSWD = gitea
USER = gitea
```

#### Server defaults

The server defaults are a bit more complex.
If ingress is `enabled`, the `ROOT_URL`, `DOMAIN` and `SSH_DOMAIN` will be set accordingly.
`HTTP_PORT` always defaults to `3000` as well as `SSH_PORT` to `22`.

```ini
[server]
APP_DATA_PATH = /data
DOMAIN = git.example.com
HTTP_PORT = 3000
PROTOCOL = http
ROOT_URL = http://git.example.com
SSH_DOMAIN = git.example.com
SSH_LISTEN_PORT = 22
SSH_PORT = 22
ENABLE_PPROF = false
```

#### Metrics defaults

The Prometheus `/metrics` endpoint is disabled by default.

```ini
[metrics]
ENABLED = false
```

#### Rootless Defaults

If `.Values.image.rootless: true`, then the following will occur. In case you use `.Values.image.fullOverride`, check that this works in your image:

- `$HOME` becomes `/data/gitea/git`

  [see deployment.yaml](./templates/gitea/deployment.yaml) template inside (init-)container "env" declarations

- `START_SSH_SERVER: true` (Unless explicity overwritten by `gitea.config.server.START_SSH_SERVER`)

  [see \_helpers.tpl](./templates/_helpers.tpl) in `gitea.inline_configuration.defaults.server` definition

- `SSH_LISTEN_PORT: 2222` (Unless explicity overwritten by `gitea.config.server.SSH_LISTEN_PORT`)

  [see \_helpers.tpl](./templates/_helpers.tpl) in `gitea.inline_configuration.defaults.server` definition

- `SSH_LOG_LEVEL` environment variable is not injected into the container

  [see deployment.yaml](./templates/gitea/deployment.yaml) template inside container "env" declarations

#### Session, Cache and Queue

The session, cache and queue settings are set to use the built-in Redis Cluster sub-chart dependency.
If Redis Cluster is disabled, the chart will fall back to the Gitea defaults which use "memory" for `session` and `cache` and "level" for `queue`.

While these will work and even not cause immediate issues after startup, **they are not recommended for production use**.
Reasons being that a single pod will take on all the work for `session` and `cache` tasks in its available memory.
It is likely that the pod will run out of memory or will face substantial memory spikes, depending on the workload.
External tools such as `redis-cluster` or `memcached` handle these workloads much better.

### Single-Pod Configurations

If HA is not needed/desired, the following configurations can be used to deploy a single-pod Gitea instance.

1. For a production-ready single-pod Gitea instance without external dependencies (using the chart dependency `postgresql`):

   <details>

   <summary>values.yml</summary>

   ```yaml
   redis-cluster:
     enabled: false
   postgresql:
     enabled: true
   postgresql-ha:
     enabled: false

   persistence:
     enabled: true

   gitea:
     config:
       database:
         DB_TYPE: postgres
       session:
         PROVIDER: db
       cache:
         ADAPTER: memory
       queue:
         TYPE: level
       indexer:
         ISSUE_INDEXER_TYPE: bleve
         REPO_INDEXER_ENABLED: true
   ```

   </details>

2. For a minimal DEV installation (using the built-in sqlite DB instead of Postgres):

   This will result in a single-pod Gitea instance _without any dependencies and persistence_.
   **Do not use this configuration for production use**.

   <details>

   <summary>values.yml</summary>

   ```yaml
   redis-cluster:
     enabled: false
   postgresql:
     enabled: false
   postgresql-ha:
     enabled: false

   persistence:
     enabled: false

   gitea:
     config:
       database:
         DB_TYPE: sqlite3
       session:
         PROVIDER: memory
       cache:
         ADAPTER: memory
       queue:
         TYPE: level
   ```

   </details>

### Additional _app.ini_ settings

> **The [generic](https://docs.gitea.com/administration/config-cheat-sheet#overall-default)
> section cannot be defined that way.**

Some settings inside _app.ini_ (like passwords or whole authentication configurations) must be considered sensitive and therefore should not be passed via plain text inside the _values.yaml_ file.
In times of _GitOps_ the values.yaml could be stored in a Git repository where sensitive data should never be accessible.

The Helm Chart supports this approach and let the user define custom sources like
Kubernetes Secrets to be loaded as environment variables during _app.ini_ creation or update.

```yaml
gitea:
  additionalConfigSources:
    - secret:
        secretName: gitea-app-ini-oauth
    - configMap:
        name: gitea-app-ini-plaintext
```

This would mount the two additional volumes (`oauth` and `some-additionals`) from different sources to the init container where the _app.ini_ gets updated.
All files mounted that way will be read and converted to environment variables and then added to the _app.ini_ using [environment-to-ini](https://github.com/go-gitea/gitea/tree/main/contrib/environment-to-ini).

The key of such additional source represents the section inside the _app.ini_.
The value for each key can be multiline ini-like definitions.

In example, the referenced `gitea-app-ini-plaintext` could look like this.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gitea-app-ini-plaintext
data:
  session: |
    PROVIDER=memory
    SAME_SITE=strict
  cron.archive_cleanup: |
    ENABLED=true
```

Or when using a Kubernetes secret, having the same data structure:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gitea-security-related-configuration
type: Opaque
stringData:
  security: |
    PASSWORD_COMPLEXITY=off
  session: |
    SAME_SITE=strict
```

#### User defined environment variables in app.ini

Users are able to define their own environment variables, which are loaded into the containers.
We also support to directly interact with the generated _app.ini_.

To inject self defined variables into the _app.ini_ a certain format needs to be honored.
This is described in detail on the [env-to-ini](https://github.com/go-gitea/gitea/tree/main/contrib/environment-to-ini) page.

Prior to Gitea 1.20 and Chart 9.0.0 the helm chart had a custom prefix `ENV_TO_INI`.
After the support for a custom prefix was removed in Gite core, the prefix was changed to `GITEA`.

For example a database setting needs to have the following format:

```yaml
gitea:
  additionalConfigFromEnvs:
    - name: GITEA__DATABASE__HOST
      value: my.own.host
    - name: GITEA__DATABASE__PASSWD
      valueFrom:
        secretKeyRef:
          name: postgres-secret
          key: password
```

Priority (highest to lowest) for defining app.ini variables:

1. Environment variables prefixed with `GITEA`
1. Additional config sources
1. Values defined in `gitea.config`

### External Database

Any external database listed in [https://docs.gitea.com/installation/database-prep](https://docs.gitea.com/installation/database-prep) can be used instead of the built-in PostgreSQL.
In fact, it is **highly recommended** to use an external database to ensure a stable Gitea installation longterm.

If an external database is used, no matter which type, make sure to set `postgresql.enabled` to `false` to disable the use of the built-in PostgreSQL.

```yaml
gitea:
  config:
    database:
      DB_TYPE: mysql
      HOST: <mysql HOST>
      NAME: gitea
      USER: root
      PASSWD: gitea
      SCHEMA: gitea

postgresql:
  enabled: false
```

### Ports and external url

By default port `3000` is used for web traffic and `22` for ssh.
Those can be changed:

```yaml
service:
  http:
    port: 3000
  ssh:
    port: 22
```

This helm chart automatically configures the clone urls to use the correct ports.
You can change these ports by hand using the `gitea.config` dict.
However you should know what you're doing.

### ClusterIP

By default the `clusterIP` will be set to `None`, which is the default for headless services.
However if you want to omit the clusterIP field in the service, use the following values:

```yaml
service:
  http:
    type: ClusterIP
    port: 3000
    clusterIP:
  ssh:
    type: ClusterIP
    port: 22
    clusterIP:
```

### SSH and Ingress

If you're using ingress and want to use SSH, keep in mind, that ingress is not able to forward SSH Ports.
You will need a LoadBalancer like `metallb` and a setting in your ssh service annotations.

```yaml
service:
  ssh:
    annotations:
      metallb.universe.tf/allow-shared-ip: test
```

### SSH on crio based kubernetes cluster

If you use `crio` as container runtime it is not possible to read from a remote repository.
You should get an error message like this:

```bash
$ git clone git@k8s-demo.internal:admin/test.git
Cloning into 'test'...
Connection reset by 192.168.179.217 port 22
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
```

To solve this problem add the capability `SYS_CHROOT` to the `securityContext`.
More about this issue [here](https://gitea.com/gitea/helm-chart/issues/161).

### Cache

The cache handling is done via `redis-cluster` (via the `bitnami` chart) by default.
This deployment is HA-ready but can also be used for single-pod deployments.
By default, 6 replicas are deployed for a working `redis-cluster` deployment.
Many cloud providers offer a managed redis service, which can be used instead of the built-in `redis-cluster`.

```yaml
redis-cluster:
  enabled: true
```

### Persistence

Gitea will be deployed as a deployment.
By simply enabling the persistence and setting the storage class according to your cluster everything else will be taken care of.
The following example will create a PVC as a part of the deployment.

Please note, that an empty `storageClass` in the persistence will result in kubernetes using your default storage class.

If you want to use your own storage class define it as follows:

```yaml
persistence:
  enabled: true
  storageClass: myOwnStorageClass
```

If you want to manage your own PVC you can simply pass the PVC name to the chart.

```yaml
persistence:
  enabled: true
  claimName: MyAwesomeGiteaClaim
```

In case that persistence has been disabled it will simply use an empty dir volume.

PostgreSQL handles the persistence in the exact same way.
You can interact with the postgres settings as displayed in the following example:

```yaml
postgresql:
  persistence:
    enabled: true
    claimName: MyAwesomeGiteaPostgresClaim
```

### Admin User

This chart enables you to create a default admin user.
It is also possible to update the password for this user by upgrading or redeploying the chart.
It is not possible to delete an admin user after it has been created.
This has to be done in the ui.
You cannot use `admin` as username.

```yaml
gitea:
  admin:
    username: "MyAwesomeGiteaAdmin"
    password: "AReallyAwesomeGiteaPassword"
    email: "gi@tea.com"
```

You can also use an existing Secret to configure the admin user:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gitea-admin-secret
type: Opaque
stringData:
  username: MyAwesomeGiteaAdmin
  password: AReallyAwesomeGiteaPassword
```

```yaml
gitea:
  admin:
    existingSecret: gitea-admin-secret
```

### LDAP Settings

Like the admin user the LDAP settings can be updated.
All LDAP values from <https://docs.gitea.com/administration/command-line#admin> are available.

Multiple LDAP sources can be configured with additional LDAP list items.

```yaml
gitea:
  ldap:
    - name: MyAwesomeGiteaLdap
      securityProtocol: unencrypted
      host: "127.0.0.1"
      port: "389"
      userSearchBase: ou=Users,dc=example,dc=com
      userFilter: sAMAccountName=%s
      adminFilter: CN=Admin,CN=Group,DC=example,DC=com
      emailAttribute: mail
      bindDn: CN=ldap read,OU=Spezial,DC=example,DC=com
      bindPassword: JustAnotherBindPw
      usernameAttribute: CN
      publicSSHKeyAttribute: publicSSHKey
```

You can also use an existing secret to set the `bindDn` and `bindPassword`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gitea-ldap-secret
type: Opaque
stringData:
  bindDn: CN=ldap read,OU=Spezial,DC=example,DC=com
  bindPassword: JustAnotherBindPw
```

```yaml
gitea:
    ldap:
      - existingSecret: gitea-ldap-secret
        ...
```

⚠️ Some options are just flags and therefore don't have any values.
If they are defined in `gitea.ldap` configuration, they will be passed to the Gitea CLI without any value.
Affected options:

- notActive
- skipTlsVerify
- allowDeactivateAll
- synchronizeUsers
- attributesInBind

### OAuth2 Settings

Like the admin user, OAuth2 settings can be updated and disabled but not deleted.
Deleting OAuth2 settings has to be done in the ui.
All OAuth2 values, which are documented [here](https://docs.gitea.com/administration/command-line#admin), are
available.

Multiple OAuth2 sources can be configured with additional OAuth list items.

```yaml
gitea:
  oauth:
    - name: "MyAwesomeGiteaOAuth"
      provider: "openidConnect"
      key: "hello"
      secret: "world"
      autoDiscoverUrl: "https://gitea.example.com/.well-known/openid-configuration"
      #useCustomUrls:
      #customAuthUrl:
      #customTokenUrl:
      #customProfileUrl:
      #customEmailUrl:
```

You can also use an existing secret to set the `key` and `secret`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gitea-oauth-secret
type: Opaque
stringData:
  key: hello
  secret: world
```

```yaml
gitea:
  oauth:
    - name: "MyAwesomeGiteaOAuth"
      existingSecret: gitea-oauth-secret
        ...
```

## Configure commit signing

When using the rootless image the gpg key folder is not persistent by default.
If you consider using signed commits for internal Gitea activities (e.g. initial commit), you'd need to provide a signing key.
Prior to [PR186](https://gitea.com/gitea/helm-chart/pulls/186), imported keys had to be re-imported once the container got replaced by another.

The mentioned PR introduced a new configuration object `signing` allowing you to configure prerequisites for commit signing.
By default this section is disabled to maintain backwards compatibility.

```yaml
signing:
  enabled: false
  gpgHome: /data/git/.gnupg
```

Regardless of the used container image the `signing` object allows to specify a private gpg key.
Either using the `signing.privateKey` to define the key inline, or refer to an existing secret containing the key data by using `signing.existingSecret`.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: custom-gitea-gpg-key
type: Opaque
stringData:
  privateKey: |-
    -----BEGIN PGP PRIVATE KEY BLOCK-----
    ...
    -----END PGP PRIVATE KEY BLOCK-----
```

```yaml
signing:
  existingSecret: custom-gitea-gpg-key
```

To use the gpg key, Gitea needs to be configured accordingly.
A detailed description can be found in the [official Gitea documentation](https://docs.gitea.com/administration/signing#general-configuration).

## Metrics and profiling

A Prometheus `/metrics` endpoint on the `HTTP_PORT` and `pprof` profiling endpoints on port 6060 can be enabled under `gitea`.
Beware that the metrics endpoint is exposed via the ingress, manage access using ingress annotations for example.

To deploy the `ServiceMonitor`, you first need to ensure that you have deployed `prometheus-operator` and its [CRDs](https://github.com/prometheus-operator/prometheus-operator#customresourcedefinitions).

```yaml
gitea:
  metrics:
    enabled: true
    serviceMonitor:
      enabled: true

  config:
    server:
      ENABLE_PPROF: true
```

## Pod annotations

Annotations can be added to the Gitea pod.

```yaml
gitea:
  podAnnotations: {}
```

## Themes

Custom themes can be added via k8s secrets and referencing them in `values.yaml`.

The [http provider](https://registry.terraform.io/providers/hashicorp/http/latest/docs/data-sources/http) is useful here.

```yaml
extraVolumes:
  - name: gitea-themes
    secret:
      secretName: gitea-themes

extraVolumeMounts:
  - name: gitea-themes
    readOnly: true
    mountPath: "/data/gitea/public/assets/css"
```

The secret can be created via `terraform`:

```hcl
resource "kubernetes_secret" "gitea-themes" {
  metadata {
    name      = "gitea-themes"
    namespace = "gitea"
  }

  data = {
    "my-theme.css"      = data.http.gitea-theme-light.body
    "my-theme-dark.css" = data.http.gitea-theme-dark.body
    "my-theme-auto.css" = data.http.gitea-theme-auto.body
  }

  type = "Opaque"
}


data "http" "gitea-theme-light" {
  url = "<raw theme url>"

  request_headers = {
    Accept = "application/json"
  }
}

data "http" "gitea-theme-dark" {
  url = "<raw theme url>"

  request_headers = {
    Accept = "application/json"
  }
}

data "http" "gitea-theme-auto" {
  url = "<raw theme url>"

  request_headers = {
    Accept = "application/json"
  }
}
```

or natively via `kubectl`:

```bash
kubectl create secret generic gitea-themes --from-file={{FULL-PATH-TO-CSS}} --namespace gitea
```

## Renovate

To be able to use a digest value which is automatically updated by `Renovate` a [customManager](https://docs.renovatebot.com/modules/manager/regex/) is required.
Here's an examplary `values.yml` definition which makes use of a digest:

```yaml
image:
  repository: gitea/gitea
  tag: 1.20.2
  digest: sha256:6e3b85a36653894d6741d0aefb41dfaac39044e028a42e0a520cc05ebd7bfc3f
```

By default Renovate adds digest after the `tag`.
To comply with the Gitea helm chart definition of the digest parameter, a "customManagers" definition is required:

```json
"customManagers": [
  {
    "customType": "regex",
    "description": "Apply an explicit gitea digest field match",
    "fileMatch": ["values\\.ya?ml"],
    "matchStrings": ["(?<depName>gitea\\/gitea)\\n(?<indentation>\\s+)tag: (?<currentValue>[^@].*?)\\n\\s+digest: (?<currentDigest>sha256:[a-f0-9]+)"],
    "datasourceTemplate": "docker",
    "autoReplaceStringTemplate": "{{depName}}\n{{indentation}}tag: {{newValue}}\n{{indentation}}digest: {{#if newDigest}}{{{newDigest}}}{{else}}{{{currentDigest}}}{{/if}}"
  }
]
```

## Parameters

### Global

| Name                      | Description                                                               | Value |
| ------------------------- | ------------------------------------------------------------------------- | ----- |
| `global.imageRegistry`    | global image registry override                                            | `""`  |
| `global.imagePullSecrets` | global image pull secrets override; can be extended by `imagePullSecrets` | `[]`  |
| `global.storageClass`     | global storage class override                                             | `""`  |
| `global.hostAliases`      | global hostAliases which will be added to the pod's hosts files           | `[]`  |
| `replicaCount`            | number of replicas for the deployment                                     | `1`   |

### strategy

| Name                                    | Description    | Value           |
| --------------------------------------- | -------------- | --------------- |
| `strategy.type`                         | strategy type  | `RollingUpdate` |
| `strategy.rollingUpdate.maxSurge`       | maxSurge       | `100%`          |
| `strategy.rollingUpdate.maxUnavailable` | maxUnavailable | `0`             |
| `clusterDomain`                         | cluster domain | `cluster.local` |

### Image

| Name                 | Description                                                                                                                                                      | Value          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `image.registry`     | image registry, e.g. gcr.io,docker.io                                                                                                                            | `""`           |
| `image.repository`   | Image to start for this pod                                                                                                                                      | `gitea/gitea`  |
| `image.tag`          | Visit: [Image tag](https://hub.docker.com/r/gitea/gitea/tags?page=1&ordering=last_updated). Defaults to `appVersion` within Chart.yaml.                          | `""`           |
| `image.digest`       | Image digest. Allows to pin the given image tag. Useful for having control over mutable tags like `latest`                                                       | `""`           |
| `image.pullPolicy`   | Image pull policy                                                                                                                                                | `IfNotPresent` |
| `image.rootless`     | Wether or not to pull the rootless version of Gitea, only works on Gitea 1.14.x or higher                                                                        | `true`         |
| `image.fullOverride` | Completely overrides the image registry, path/image, tag and digest. **Adjust `image.rootless` accordingly and review [Rootless defaults](#rootless-defaults).** | `""`           |
| `imagePullSecrets`   | Secret to use for pulling the image                                                                                                                              | `[]`           |

### Security

| Name                         | Description                                                     | Value  |
| ---------------------------- | --------------------------------------------------------------- | ------ |
| `podSecurityContext.fsGroup` | Set the shared file system group for all containers in the pod. | `1000` |
| `containerSecurityContext`   | Security context                                                | `{}`   |
| `securityContext`            | Run init and Gitea containers as a specific securityContext     | `{}`   |
| `podDisruptionBudget`        | Pod disruption budget                                           | `{}`   |

### Service

| Name                                    | Description                                                                                                                                                                                          | Value       |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `service.http.type`                     | Kubernetes service type for web traffic                                                                                                                                                              | `ClusterIP` |
| `service.http.port`                     | Port number for web traffic                                                                                                                                                                          | `3000`      |
| `service.http.clusterIP`                | ClusterIP setting for http autosetup for deployment is None                                                                                                                                          | `None`      |
| `service.http.loadBalancerIP`           | LoadBalancer IP setting                                                                                                                                                                              | `nil`       |
| `service.http.nodePort`                 | NodePort for http service                                                                                                                                                                            | `nil`       |
| `service.http.externalTrafficPolicy`    | If `service.http.type` is `NodePort` or `LoadBalancer`, set this to `Local` to enable source IP preservation                                                                                         | `nil`       |
| `service.http.externalIPs`              | External IPs for service                                                                                                                                                                             | `nil`       |
| `service.http.ipFamilyPolicy`           | HTTP service dual-stack policy                                                                                                                                                                       | `nil`       |
| `service.http.ipFamilies`               | HTTP service dual-stack familiy selection,for dual-stack parameters see official kubernetes [dual-stack concept documentation](https://kubernetes.io/docs/concepts/services-networking/dual-stack/). | `nil`       |
| `service.http.loadBalancerSourceRanges` | Source range filter for http loadbalancer                                                                                                                                                            | `[]`        |
| `service.http.annotations`              | HTTP service annotations                                                                                                                                                                             | `{}`        |
| `service.http.labels`                   | HTTP service additional labels                                                                                                                                                                       | `{}`        |
| `service.ssh.type`                      | Kubernetes service type for ssh traffic                                                                                                                                                              | `ClusterIP` |
| `service.ssh.port`                      | Port number for ssh traffic                                                                                                                                                                          | `22`        |
| `service.ssh.clusterIP`                 | ClusterIP setting for ssh autosetup for deployment is None                                                                                                                                           | `None`      |
| `service.ssh.loadBalancerIP`            | LoadBalancer IP setting                                                                                                                                                                              | `nil`       |
| `service.ssh.nodePort`                  | NodePort for ssh service                                                                                                                                                                             | `nil`       |
| `service.ssh.externalTrafficPolicy`     | If `service.ssh.type` is `NodePort` or `LoadBalancer`, set this to `Local` to enable source IP preservation                                                                                          | `nil`       |
| `service.ssh.externalIPs`               | External IPs for service                                                                                                                                                                             | `nil`       |
| `service.ssh.ipFamilyPolicy`            | SSH service dual-stack policy                                                                                                                                                                        | `nil`       |
| `service.ssh.ipFamilies`                | SSH service dual-stack familiy selection,for dual-stack parameters see official kubernetes [dual-stack concept documentation](https://kubernetes.io/docs/concepts/services-networking/dual-stack/).  | `nil`       |
| `service.ssh.hostPort`                  | HostPort for ssh service                                                                                                                                                                             | `nil`       |
| `service.ssh.loadBalancerSourceRanges`  | Source range filter for ssh loadbalancer                                                                                                                                                             | `[]`        |
| `service.ssh.annotations`               | SSH service annotations                                                                                                                                                                              | `{}`        |
| `service.ssh.labels`                    | SSH service additional labels                                                                                                                                                                        | `{}`        |

### Ingress

| Name                                 | Description                                                                 | Value             |
| ------------------------------------ | --------------------------------------------------------------------------- | ----------------- |
| `ingress.enabled`                    | Enable ingress                                                              | `false`           |
| `ingress.className`                  | Ingress class name                                                          | `nil`             |
| `ingress.annotations`                | Ingress annotations                                                         | `{}`              |
| `ingress.hosts[0].host`              | Default Ingress host                                                        | `git.example.com` |
| `ingress.hosts[0].paths[0].path`     | Default Ingress path                                                        | `/`               |
| `ingress.hosts[0].paths[0].pathType` | Ingress path type                                                           | `Prefix`          |
| `ingress.tls`                        | Ingress tls settings                                                        | `[]`              |
| `ingress.apiVersion`                 | Specify APIVersion of ingress object. Mostly would only be used for argocd. |                   |

### deployment

| Name                                       | Description                                            | Value |
| ------------------------------------------ | ------------------------------------------------------ | ----- |
| `resources`                                | Kubernetes resources                                   | `{}`  |
| `schedulerName`                            | Use an alternate scheduler, e.g. "stork"               | `""`  |
| `nodeSelector`                             | NodeSelector for the deployment                        | `{}`  |
| `tolerations`                              | Tolerations for the deployment                         | `[]`  |
| `affinity`                                 | Affinity for the deployment                            | `{}`  |
| `topologySpreadConstraints`                | TopologySpreadConstraints for the deployment           | `[]`  |
| `dnsConfig`                                | dnsConfig for the deployment                           | `{}`  |
| `priorityClassName`                        | priorityClassName for the deployment                   | `""`  |
| `deployment.env`                           | Additional environment variables to pass to containers | `[]`  |
| `deployment.terminationGracePeriodSeconds` | How long to wait until forcefully kill the pod         | `60`  |
| `deployment.labels`                        | Labels for the deployment                              | `{}`  |
| `deployment.annotations`                   | Annotations for the Gitea deployment to be created     | `{}`  |

### ServiceAccount

| Name                                          | Description                                                                                                                               | Value   |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `serviceAccount.create`                       | Enable the creation of a ServiceAccount                                                                                                   | `false` |
| `serviceAccount.name`                         | Name of the created ServiceAccount, defaults to release name. Can also link to an externally provided ServiceAccount that should be used. | `""`    |
| `serviceAccount.automountServiceAccountToken` | Enable/disable auto mounting of the service account token                                                                                 | `false` |
| `serviceAccount.imagePullSecrets`             | Image pull secrets, available to the ServiceAccount                                                                                       | `[]`    |
| `serviceAccount.annotations`                  | Custom annotations for the ServiceAccount                                                                                                 | `{}`    |
| `serviceAccount.labels`                       | Custom labels for the ServiceAccount                                                                                                      | `{}`    |

### Persistence

| Name                                              | Description                                                                                           | Value                  |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------- |
| `persistence.enabled`                             | Enable persistent storage                                                                             | `true`                 |
| `persistence.create`                              | Whether to create the persistentVolumeClaim for shared storage                                        | `true`                 |
| `persistence.mount`                               | Whether the persistentVolumeClaim should be mounted (even if not created)                             | `true`                 |
| `persistence.claimName`                           | Use an existing claim to store repository information                                                 | `gitea-shared-storage` |
| `persistence.size`                                | Size for persistence to store repo information                                                        | `10Gi`                 |
| `persistence.accessModes`                         | AccessMode for persistence                                                                            | `["ReadWriteOnce"]`    |
| `persistence.labels`                              | Labels for the persistence volume claim to be created                                                 | `{}`                   |
| `persistence.annotations.helm.sh/resource-policy` | Resource policy for the persistence volume claim                                                      | `keep`                 |
| `persistence.storageClass`                        | Name of the storage class to use                                                                      | `nil`                  |
| `persistence.subPath`                             | Subdirectory of the volume to mount at                                                                | `nil`                  |
| `persistence.volumeName`                          | Name of persistent volume in PVC                                                                      | `""`                   |
| `extraVolumes`                                    | Additional volumes to mount to the Gitea deployment                                                   | `[]`                   |
| `extraContainerVolumeMounts`                      | Mounts that are only mapped into the Gitea runtime/main container, to e.g. override custom templates. | `[]`                   |
| `extraInitVolumeMounts`                           | Mounts that are only mapped into the init-containers. Can be used for additional preconfiguration.    | `[]`                   |
| `extraVolumeMounts`                               | **DEPRECATED** Additional volume mounts for init containers and the Gitea main container              | `[]`                   |

### Init

| Name                                       | Description                                                                          | Value   |
| ------------------------------------------ | ------------------------------------------------------------------------------------ | ------- |
| `initPreScript`                            | Bash shell script copied verbatim to the start of the init-container.                | `""`    |
| `initContainers.resources.limits`          | initContainers.limits Kubernetes resource limits for init containers                 | `{}`    |
| `initContainers.resources.requests.cpu`    | initContainers.requests.cpu Kubernetes cpu resource limits for init containers       | `100m`  |
| `initContainers.resources.requests.memory` | initContainers.requests.memory Kubernetes memory resource limits for init containers | `128Mi` |

### Signing

| Name                     | Description                                                       | Value              |
| ------------------------ | ----------------------------------------------------------------- | ------------------ |
| `signing.enabled`        | Enable commit/action signing                                      | `false`            |
| `signing.gpgHome`        | GPG home directory                                                | `/data/git/.gnupg` |
| `signing.privateKey`     | Inline private gpg key for signed Gitea actions                   | `""`               |
| `signing.existingSecret` | Use an existing secret to store the value of `signing.privateKey` | `""`               |

### Gitea

| Name                                   | Description                                                               | Value                |
| -------------------------------------- | ------------------------------------------------------------------------- | -------------------- |
| `gitea.admin.username`                 | Username for the Gitea admin user                                         | `gitea_admin`        |
| `gitea.admin.existingSecret`           | Use an existing secret to store admin user credentials                    | `nil`                |
| `gitea.admin.password`                 | Password for the Gitea admin user                                         | `r8sA8CPHD9!bt6d`    |
| `gitea.admin.email`                    | Email for the Gitea admin user                                            | `gitea@local.domain` |
| `gitea.metrics.enabled`                | Enable Gitea metrics                                                      | `false`              |
| `gitea.metrics.serviceMonitor.enabled` | Enable Gitea metrics service monitor                                      | `false`              |
| `gitea.ldap`                           | LDAP configuration                                                        | `[]`                 |
| `gitea.oauth`                          | OAuth configuration                                                       | `[]`                 |
| `gitea.config.server.SSH_PORT`         | SSH port for rootlful Gitea image                                         | `22`                 |
| `gitea.config.server.SSH_LISTEN_PORT`  | SSH port for rootless Gitea image                                         | `2222`               |
| `gitea.additionalConfigSources`        | Additional configuration from secret or configmap                         | `[]`                 |
| `gitea.additionalConfigFromEnvs`       | Additional configuration sources from environment variables               | `[]`                 |
| `gitea.podAnnotations`                 | Annotations for the Gitea pod                                             | `{}`                 |
| `gitea.ssh.logLevel`                   | Configure OpenSSH's log level. Only available for root-based Gitea image. | `INFO`               |

### LivenessProbe

| Name                                      | Description                                      | Value  |
| ----------------------------------------- | ------------------------------------------------ | ------ |
| `gitea.livenessProbe.enabled`             | Enable liveness probe                            | `true` |
| `gitea.livenessProbe.tcpSocket.port`      | Port to probe for liveness                       | `http` |
| `gitea.livenessProbe.initialDelaySeconds` | Initial delay before liveness probe is initiated | `200`  |
| `gitea.livenessProbe.timeoutSeconds`      | Timeout for liveness probe                       | `1`    |
| `gitea.livenessProbe.periodSeconds`       | Period for liveness probe                        | `10`   |
| `gitea.livenessProbe.successThreshold`    | Success threshold for liveness probe             | `1`    |
| `gitea.livenessProbe.failureThreshold`    | Failure threshold for liveness probe             | `10`   |

### ReadinessProbe

| Name                                       | Description                                       | Value  |
| ------------------------------------------ | ------------------------------------------------- | ------ |
| `gitea.readinessProbe.enabled`             | Enable readiness probe                            | `true` |
| `gitea.readinessProbe.tcpSocket.port`      | Port to probe for readiness                       | `http` |
| `gitea.readinessProbe.initialDelaySeconds` | Initial delay before readiness probe is initiated | `5`    |
| `gitea.readinessProbe.timeoutSeconds`      | Timeout for readiness probe                       | `1`    |
| `gitea.readinessProbe.periodSeconds`       | Period for readiness probe                        | `10`   |
| `gitea.readinessProbe.successThreshold`    | Success threshold for readiness probe             | `1`    |
| `gitea.readinessProbe.failureThreshold`    | Failure threshold for readiness probe             | `3`    |

### StartupProbe

| Name                                     | Description                                     | Value   |
| ---------------------------------------- | ----------------------------------------------- | ------- |
| `gitea.startupProbe.enabled`             | Enable startup probe                            | `false` |
| `gitea.startupProbe.tcpSocket.port`      | Port to probe for startup                       | `http`  |
| `gitea.startupProbe.initialDelaySeconds` | Initial delay before startup probe is initiated | `60`    |
| `gitea.startupProbe.timeoutSeconds`      | Timeout for startup probe                       | `1`     |
| `gitea.startupProbe.periodSeconds`       | Period for startup probe                        | `10`    |
| `gitea.startupProbe.successThreshold`    | Success threshold for startup probe             | `1`     |
| `gitea.startupProbe.failureThreshold`    | Failure threshold for startup probe             | `10`    |

### redis-cluster

| Name                             | Description                                  | Value   |
| -------------------------------- | -------------------------------------------- | ------- |
| `redis-cluster.enabled`          | Enable redis                                 | `true`  |
| `redis-cluster.usePassword`      | Whether to use password authentication       | `false` |
| `redis-cluster.cluster.nodes`    | Number of redis cluster master nodes         | `3`     |
| `redis-cluster.cluster.replicas` | Number of redis cluster master node replicas | `0`     |

### PostgreSQL-ha

| Name                                        | Description                                                      | Value       |
| ------------------------------------------- | ---------------------------------------------------------------- | ----------- |
| `postgresql-ha.enabled`                     | Enable PostgreSQL-ha                                             | `true`      |
| `postgresql-ha.postgresql.password`         | Password for the `gitea` user (overrides `auth.password`)        | `changeme4` |
| `postgresql-ha.global.postgresql.database`  | Name for a custom database to create (overrides `auth.database`) | `gitea`     |
| `postgresql-ha.global.postgresql.username`  | Name for a custom user to create (overrides `auth.username`)     | `gitea`     |
| `postgresql-ha.global.postgresql.password`  | Name for a custom password to create (overrides `auth.password`) | `gitea`     |
| `postgresql-ha.postgresql.repmgrPassword`   | Repmgr Password                                                  | `changeme2` |
| `postgresql-ha.postgresql.postgresPassword` | postgres Password                                                | `changeme1` |
| `postgresql-ha.pgpool.adminPassword`        | pgpool adminPassword                                             | `changeme3` |
| `postgresql-ha.service.ports.postgresql`    | PostgreSQL service port (overrides `service.ports.postgresql`)   | `5432`      |
| `postgresql-ha.primary.persistence.size`    | PVC Storage Request for PostgreSQL-ha volume                     | `10Gi`      |

### PostgreSQL

| Name                                                    | Description                                                      | Value   |
| ------------------------------------------------------- | ---------------------------------------------------------------- | ------- |
| `postgresql.enabled`                                    | Enable PostgreSQL                                                | `false` |
| `postgresql.global.postgresql.auth.password`            | Password for the `gitea` user (overrides `auth.password`)        | `gitea` |
| `postgresql.global.postgresql.auth.database`            | Name for a custom database to create (overrides `auth.database`) | `gitea` |
| `postgresql.global.postgresql.auth.username`            | Name for a custom user to create (overrides `auth.username`)     | `gitea` |
| `postgresql.global.postgresql.service.ports.postgresql` | PostgreSQL service port (overrides `service.ports.postgresql`)   | `5432`  |
| `postgresql.primary.persistence.size`                   | PVC Storage Request for PostgreSQL volume                        | `10Gi`  |

### Advanced

| Name               | Description                                                        | Value     |
| ------------------ | ------------------------------------------------------------------ | --------- |
| `checkDeprecation` | Set it to false to skip this basic validation check.               | `true`    |
| `test.enabled`     | Set it to false to disable test-connection Pod.                    | `true`    |
| `test.image.name`  | Image name for the wget container used in the test-connection Pod. | `busybox` |
| `test.image.tag`   | Image tag for the wget container used in the test-connection Pod.  | `latest`  |
| `extraDeploy`      | Array of extra objects to deploy with the release                  | `[]`      |

## Contributing

Expected workflow is: Fork -> Patch -> Push -> Pull Request

See [CONTRIBUTORS GUIDE](CONTRIBUTING.md) for details.

## Upgrading

This section lists major and breaking changes of each Helm Chart version.
Please read them carefully to upgrade successfully, especially the change of the **default database backend**!
If you miss this, blindly upgrading may delete your Postgres instance and you may lose your data!

<details>

<summary>To 10.0.0</summary>

<!-- prettier-ignore-start -->
<!-- markdownlint-disable-next-line -->
**Breaking changes**
<!-- prettier-ignore-end -->

- Update PostgreSQL sub-chart dependencies to appVersion 16.x
- Update to sub-charts versioning approach: Users are encouraged to pin the version tag of the sub-chart dependencies to a major appVersion.
  This avoids issues during chart upgrades and allows to incorporate new sub-chart versions as they are released.
  Please see the new [README section describing the versioning approach for sub-chart versions](#dependency-versioning).

</details>

<details>

<summary>To 9.6.0</summary>

Chart 9.6.0 ships with Gitea 1.21.0.
While there are no breaking changes in the chart, please check the changes of the [1.21 release blog post](https://blog.gitea.com/release-of-1.21.0/).

</details>

<details>

<summary>To 9.0.0</summary>

This chart release comes with many breaking changes while aiming for a HA-ready setup.
Please go through all of them carefully to perform a successful upgrade.
Here's a brief summary again, followed by more detailed migration instructions:

- Switch from `Statefulset` to `Deployment`
- Switch from `Memcached` to `redis-cluster` as the default session and queue provider
- Switch from `postgres` to `postgres-ha` as the default database provider
- A chart-internal PVC bootstrapping logic
  - New `persistence.mount`: whether to mount an existent PVC (even if not creating it)
  - New `persistence.create`: whether to create a new PVC
  - Renamed `persistence.existingClaim` to `persistence.claimName`

While not required, we recommend to start with a RWX PV for new installations.
A RWX volume is required for installation aiming for HA.

If you want to stay with a pre-existing RWO PV, you need to set

- `persistence.mount=true`
- `persistence.create=false`
- `persistence.claimName` to the name of your existing PVC.

If you do not, Gitea will create a new PVC which will in turn create a new PV.
If this happened to you by accident, you can still recover your data by setting using the settings from above in a subsequent run.

If you want to stay with a `memcache` instead of `redis-cluster`, you need to deploy `memcache` manually (e.g. from [bitnami](https://github.com/bitnami/charts/tree/main/bitnami/memcached)) and set

- `cache.HOST = "<memcache connection string>"`
- `cache.ADAPTER = "memcache"`
- `session.PROVIDER = "memcache"`
- `session.PROVIDER_CONFIG = "<memcache connection string>"`
- `queue.TYPE = "memcache"`
- `queue.CONN_STR = "<memcache connection string>"`

The `memcache` connection string has the scheme `memcache://<memcache service name>:<memcache service port>`, e.g. `gitea-memcached.gitea.svc.cluster.local:11211`.
The first item here (`<memcache service name>`) will be different compared to the example if you deploy `memcache` yourself.

The above changes are motivated by the idea to tidy dependencies but also have HA-ready ones at the same time.
The previous `memcache` default was not HA-ready, hence we decided to switch to `redis-cluster` by default.

If you are coming from an existing deployment and [#356](https://gitea.com/gitea/helm-chart/issues/356) is still open, you need to set the config sections for `cache`, `session` and `queue` explicitly:

```yaml
gitea:
  config:
    session:
      PROVIDER: redis-cluster
      PROVIDER_CONFIG: redis+cluster://:gitea@gitea-redis-cluster-headless.<namespace>.svc.cluster.local:6379/0?pool_size=100&idle_timeout=180s&

    cache:
      ENABLED: true
      ADAPTER: redis-cluster
      HOST: redis+cluster://:gitea@gitea-redis-cluster-headless.<namespace>.svc.cluster.local:6379/0?pool_size=100&idle_timeout=180s&

    queue:
      TYPE: redis
      CONN_STR: redis+cluster://:gitea@gitea-redis-cluster-headless.<namespace>.svc.cluster.local:6379/0?pool_size=100&idle_timeout=180s&
```

<!-- prettier-ignore-start -->
<!-- markdownlint-disable-next-line -->
**Switch to rootless image by default**
<!-- prettier-ignore-end -->

If you are facing errors like `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` due to this automatic transition:
Have a look at [this discussion](https://gitea.com/gitea/helm-chart/issues/487#issue-220660) and either set `image.rootless: false` or manually update your `~/.ssh/known_hosts` file(s).

<!-- prettier-ignore-start -->
<!-- markdownlint-disable-next-line -->
**Transitioning from a RWO to RWX Persistent Volume**
<!-- prettier-ignore-end -->

If you want to switch to a RWX volume and go for HA, you need to

1. Backup the data stored under `/data`
2. Let the chart create a new RWX PV (or do it statically yourself)
3. Restore the backup to the same location in the new PV

<!-- prettier-ignore-start -->
<!-- markdownlint-disable-next-line -->
**Transitioning from Postgres to Postgres HA**
<!-- prettier-ignore-end -->

If you are running with a non-HA PG DB from a previous chart release, you need to set

- `postgresql-ha.enabled=false`
- `postgresql.enabled=true`

This is needed to stay with your existing single-instance DB (as the HA-variant is the new default).

<!-- prettier-ignore-start -->
<!-- markdownlint-disable-next-line -->
**Change of env-to-ini prefix**
<!-- prettier-ignore-end -->

Before this release, the env-to-ini prefix was `ENV_TO_INI__`.
This allowed a clear distinction between user-provided and chart-provided env-to-ini variables.
Due to the removal custom prefix feature in the upstream implementation of env-to-ini, the prefix has been changed to the default `GITEA__`.

If you previously had defined env vars that had the `ENV_TO_INI__` prefix, you need to change them to `GITEA__` in order for them to be picked up by the chart.

</details>

<details>

<summary>To 8.0.0</summary>

### Removal of MariaDB and MySQL DB chart dependencies <!-- omit from toc -->

In this version support for DB chart dependencies of MySQL and MariaDB have been removed to simplify the maintenance of the helm chart.
External MySQL and MariaDB databases are still supported and will be in the future.

### Postgres Update from v11 to v15 <!-- omit from toc -->

This Chart version updates the Postgres chart dependency and subsequently Postgres from v11 to v15.
Please read the [Postgres Release Notes](https://www.postgresql.org/docs/release/) for version-specific changes.
With respect to `values.yaml`, parameters `username`, `database` and `password` have been regrouped under `auth` and slightly renamed.
`persistence` has also been regrouped under the `primary` key.
Please adjust your `values.yaml` accordingly.

**Attention**: The Postgres upgrade is not automatically handled by the chart and must be done by yourself.
See [this comment](https://gitea.com/gitea/helm-chart/issues/452#issuecomment-740885) for an extensive walkthrough.
We again highly encourage users to use an external (managed) database for production instances.

</details>

<details>

<summary>To 7.0.0</summary>

### Private GPG key configuration for Gitea signing actions <!-- omit from toc -->

Having `signing.enabled=true` now requires to use either `signing.privateKey` or `signing.existingSecret` so that the Chart can automatically prepare the GPG key for Gitea internal signing actions.
See [Configure commit signing](#configure-commit-signing) for details.

</details>

<details>

<summary>To 6.0.0</summary>

### Different volume mounts for init-containers and runtime container <!-- omit from toc -->

**The `extraVolumeMounts` is deprecated** in favor of `extraInitVolumeMounts` and `extraContainerVolumeMounts`.
You can now have different mounts for the initialization phase and Gitea runtime.
The deprecated `extraVolumeMounts` will still be available for the time being and is mounted into every container.
If you want to switch to the new settings and want to mount specific volumes into all containers, you have to configure their mount points within both new settings.

**Combining values from the deprecated setting with values from the new settings is not possible.**

### New `enabled` flag for `startupProbe` <!-- omit from toc -->

Prior to this version the `startupProbe` was just a commented sample within the `values.yaml`.
With the migration to an auto-generated [Parameters](#parameters) section, a new parameter `gitea.startupProbe.enabled` has been introduced set to
`false` by default.

If you are using the `startupProbe` you need to add that new parameter and set it to `true`.
Otherwise, your defined probe won't be considered after the upgrade.

</details>

<details>

<summary>To 5.0.0</summary>

> 💥 The Helm Chart now requires Gitea versions of at least 1.11.0.

### Enable Dependencies <!-- omit from toc -->

The values to enable the dependencies, such as PostgreSQL, Memcached, MySQL and MariaDB have been moved from `gitea.database.builtIn.` to the dependency values.

You can now enable the dependencies as followed:

```yaml
memcached:
  enabled: true

postgresql:
  enabled: true

mysql:
  enabled: false

mariadb:
  enabled: false
```

### App.ini generation <!-- omit from toc -->

The app.ini generation has changed and now utilizes the environment-to-ini script provided by newer Gitea versions.
This change ensures, that the app.ini is now persistent.

### Secret Key generation <!-- omit from toc -->

Gitea secret keys (SECRET_KEY, INTERNAL_TOKEN, JWT_SECRET) are now generated automatically in certain situations:

- New install: By default the secrets are created automatically.
  If you provide secrets via `gitea.config` they will be used instead of automatic generation.
- Existing installs: The secrets won't be deployed, neither via configuration nor via auto generation.
  We explicitly prevent to set new secrets.

> 💡 It would be possible to set new secret keys manually by entering the running container and rewriting the app.ini by hand.
> However, this it is not advisable to do so for existing installations.
> Certain settings like _LDAP_ would not be readable anymore.

### Probes <!-- omit from toc -->

`gitea.customLivenessProbe`, `gitea.customReadinessProbe` and `gitea.customStartupProbe` have been removed.

They are replaced by the settings `gitea.livenessProbe`, `gitea.readinessProbe` and `gitea.startupProbe` which are now fully configurable and used _as-is_ for
a Chart deployment.
If you have customized their values instead of using the `custom` prefixed settings, please ensure that you remove the `enabled` property from each of them.

In case you want to disable one of these probes, let's say the `livenessProbe`, add the following to your values.
The `podAnnotation` is just there to have a bit more context.

```diff
gitea:
+ livenessProbe:
  podAnnotations: {}
```

### Multiple OAuth and LDAP authentication sources <!-- omit from toc -->

With `5.0.0` of this Chart it is now possible to configure Gitea with multiple OAuth and LDAP sources.
As a result, you need to update an existing OAuth/LDAP configuration in your customized `values.yaml` by replacing the object with settings to a list
of settings objects.
See [OAuth2 Settings](#oauth2-settings) and [LDAP Settings](#ldap-settings) section for details.

</details>

<details>

<summary>To 4.0.0</summary>

### Ingress changes <!-- omit from toc -->

To provide a more flexible Ingress configuration we now support not only host settings but also provide configuration for the path and pathType.
So this change changes the hosts from a simple string list, to a list containing a more complex object for more configuration.

```diff
ingress:
  enabled: false
  annotations: {}
    # kubernetes.io/ingress.class: nginx
    # kubernetes.io/tls-acme: "true"
-  hosts:
-    - git.example.com
+  hosts:
+    - host: git.example.com
+      paths:
+        - path: /
+          pathType: Prefix
  tls: []
  #  - secretName: chart-example-tls
  #    hosts:
  #      - git.example.com
```

If you want everything as it was before, you can simply add the following code to all your host entries.

```yaml
paths:
  - path: /
    pathType: Prefix
```

### Dropped kebab-case support <!-- omit from toc -->

In 3.x.x it was possible to provide an ldap configuration via kebab-case, this support has now been dropped and only camel case is supported.
See [LDAP section](#ldap-settings) for more information.

### Dependency update <!-- omit from toc -->

The chart comes with multiple databases and Memcached as dependency, the latest release updated the dependencies.

- Memcached: `4.2.20` -> `5.9.0`
- PostgreSQL: `9.7.2` -> `10.3.17`
- MariaDB: `8.0.0` -> `9.3.6`

If you're using the builtin databases you will most likely redeploy the chart in order to update the database correctly.

### Execution of initPreScript <!-- omit from toc -->

Generally spoken, this might not be a breaking change, but it is worth to be mentioned.

Prior to `4.0.0` only one init container was used to both setup directories and configure Gitea.
As of now the actual Gitea configuration is separated from the other pre-execution.
This also includes the execution of _initPreScript_.
If you have such script, please be aware of this.
Dynamically prepare the Gitea setup during execution by e.g. adding environment variables to the execution context won't work anymore.

### Gitea Version 1.14.X repository ROOT <!-- omit from toc -->

Previously the ROOT folder for the Gitea repositories was located at `/data/git/gitea-repositories`.
In version `1.14` has the path been changed to `/data/gitea-repositories`.

This chart will set the `gitea.config.repository.ROOT` value default to `/data/git/gitea-repositories`.

</details>
