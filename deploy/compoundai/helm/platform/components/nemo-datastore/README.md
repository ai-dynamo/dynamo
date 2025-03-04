# Deploying with Helm

This guide explains steps to prepare your environment, create custom values for your deployment and deploy NeMo Data Store. 
NeMo Data Store uses storage components which must be provisioned for a robust datastore deployment. 

## Deploy datastore microservice
You can install NeMo Data Store in the preferred namespace using command below. In following sections, we will prepare a customized values.yaml based on your requirements.
You can proceed with`Basic Setup` with database and object store provisioned in the same cluster to get started quickly. For a more robust production set up we recommend following the `Production Setup` section.

```bash
helm --namespace datastore-ms install datastore <datastore_helmchart> -f custom-values.yaml
```

## Basic Setup
You cnan use following `values.yaml` to get started quickly. Please replace the `storageClass` with appropriate storage class available for your k8s cluster.

`values.yaml`
```yaml
image:
  repository: nvcr.io/nvidian/nemo-llm/nemo-datastore-service
  tag: 30b49b59071eff4e89b269ef2a268f84c1e46521

minio:
  enabled: true
  persistence:
    enabled: true
    size: 5Gi
    storageClass: local-path

postgresqlapi:
  enabled: true
  primary:
    persistence:
      storageClass: local-path
      accessModes:
        - ReadWriteOnce

postgresqlgitea:
  enabled: true
  primary:
    persistence:
      storageClass: local-path
      accessModes:
        - ReadWriteOnce

createInClusterSecrets: true
imagePullSecrets:
- name: nvcrimagepullsecret
ingress:
  enabled: false

gitea:
  image:
    repository: nvcr.io/nvidian/nemo-llm/nemo-datastore-gitea
    tag: v1.21.3-30b49b59071eff4e89b269ef2a268f84c1e46521
    rootless: true
  gitea:
    config:
      server:
        LFS_JWT_SECRET: "3iCw8yOpo0Ci7suzISAewlI9v0srCdIq9XkHSz73Z3s"
  persistence:
    enabled: true
    storageClass: local-path
    claimName: gitea-shared-storage
    size: 2Gi
    accessModes:
      - ReadWriteOnce
  serviceAccount:
    create: true
    imagePullSecrets:
    - name: nvcrimagepullsecret
    name: gitea
```


## Production Setup
We recommend using storage components (database, object store) hosted by cloud service providers for a robust production deployment. 

### Provision S3/Minio compatible cloud object storage:
Nemo Data Store Service requires s3/minio compatible object store to store models, datasets and evaluation results. 

### Provision PostgreSQL Database
NeMo Data Store requires two PostgreSQL databases to store data entities and versions respectively.
- PostgreSQL `gitea-database` will be used by gitea
- PostgreSQL `api-database` will be used by datastore api service
 
### Create Secrets
NeMo Data Store requires you to create the following secrets in the namespace prior to deployment.

#### Gitea admin secret:
This secret name should be supplied to `gitea.gitea.admin.existingSecret` field in values.yaml as explained in section-2 below.

```
data:
    username: "nvidia"
    password: ""
```
Note: username should be "nvidia"

#### Gitea storage secret:
The secret should be created in the following format and it's name should be supplied to `gitea.gitea.additionalConfigSources.secret.secretName` field in *values.yaml* as explained in the (Api storage secret)[Api storage secret] section.

```
data:
    database: |
      HOST="gitea-database's HOST:PORT"
      PASSWD="gitea-database's password"
    server: LFS_JWT_SECRET=""
    storage.minio: |
      MINIO_ACCESS_KEY_ID="object store access key id"
      MINIO_SECRET_ACCESS_KEY="object store secret access key"
```

Note:
1. database.HOST field should have host address and port: `HOST=<DB_1_HOST:DB_1_PORT>`
If port field is omitted (`HOST=<DB_1HOST>`), default postgresql port (5432) will be used. 

2. `LFS_JWT_SECRET` value be base64 encoded 32 bytes.
Sample command to generate a secret is mentioned below:
```
dd if=/dev/urandom bs=1 count=32 status=none | base64 | tr '/+' '_-' | tr -d '='
```

#### Api storage secret
This should be created in following format.
This secret name corresponds to `external-postgresqlgitea.existingSecret` field in values.yaml

```
data:
    DB_USER: database username
    DB_PASSWORD: database password
    DB_HOST: database host
    DB_NAME: database name
    DB_PORT: database port
```

### Set up your helm values file
Based on the set up above we can prepare your values.yaml for deployment.

`custom-values.yaml`
```yaml
externalPostgresqlApi:
  auth:
    username: <detail from api-database>
    database: <detail from api-database>
    existingSecret: <Name of Api storage secret>
  host: <detail from api-database>
  port: <detail from api-database>
gitea:
  gitea:
    admin:
      existingSecret: <Name of Gitea admin secret>
    additionalConfigSources:
      - secret:
          secretName: <Name of Gitea storage secret>
    config:
      database:
        NAME: <gitea-database name>
        USER: <gitea-database user>
        SSL_MODE: require
      storage.minio:
        MINIO_USE_SSL: true
        MINIO_BUCKET: <object store bucket name>
        MINIO_LOCATION: <object store region>
        MINIO_ENDPOINT: <object store end-point>
```
