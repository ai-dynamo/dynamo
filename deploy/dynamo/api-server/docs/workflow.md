## CAI Workflow

The CAI is in charge of handling deployments and their metadata. Currently, any API calls
regarding a Dynamo NIM are proxied to and handled by NDS v1.


### Deployments

#### 1. Creating a deployment

When creating a deployment, we first create the `deployment` entity, followed by the `deployment_revision` (which is set to active), and then lastly we create all of the specified `deployment_target`'s.
After all of these entities are created, we then send 2 requests to DMS per `deployment_target`. These create a `DynamoNimRequest` and `DynamoNimDeployment` CRDs. We store the uid of each of these
resources within the `deployment_target`.


#### 2. Updating a deployment

We update any metadata in the `deployment` entity that is specified in the request. Following this we mark any active `deployment_revision`'s (should only be 1) as inactive. For any `deployment_target`'s that
belong to the old active revisions, we delete the `DynamoNimRequest` and `DynamoNimDeployment` CRDs which cause the deployment to be terminated on K8s. Following this, we create a new active `deployment_revision`,
all `deployment_target`'s, and all required `DynamoNimRequest` and `DynamoNimDeployment` CRDs.

#### 3. Terminating a deployment

We mark any active `deployment_revision`'s (should only be 1) as inactive. For any `deployment_target`'s that
belong to the old active revisions, we delete the `DynamoNimRequest` and `DynamoNimDeployment` CRDs which cause the deployment to be terminated on K8s.


#### 4. Deleting a deployment

If there is an active `deployment_revision` this request will error. Otherwise, we will delete all data models associated with a deployment including `deployment`, `deployment_revision`'s, and `deployment_target`'s.
