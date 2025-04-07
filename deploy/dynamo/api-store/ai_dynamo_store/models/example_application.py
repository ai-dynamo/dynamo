from ai_dynamo_store.models.dynamo_deployment import (
    DynamoDeployment,
    DynamoDeploymentSpec,
    DynamoNimDeployment,
    DynamoNimDeploymentSpec,
    ObjectMeta,
    ResourceRequirements,
    Resources,
    TypeMeta,
)

# Create a new deployment
deployment = DynamoDeployment(
    type_meta=TypeMeta(api_version="nvidia.com/v1alpha1", kind="DynamoDeployment"),
    object_meta=ObjectMeta(
        name="my-deployment",
        namespace="my-namespace",
        labels={"ngc-organization": "my-org", "ngc-user": "my-user"},
    ),
    spec=DynamoDeploymentSpec(
        dynamo_nim="my-nim",
        services={
            "service1": DynamoNimDeployment(
                type_meta=TypeMeta(
                    api_version="nvidia.com/v1alpha1", kind="DynamoNimDeployment"
                ),
                object_meta=ObjectMeta(name="service1", namespace="my-namespace"),
                spec=DynamoNimDeploymentSpec(
                    dynamo_nim="my-nim",
                    dynamo_tag="my-tag",
                    resources=ResourceRequirements(
                        requests=Resources(cpu="500m", memory="1Gi")
                    ),
                ),
            )
        },
    ),
)

# The model will handle camelCase to snake_case conversion automatically
# You can access fields using snake_case in Python
print(deployment.object_meta.name)  # "my-deployment"
print(deployment.type_meta.api_version)  # "nvidia.com/v1alpha1"

# When serializing to JSON, it will use camelCase
json_data = deployment.model_dump_json(by_alias=True)
