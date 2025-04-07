import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def to_snake_case(name: str) -> str:
    """Convert camelCase to snake_case"""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_python_type(property_type: str, format: Optional[str] = None) -> str:
    """Convert OpenAPI type to Python type"""
    if format == "int32" or format == "int64":
        return "int"
    elif format == "float" or format == "double":
        return "float"
    elif property_type == "boolean":
        return "bool"
    elif property_type == "string":
        return "str"
    elif property_type == "array":
        return "List"
    elif property_type == "object":
        return "Dict"
    return "Any"


def generate_pydantic_model(
    name: str, properties: Dict[str, Any], required: List[str] = None
) -> str:
    """Generate Pydantic model class from OpenAPI properties"""
    if required is None:
        required = []

    fields = []
    for prop_name, prop in properties.items():
        field_name = to_snake_case(prop_name)
        prop_type = prop.get("type", "object")
        prop_format = prop.get("format")

        # Handle nested objects
        if prop_type == "object" and "properties" in prop:
            nested_model_name = f"{name}{prop_name.title()}"
            nested_model = generate_pydantic_model(
                nested_model_name, prop["properties"], prop.get("required", [])
            )
            fields.append(nested_model)
            field_type = nested_model_name
        # Handle arrays
        elif prop_type == "array" and "items" in prop:
            items = prop["items"]
            if "properties" in items:
                nested_model_name = f"{name}{prop_name.title()}Item"
                nested_model = generate_pydantic_model(
                    nested_model_name, items["properties"], items.get("required", [])
                )
                fields.append(nested_model)
                field_type = f"List[{nested_model_name}]"
            else:
                item_type = get_python_type(
                    items.get("type", "string"), items.get("format")
                )
                field_type = f"List[{item_type}]"
        else:
            field_type = get_python_type(prop_type, prop_format)

        # Add Field with description if available
        field_def = f"{field_name}: {field_type}"
        if prop_name in required:
            field_def += " = Field(...)"
        else:
            field_def += " = Field(None)"

        if "description" in prop:
            field_def += f"  # {prop['description']}"

        fields.append(field_def)

    # Generate the model class
    model_class = f"class {name}(BaseModel):\n"
    for field in fields:
        if isinstance(field, str) and not field.startswith("class"):
            model_class += f"    {field}\n"
        else:
            model_class += f"{field}\n"

    return model_class


def main():
    # Read the CRD file
    crd_path = (
        Path(__file__).parent.parent.parent.parent
        / "operator"
        / "config"
        / "crd"
        / "bases"
        / "nvidia.com_dynamodeployments.yaml"
    )
    with open(crd_path) as f:
        crd = yaml.safe_load(f)

    # Extract the OpenAPI schema
    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]

    # Generate the models
    models = []
    models.append("from typing import List, Dict, Any, Optional")
    models.append("from pydantic import BaseModel, Field")
    models.append("\n")

    # Add TypeMeta model
    models.append(
        """class TypeMeta(BaseModel):
    api_version: str = Field(..., alias="apiVersion")
    kind: str = Field(...)
"""
    )

    # Add ObjectMeta model
    models.append(
        """class ObjectMeta(BaseModel):
    name: str = Field(...)
    namespace: str = Field(...)
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
"""
    )

    # Generate the spec model
    spec_model = generate_pydantic_model(
        "DynamoDeploymentSpec", schema["properties"]["spec"]["properties"]
    )
    models.append(spec_model)

    # Generate the main DynamoDeployment model
    models.append(
        """class DynamoDeployment(BaseModel):
    type_meta: TypeMeta = Field(..., alias="typeMeta")
    object_meta: ObjectMeta = Field(..., alias="objectMeta")
    spec: DynamoDeploymentSpec = Field(...)
"""
    )

    # Write the models to a file
    output_path = Path(__file__).parent / "dynamo_deployment.py"
    with open(output_path, "w") as f:
        f.write("\n".join(models))


if __name__ == "__main__":
    main()
