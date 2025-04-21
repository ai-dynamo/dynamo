import pytest
from .deployments import get_deployment_status, get_urls

def test_get_deployment_status():
    # Test case 1: Ready condition present with message
    resource = {
        "status": {
            "conditions": [
                {
                    "type": "Ready",
                    "message": "Deployment is ready"
                }
            ]
        }
    }
    assert get_deployment_status(resource) == "Deployment is ready"

    # Test case 2: Ready condition not present
    resource = {
        "status": {
            "conditions": [
                {
                    "type": "Available",
                    "message": "Some other condition"
                }
            ]
        }
    }
    assert get_deployment_status(resource) == "unknown"

    # Test case 3: Empty conditions list
    resource = {
        "status": {
            "conditions": []
        }
    }
    assert get_deployment_status(resource) == "unknown"

    # Test case 4: No status field
    resource = {}
    assert get_deployment_status(resource) == "unknown"

    # Test case 5: No conditions field in status
    resource = {
        "status": {}
    }
    assert get_deployment_status(resource) == "unknown"

    # Test case 6: Ready condition present without message
    resource = {
        "status": {
            "conditions": [
                {
                    "type": "Ready"
                }
            ]
        }
    }
    assert get_deployment_status(resource) == "unknown"


def test_get_urls():
    resource = {
        "status": {
            "conditions": [
                {
                    "type": "IngressHostSet",
                    "message": "example.com"
                }
            ]
        }
    }
    assert get_urls(resource) == ["https://example.com"]
