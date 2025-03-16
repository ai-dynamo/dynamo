from fastapi.testclient import TestClient

from ai_dynamo_store import create_app


def test_health_check():
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
