"""Package-root import and liveness/readiness without model artifacts."""

from fastapi.testclient import TestClient

from api.main import Settings, create_app


def test_missing_artifact_is_live_but_not_ready(tmp_path):
    with TestClient(create_app(Settings(str(tmp_path / "missing.pt")))) as client:
        assert client.get("/api/health/live").status_code == 200
        assert client.get("/api/health/ready").status_code == 503
        assert client.get("/api/ready").status_code == 503
        assert client.get("/api/unknown").status_code == 404
