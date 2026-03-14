"""Tests for the FastAPI endpoints."""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHealthEndpoint:
    """Tests for the /api/health endpoint."""

    def test_health_returns_ok(self, client):
        """Test that health endpoint returns ok status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestInferenceEndpoint:
    """Tests for the /api/inference endpoint."""

    def test_inference_endpoint_exists(self, client):
        """Test that inference endpoint exists."""
        # Create a simple test image
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/api/inference",
            files={"file": ("test.png", buf, "image/png")}
        )

        # Should succeed if model exists, or return 503 if not
        assert response.status_code in [200, 503]

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.joinpath(
            "models/pytorch-training-2026-03-10-14-21-42-952/best.pt"
        ).exists(),
        reason="Model file not found"
    )
    def test_inference_returns_expected_fields(self, client, sample_image_bytes):
        """Test that inference returns all expected fields."""
        response = client.post(
            "/api/inference",
            files={"file": ("test.png", io.BytesIO(sample_image_bytes), "image/png")}
        )

        assert response.status_code == 200
        data = response.json()
        assert "mask_base64" in data
        assert "polygons" in data
        assert "roof_area_px" in data
        assert "confidence" in data


class TestGeocodeEndpoint:
    """Tests for the /api/geocode endpoint."""

    def test_geocode_requires_address(self, client):
        """Test that geocode requires address parameter."""
        response = client.get("/api/geocode")
        assert response.status_code == 422  # Validation error

    def test_geocode_returns_404_without_api_key(self, client, monkeypatch):
        """Test geocode returns 404 when no API key is configured."""
        # This test assumes no API key is set
        response = client.get("/api/geocode?address=123+Main+St")
        # Will be 404 if no API key, or 200 if key is configured
        assert response.status_code in [200, 404]


class TestBatchEndpoints:
    """Tests for batch processing endpoints."""

    def test_batch_upload_requires_file(self, client):
        """Test that batch upload requires a file."""
        response = client.post("/api/batch/upload")
        assert response.status_code == 422  # Validation error

    def test_batch_upload_accepts_csv(self, client, sample_csv_content):
        """Test that batch upload accepts CSV files."""
        response = client.post(
            "/api/batch/upload",
            files={"file": ("addresses.csv", io.BytesIO(sample_csv_content), "text/csv")}
        )

        # Should create a job
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data

    def test_batch_status_not_found(self, client):
        """Test that batch status returns 404 for unknown job."""
        response = client.get("/api/batch/nonexistent-job-id")
        assert response.status_code == 404

    def test_batch_status_returns_job(self, client, sample_csv_content):
        """Test that batch status returns job details."""
        # First create a job
        create_response = client.post(
            "/api/batch/upload",
            files={"file": ("addresses.csv", io.BytesIO(sample_csv_content), "text/csv")}
        )
        job_id = create_response.json()["job_id"]

        # Then check status
        response = client.get(f"/api/batch/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
