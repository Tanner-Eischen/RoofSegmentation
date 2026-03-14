"""Pytest fixtures for API tests."""
import io
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image_bytes():
    """Create a sample satellite image as bytes (640x640 RGB)."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[100:200, 100:250] = [100, 100, 100]
    img[300:450, 350:550] = [120, 120, 120]

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_csv_content():
    """Create sample CSV content with addresses."""
    return b"address\n123 Main St, Springfield, IL\n456 Oak Ave, Chicago, IL\n"


@pytest.fixture
def mock_inference():
    """Mock inference functions to avoid loading the actual model."""
    mock_mask = np.zeros((640, 640), dtype=np.uint8)
    mock_mask[100:200, 100:250] = 255
    mock_polygons = [[[100, 100], [250, 100], [250, 200], [100, 200]]]
    mock_area = 37500.0
    mock_confidence = 87.5

    def mock_run_inference(image, model_path, threshold=0.5):
        return mock_mask, mock_polygons, mock_area, mock_confidence

    def mock_bytes_to_mask(image_bytes, model_path):
        return io.BytesIO(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100).getvalue(), mock_polygons, mock_area, mock_confidence

    with patch('inference.run_inference', side_effect=mock_run_inference):
        with patch('inference.image_bytes_to_mask_and_polygons', side_effect=mock_bytes_to_mask):
            with patch('inference.load_model', return_value=(MagicMock(), MagicMock(), 256, {})):
                with patch('main.load_model', return_value=(MagicMock(), MagicMock(), 256, {})):
                    yield {
                        'mask': mock_mask,
                        'polygons': mock_polygons,
                        'area': mock_area,
                        'confidence': mock_confidence
                    }


@pytest.fixture
def client(mock_inference):
    """Create a test client for the FastAPI app with mocked model."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)
