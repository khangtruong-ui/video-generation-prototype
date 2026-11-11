"""Integration tests for FastAPI REST API endpoints."""

import time
from fastapi.testclient import TestClient
from video_gen.api.app import app

client = TestClient(app)


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "endpoints" in data
    assert data["version"] == "0.1.0"


def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data
    assert "default_config" in data


def test_generate_sync_mock():
    payload = {
        "prompt": "Gentle waves lapping on a peaceful shore at sunset",
        "duration_seconds": 2.0,
        "fps": 5,
        "pipeline_type": "mock",
        "async_mode": False,
    }
    res = client.post("/api/v1/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["video_id"] is not None
    assert data["download_url"] is not None
    assert data["duration_seconds"] > 0

    # Test downloading the video
    dl_res = client.get(data["download_url"])
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"] == "video/mp4"
    assert len(dl_res.content) > 0


def test_generate_sync_with_interpolation():
    payload = {
        "prompt": "Sunlight dancing across glistening waves",
        "duration_seconds": 2.0,
        "fps": 5,
        "target_fps": 15,
        "interpolate_method": "blend",
        "pipeline_type": "mock",
        "async_mode": False,
    }
    res = client.post("/api/v1/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["fps"] == 15
    assert data["metadata"]["interpolated"] is True


def test_generate_async_mock():
    payload = {
        "prompt": "Fast drone flight through futuristic skyscraper canyon",
        "duration_seconds": 2.0,
        "fps": 5,
        "pipeline_type": "mock",
        "async_mode": True,
    }
    res = client.post("/api/v1/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "processing"
    job_id = data["video_id"]

    # Poll status for completion (mock generates in < 1 second)
    for _ in range(10):
        time.sleep(0.3)
        poll_res = client.get(f"/api/v1/jobs/{job_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()
        if poll_data["status"] == "completed":
            break

    assert poll_data["status"] == "completed"
    assert poll_data["progress_percent"] == 100.0


def test_generate_upload_mock():
    files = {
        "prompt_file": ("script.txt", b"A breathtaking valley in autumn mist.", "text/plain"),
    }
    data = {
        "duration_seconds": "2.0",
        "fps": "5",
        "pipeline_type": "mock",
    }
    res = client.post("/api/v1/generate/upload", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "completed"
