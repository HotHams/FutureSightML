"""Tests for the FastAPI server endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked app state."""
    # Mock the app state initialization to avoid needing real model files
    with patch("showdown.api.server.app_state") as mock_state:
        mock_state.initialize = AsyncMock()
        mock_state.shutdown = AsyncMock()
        mock_state.formats = {
            "gen9ou": {
                "id": "gen9ou",
                "name": "Gen 9 OU",
                "has_model": False,
                "meta_teams": 0,
                "pool_size": 0,
            }
        }
        mock_state.get_format_list = MagicMock(return_value=[
            {"id": "gen9ou", "name": "Gen 9 OU", "has_model": False,
             "meta_teams": 0, "pool_size": 0}
        ])
        mock_state.pokemon_data = None
        mock_state.evaluator = None

        from showdown.api.server import app
        with TestClient(app) as c:
            yield c


class TestFormatsEndpoint:
    def test_get_formats_returns_list(self, client):
        response = client.get("/api/formats")
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert isinstance(data["formats"], list)


class TestHealthCheck:
    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
