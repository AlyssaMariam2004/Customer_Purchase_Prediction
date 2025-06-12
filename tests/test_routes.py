"""
test_routes.py

Unit tests for the FastAPI routes defined in app.routes.
These tests validate the recommendation endpoint with valid and invalid input.

Fixtures are used to mock the recommender function and prevent actual model calls.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


@pytest.fixture
def valid_request():
    """Fixture returning a sample valid customer request payload."""
    return {"customer_id": "CUST123", "top_n": 5}


@pytest.fixture
def invalid_request():
    """Fixture returning a sample invalid request payload."""
    return {"customer_id": "", "top_n": 5}


@patch("app.routes.recommend_products")
def test_recommend_endpoint_success(mock_recommend, valid_request):
    """
    Test the /user endpoint returns recommendations for valid input.
    """
    mock_recommend.return_value = ["PROD1", "PROD2", "PROD3"]

    response = client.post("/user", json=valid_request)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert all(isinstance(prod, str) for prod in response.json())
    assert len(response.json()) == 3


@patch("app.routes.recommend_products")
def test_recommend_endpoint_customer_not_found(mock_recommend, valid_request):
    """
    Test the /user endpoint returns 404 when customer is not found.
    """
    from fastapi import HTTPException
    mock_recommend.side_effect = HTTPException(status_code=404, detail="Customer not found")

    response = client.post("/user", json=valid_request)

    assert response.status_code == 404
    assert "Customer not found" in response.text


@patch("app.routes.recommend_products")
def test_recommend_endpoint_internal_error(mock_recommend, valid_request):
    """
    Test the /user endpoint returns 500 for unexpected internal errors.
    """
    from fastapi import HTTPException
    mock_recommend.side_effect = HTTPException(status_code=500, detail="Internal server error")

    response = client.post("/user", json=valid_request)

    assert response.status_code == 500
    assert "Internal server error" in response.text


@patch("app.routes.recommend_products")
def test_recommend_endpoint_fallback_error_dict(mock_recommend, valid_request):
    """
    Test if returned dictionary with 'error' key raises HTTP 400.
    """
    mock_recommend.return_value = {"error": "Malformed request"}

    response = client.post("/user", json=valid_request)

    assert response.status_code == 400
    assert "Malformed request" in response.text
