from unittest.mock import patch
from fastapi import status
from fastapi.testclient import TestClient

from app.routes import router
from app.schemas import CustomerRequest
from fastapi import FastAPI

# Setup FastAPI app with router for testing
app = FastAPI()
app.include_router(router)

client = TestClient(app)


def test_recommend_success():
    """
    Positive Test:
    Should return a list of product recommendations when recommend_products works correctly.
    """
    mock_recommendations = ["P001", "P002", "P003"]

    with patch("app.routes.recommend_products", return_value=mock_recommendations):
        response = client.post("/user", json={"customer_id": "C123", "top_n": 3})
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == mock_recommendations


def test_recommend_failure_returns_http_exception():
    """
    Negative Test:
    Should raise HTTPException with 400 status if recommend_products returns an error dict.
    """
    mock_error = {"error": "Customer not found"}

    with patch("app.routes.recommend_products", return_value=mock_error):
        response = client.post("/user", json={"customer_id": "INVALID", "top_n": 5})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["detail"] == "Customer not found"
