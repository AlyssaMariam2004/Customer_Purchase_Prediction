"""
routes.py

Defines API routes for the customer product recommendation service.
"""

from fastapi import APIRouter, HTTPException
from app.api.schemas import CustomerRequest
from app.services.recommender import recommend_products

# Create a router instance to register API endpoints
router = APIRouter()

@router.post("/user", summary="Get product recommendations for a customer")
def recommend(req: CustomerRequest):
    """
    Endpoint to generate product recommendations for a given customer.

    Args:
        req (CustomerRequest): Request body containing `customer_id` and optional `top_n`.

    Returns:
        list: A list of recommended Product IDs.

    Raises:
        HTTPException: If the recommendation function returns an error.
    """
    # Call the recommendation engine
    result = recommend_products(req.customer_id, req.top_n)

    # If result is a dictionary with an error message, raise HTTP error (fallback safety)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result



