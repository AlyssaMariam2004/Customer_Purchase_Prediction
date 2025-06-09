from fastapi import APIRouter
from app.models import CustomerRequest
from app.trainer import maybe_retrain_model
from app.recommender import recommend_products

router = APIRouter()

@router.post("/user")
def recommend(req: CustomerRequest):
    """
    API endpoint to get product recommendations for a given customer.

    This endpoint first checks if the recommendation model needs retraining,
    then generates product recommendations based on the customer ID and
    requested number of top recommendations.

    Args:
        req (CustomerRequest): Pydantic model containing:
            - customer_id (str): The ID of the customer.
            - top_n (int): Number of top recommendations requested.

    Returns:
        dict: A dictionary with key 'recommended_products' mapping to a list
              of product IDs recommended for the customer.
    """
    # Check and perform model retraining if needed before making recommendations
    maybe_retrain_model()
    
    # Generate and return recommendations for the specified customer
    return {"recommended_products": recommend_products(req.customer_id, req.top_n)}
