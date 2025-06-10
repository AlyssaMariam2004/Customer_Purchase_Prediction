from fastapi import APIRouter
from app.models import CustomerRequest
from app.recommender import recommend_products

router = APIRouter()

@router.post("/user")
def recommend(req: CustomerRequest):
    return {"recommended_products": recommend_products(req.customer_id, req.top_n)}

