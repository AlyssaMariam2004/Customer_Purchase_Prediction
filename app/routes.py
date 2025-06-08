from fastapi import APIRouter
from app.models import CustomerRequest
from app.trainer import maybe_retrain_model
from app.recommender import recommend_products

router = APIRouter()

@router.post("/user")
def recommend(req: CustomerRequest):
    maybe_retrain_model()
    return {"recommended_products": recommend_products(req.customer_id, req.top_n)}
