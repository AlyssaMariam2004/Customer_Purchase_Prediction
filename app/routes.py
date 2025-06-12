from fastapi import APIRouter, HTTPException
from app.model import CustomerRequest
from app.recommender import recommend_products

router = APIRouter()

@router.post("/user")
def recommend(req: CustomerRequest):
    result = recommend_products(req.customer_id, req.top_n)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result



