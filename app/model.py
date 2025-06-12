from pydantic import BaseModel
from app.config import TOP_N_DEFAULT

class CustomerRequest(BaseModel):
    customer_id: str
    top_n: int = TOP_N_DEFAULT


    
