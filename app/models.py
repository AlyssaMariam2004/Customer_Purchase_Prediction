from pydantic import BaseModel

class CustomerRequest(BaseModel):
    customer_id: str
    top_n: int=5

    
