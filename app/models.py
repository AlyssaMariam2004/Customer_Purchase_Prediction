from pydantic import BaseModel
from app.config import TOP_N_DEFAULT

class CustomerRequest(BaseModel):
    """
    Schema for customer request input data.

    Attributes:
        customer_id (str): Unique identifier for the customer.
        top_n (int): Number of top recommendations to return. Defaults to
                     the configured TOP_N_DEFAULT value.
    """
    customer_id: str
    top_n: int = TOP_N_DEFAULT

    
