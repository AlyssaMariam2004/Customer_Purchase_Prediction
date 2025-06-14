"""
schemas.py

This module defines request schemas for the FastAPI application using Pydantic.
"""

from pydantic import BaseModel
from app.core.config import DEFAULT_TOP_N

class CustomerRequest(BaseModel):
    """
    Schema for incoming customer recommendation request.

    Attributes:
        customer_id (str): Unique identifier for the customer.
        top_n (int): Number of top product recommendations to return.
    """
    customer_id: str
    top_n: int = DEFAULT_TOP_N  # Default value imported from configuration


    
