"""
test_schemas.py

Unit tests for Pydantic schema definitions in `app.schemas`.
Tests ensure validation and default behavior of CustomerRequest.
"""

import pytest
from pydantic import ValidationError

from app.schemas import CustomerRequest
from app.config import DEFAULT_TOP_N


def test_customer_request_valid_data():
    """
    Test CustomerRequest with valid input data.
    """
    payload = {
        "customer_id": "CUST001",
        "top_n": 7
    }
    schema = CustomerRequest(**payload)
    assert schema.customer_id == "CUST001"
    assert schema.top_n == 7


def test_customer_request_uses_default_top_n():
    """
    Test CustomerRequest uses default value for top_n when omitted.
    """
    payload = {
        "customer_id": "CUST002"
    }
    schema = CustomerRequest(**payload)
    assert schema.customer_id == "CUST002"
    assert schema.top_n == DEFAULT_TOP_N


def test_customer_request_invalid_customer_id_type():
    """
    Test validation fails when customer_id is not a string.
    """
    payload = {
        "customer_id": 123,  # Invalid type
        "top_n": 5
    }
    with pytest.raises(ValidationError) as exc_info:
        CustomerRequest(**payload)
    assert "customer_id" in str(exc_info.value)


def test_customer_request_invalid_top_n_type():
    """
    Test validation fails when top_n is not an integer.
    """
    payload = {
        "customer_id": "CUST003",
        "top_n": "five"  # Invalid type
    }
    with pytest.raises(ValidationError) as exc_info:
        CustomerRequest(**payload)
    assert "top_n" in str(exc_info.value)
