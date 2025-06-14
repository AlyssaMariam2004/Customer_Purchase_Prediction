# tests/test_schemas.py

import pytest
from unittest.mock import patch
from pydantic import ValidationError

from app.api.schemas import CustomerRequest
from app.core.config import DEFAULT_TOP_N


def test_customer_request_valid_data():
    """
    Positive Test:
    Ensure CustomerRequest accepts valid input with both fields provided.
    """
    req = CustomerRequest(customer_id="C001", top_n=5)
    assert req.customer_id == "C001"
    assert req.top_n == 5


def test_customer_request_uses_default_top_n():
    """
    Positive Test:
    Ensure top_n uses default when not provided.
    """
    req = CustomerRequest(customer_id="C002")
    assert req.customer_id == "C002"
    assert req.top_n == DEFAULT_TOP_N


def test_customer_request_invalid_missing_customer_id():
    """
    Negative Test:
    Should raise ValidationError when customer_id is missing.
    """
    with pytest.raises(ValidationError) as exc_info:
        CustomerRequest(top_n=5)
    assert "customer_id" in str(exc_info.value)


@patch("app.api.schemas.DEFAULT_TOP_N", 3)
def test_customer_request_invalid_type_for_top_n():
    """
    Negative Test:
    Should raise ValidationError when top_n is not an integer.
    """
    with pytest.raises(ValidationError) as exc_info:
        CustomerRequest(customer_id="C003", top_n="five")
    assert "Input should be a valid integer" in str(exc_info.value)



