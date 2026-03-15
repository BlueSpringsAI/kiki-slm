"""Tool call parsing, validation, and execution.

Task 3.30: Parse tool calls from model output, validate against
schemas, execute against mock or real APIs.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from kiki.data.validators import VALID_TOOLS, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "order_lookup_api": {
        "description": "Look up order details by order ID",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    "shipment_tracking_api": {
        "description": "Track shipment status",
        "parameters": {"type": "object", "properties": {"tracking_id": {"type": "string"}, "order_id": {"type": "string"}}, "required": ["tracking_id"]},
    },
    "customer_profile_api": {
        "description": "Get customer account info",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}, "email": {"type": "string"}}, "required": ["customer_id"]},
    },
    "refund_processing_api": {
        "description": "Process monetary refunds",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}, "reason": {"type": "string"}}, "required": ["order_id", "amount"]},
    },
    "payment_gateway_api": {
        "description": "Handle payment operations",
        "parameters": {"type": "object", "properties": {"payment_id": {"type": "string"}, "action": {"type": "string"}}, "required": ["payment_id"]},
    },
    "invoice_verification_api": {
        "description": "Validate invoice data",
        "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}}, "required": ["invoice_id"]},
    },
    "warranty_check_api": {
        "description": "Check warranty status",
        "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}, "serial_number": {"type": "string"}}, "required": ["product_id"]},
    },
    "ticket_update_api": {
        "description": "Update ticket state",
        "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}, "status": {"type": "string"}, "notes": {"type": "string"}}, "required": ["ticket_id", "status"]},
    },
    "notification_service": {
        "description": "Send customer notifications",
        "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}, "channel": {"type": "string"}}, "required": ["recipient", "message"]},
    },
    "policy_engine": {
        "description": "Check business rules and policies",
        "parameters": {"type": "object", "properties": {"policy_type": {"type": "string"}, "context": {"type": "object"}}, "required": ["policy_type", "context"]},
    },
    "vision_api": {
        "description": "Analyze product images for damage",
        "parameters": {"type": "object", "properties": {"image_url": {"type": "string"}, "analysis_type": {"type": "string"}}, "required": ["image_url"]},
    },
    "document_verification": {
        "description": "OCR and verify uploaded documents",
        "parameters": {"type": "object", "properties": {"document_url": {"type": "string"}, "document_type": {"type": "string"}}, "required": ["document_url"]},
    },
}


class ToolExecutor:
    """Parse and execute tool calls from model output."""

    def __init__(self, mock: bool = True) -> None:
        self.mock = mock
        self._mock_responses: dict[str, list[dict]] = {}

    def parse_tool_calls(self, model_output: str) -> list[ToolCall]:
        """Parse tool calls from model output."""
        calls = []
        try:
            parsed = json.loads(model_output)
            if isinstance(parsed, dict):
                # Check tools_required
                tools = parsed.get("tools_required", [])
                for tool in tools:
                    if isinstance(tool, str):
                        calls.append(ToolCall(name=tool, parameters={}))
                    elif isinstance(tool, dict):
                        calls.append(ToolCall(
                            name=tool.get("name", ""),
                            parameters=tool.get("parameters", tool.get("arguments", {})),
                        ))
                # Check tool_calls (OpenAI format)
                for tc in parsed.get("tool_calls", []):
                    func = tc.get("function", tc)
                    args = func.get("arguments", func.get("parameters", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append(ToolCall(name=func.get("name", ""), parameters=args))
        except json.JSONDecodeError:
            logger.debug("Could not parse tool calls from output")
        return calls

    def validate_call(self, tool_call: ToolCall) -> tuple[bool, str]:
        """Validate a tool call against its schema."""
        if tool_call.name not in VALID_TOOLS:
            return False, f"Unknown tool: {tool_call.name}"

        schema = TOOL_SCHEMAS.get(tool_call.name)
        if not schema:
            return True, ""  # Valid tool, no schema to check

        required = schema["parameters"].get("required", [])
        missing = [p for p in required if p not in tool_call.parameters]
        if missing:
            return False, f"Missing required parameters: {missing}"

        return True, ""

    def execute(self, tool_call: ToolCall) -> dict:
        """Execute a tool call. Returns mock response if mock=True."""
        valid, error = self.validate_call(tool_call)
        if not valid:
            return {"error": error, "success": False}

        if self.mock:
            return self._get_mock_response(tool_call)

        # Real execution would go here
        logger.warning("Real tool execution not implemented for '%s'", tool_call.name)
        return {"error": "Real execution not implemented", "success": False}

    def execute_batch(self, tool_calls: list[ToolCall]) -> list[dict]:
        """Execute multiple tool calls."""
        return [self.execute(tc) for tc in tool_calls]

    def register_mock_response(self, tool_name: str, params_match: dict, response: dict) -> None:
        """Register a mock response for testing."""
        if tool_name not in self._mock_responses:
            self._mock_responses[tool_name] = []
        self._mock_responses[tool_name].append({"match": params_match, "response": response})

    def _get_mock_response(self, tool_call: ToolCall) -> dict:
        """Generate a realistic mock response."""
        # Check registered mocks first
        for mock in self._mock_responses.get(tool_call.name, []):
            if all(tool_call.parameters.get(k) == v for k, v in mock["match"].items()):
                return mock["response"]

        # Default mock responses
        return _DEFAULT_MOCKS.get(tool_call.name, _default_mock)(tool_call)


# ---------------------------------------------------------------------------
# Default mock response generators
# ---------------------------------------------------------------------------

def _default_mock(tool_call: ToolCall) -> dict:
    return {"success": True, "tool": tool_call.name, "data": {"message": "Mock response"}}


def _mock_order_lookup(tc: ToolCall) -> dict:
    return {
        "success": True,
        "order_id": tc.parameters.get("order_id", "ORD-00000"),
        "status": "in_transit",
        "items": [{"name": "Widget Pro", "qty": 1, "price": 49.99}],
        "eta": "2026-03-20",
    }


def _mock_shipment_tracking(tc: ToolCall) -> dict:
    return {
        "success": True,
        "tracking_id": tc.parameters.get("tracking_id", "TRK-00000"),
        "carrier": "FedEx",
        "status": "in_transit",
        "last_location": "Distribution Center, Memphis TN",
        "eta": "2026-03-19",
    }


def _mock_refund(tc: ToolCall) -> dict:
    return {
        "success": True,
        "refund_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
        "order_id": tc.parameters.get("order_id", ""),
        "amount": tc.parameters.get("amount", 0),
        "status": "processed",
        "estimated_days": 5,
    }


def _mock_customer_profile(tc: ToolCall) -> dict:
    return {
        "success": True,
        "customer_id": tc.parameters.get("customer_id", "CUST-00000"),
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "tier": "gold",
        "account_since": "2023-06-15",
    }


_DEFAULT_MOCKS = {
    "order_lookup_api": _mock_order_lookup,
    "shipment_tracking_api": _mock_shipment_tracking,
    "refund_processing_api": _mock_refund,
    "customer_profile_api": _mock_customer_profile,
}


class MockToolExecutor(ToolExecutor):
    """Pre-configured mock executor for testing."""

    def __init__(self) -> None:
        super().__init__(mock=True)
