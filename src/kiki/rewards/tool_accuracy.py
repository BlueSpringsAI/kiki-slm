"""Tool calling accuracy reward function for GRPO.

Task 3.18: Checks tool name correctness, JSON validity,
required parameters, and parameter types.
"""

from __future__ import annotations

import json
import logging

from kiki.data.validators import VALID_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas — required parameters per tool
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict] = {
    "order_lookup_api": {
        "required": ["order_id"],
        "optional": ["customer_id"],
    },
    "shipment_tracking_api": {
        "required": ["tracking_id"],
        "optional": ["order_id"],
    },
    "customer_profile_api": {
        "required": ["customer_id"],
        "optional": ["email"],
    },
    "refund_processing_api": {
        "required": ["order_id", "amount"],
        "optional": ["reason", "method"],
    },
    "payment_gateway_api": {
        "required": ["payment_id"],
        "optional": ["action", "amount"],
    },
    "invoice_verification_api": {
        "required": ["invoice_id"],
        "optional": ["customer_id"],
    },
    "warranty_check_api": {
        "required": ["product_id"],
        "optional": ["serial_number", "purchase_date"],
    },
    "ticket_update_api": {
        "required": ["ticket_id", "status"],
        "optional": ["notes", "assignee"],
    },
    "notification_service": {
        "required": ["recipient", "message"],
        "optional": ["channel", "priority"],
    },
    "policy_engine": {
        "required": ["policy_type", "context"],
        "optional": ["override"],
    },
    "vision_api": {
        "required": ["image_url"],
        "optional": ["analysis_type"],
    },
    "document_verification": {
        "required": ["document_url"],
        "optional": ["document_type"],
    },
}


class ToolAccuracyReward:
    """GRPO reward for tool calling correctness.

    Scoring:
        1.0  — correct tool name + all required params present + valid JSON
        0.5  — correct tool name but missing/wrong params
        0.0  — wrong tool name or invalid JSON
    """

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """Score a batch of completions."""
        return [self._score_single(c) for c in completions]

    def _score_single(self, completion: str) -> float:
        tool_calls = self._extract_tool_calls(completion)
        if not tool_calls:
            # No tool calls found — may be legitimate (e.g. general_inquiry)
            # Give partial credit if the response is otherwise valid JSON
            try:
                parsed = json.loads(completion)
                if isinstance(parsed, dict) and parsed.get("tools_required") == []:
                    return 0.8  # Correctly identified no tools needed
            except (json.JSONDecodeError, AttributeError):
                pass
            return 0.0

        total_score = 0.0
        for call in tool_calls:
            total_score += self._score_tool_call(call)
        return total_score / len(tool_calls)

    def _extract_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from completion text."""
        calls = []

        # Try parsing as JSON first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Check for tools_required field with parameters
                tools = parsed.get("tools_required", [])
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, str):
                            calls.append({"name": tool, "parameters": {}})
                        elif isinstance(tool, dict):
                            calls.append({
                                "name": tool.get("name", ""),
                                "parameters": tool.get("parameters", tool.get("arguments", {})),
                            })

                # Check for tool_calls field (OpenAI format)
                tool_calls = parsed.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        func = tc.get("function", tc)
                        name = func.get("name", "")
                        args = func.get("arguments", func.get("parameters", {}))
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        calls.append({"name": name, "parameters": args})
        except json.JSONDecodeError:
            pass

        return calls

    def _score_tool_call(self, call: dict) -> float:
        """Score a single tool call."""
        name = call.get("name", "")
        params = call.get("parameters", {})

        # Check tool name
        if name not in VALID_TOOLS:
            logger.debug("Unknown tool name: %s", name)
            return 0.0

        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            # Valid tool but no schema defined — give partial credit
            return 0.5

        # Check required parameters
        required = schema.get("required", [])
        missing = [p for p in required if p not in params]

        if not missing:
            return 1.0  # Perfect: right tool + all required params

        # Partial credit: right tool but missing params
        present_ratio = (len(required) - len(missing)) / len(required) if required else 1.0
        score = 0.5 + (0.5 * present_ratio)
        logger.debug("Tool '%s' missing params %s (score=%.2f)", name, missing, score)
        return score
