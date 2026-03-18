"""Multi-stage inference pipeline using OpenAI-compatible vLLM server.

Task 3.28: Intent -> Workflow -> Tools -> Response via LoRA adapters.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from kiki.utils.logging import log_with_data, set_correlation_id

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are Kiki, an AI customer service agent. When given a customer message, analyze it and respond with:
1. Your classification (intent, urgency)
2. The workflow steps needed to resolve this
3. Which tools to invoke with what parameters
4. A professional, empathetic response to the customer

Always respond in valid JSON with these fields:
- intent: string
- urgency: string (critical/high/medium/low)
- workflow_steps: list of strings
- tools_required: list of strings with parameters
- reasoning: brief explanation of your analysis
- response: the customer-facing reply"""


_DEFAULT_ADAPTERS = {
    "intent": "intent-classifier",
    "workflow": "workflow-reasoner",
    "tools": "tool-caller",
    "response": "response-gen",
}


class InferencePipeline:
    """Multi-stage inference pipeline via vLLM multi-LoRA server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "kiki-internal",
        adapter_names: dict[str, str] | None = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.system_prompt = _SYSTEM_PROMPT
        self.adapters = adapter_names or dict(_DEFAULT_ADAPTERS)

    def process_ticket(self, customer_message: str, channel: str = "email") -> dict:
        """Run full 4-stage pipeline with timing and correlation ID."""
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)

        result: dict[str, Any] = {"input": customer_message, "channel": channel, "correlation_id": cid, "stages": {}}
        total_start = time.time()

        # Stage 1: Intent Classification
        t0 = time.time()
        intent_result = self._classify_intent(customer_message)
        latency_ms = round((time.time() - t0) * 1000)
        degraded = intent_result.get("_degraded", False)
        result["stages"]["intent"] = {**intent_result, "latency_ms": latency_ms}
        log_with_data(logger, logging.INFO, "Stage intent complete", {"latency_ms": latency_ms, "degraded": degraded})

        intent = intent_result.get("intent", "general_inquiry")
        urgency = intent_result.get("urgency", "medium")

        # Stage 2: Workflow Planning
        t0 = time.time()
        workflow_steps = self._plan_workflow(customer_message, intent, urgency)
        latency_ms = round((time.time() - t0) * 1000)
        result["stages"]["workflow"] = {"steps": workflow_steps, "latency_ms": latency_ms}
        log_with_data(logger, logging.INFO, "Stage workflow complete", {"latency_ms": latency_ms, "steps": len(workflow_steps)})

        # Stage 3: Tool Invocation
        t0 = time.time()
        tool_calls = self._invoke_tools(customer_message, workflow_steps)
        latency_ms = round((time.time() - t0) * 1000)
        result["stages"]["tools"] = {"calls": tool_calls, "latency_ms": latency_ms}
        log_with_data(logger, logging.INFO, "Stage tools complete", {"latency_ms": latency_ms, "calls": len(tool_calls)})

        # Stage 4: Response Generation
        t0 = time.time()
        context = {"intent": intent, "urgency": urgency, "workflow_steps": workflow_steps, "tool_calls": tool_calls}
        response = self._generate_response(customer_message, context)
        latency_ms = round((time.time() - t0) * 1000)
        result["stages"]["response"] = {"text": response, "latency_ms": latency_ms}
        log_with_data(logger, logging.INFO, "Stage response complete", {"latency_ms": latency_ms})

        result["total_latency_ms"] = round((time.time() - total_start) * 1000)
        result["intent"] = intent
        result["urgency"] = urgency
        result["response"] = response

        log_with_data(logger, logging.INFO, "Pipeline complete", {"total_latency_ms": result["total_latency_ms"], "intent": intent, "urgency": urgency})
        return result

    def _classify_intent(self, message: str) -> dict:
        prompt = f"Classify the intent and urgency of this customer message. Respond with JSON: {{\"intent\": \"...\", \"urgency\": \"...\"}}\n\nMessage: {message}"
        raw = self._call_adapter(self.adapters["intent"], prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Intent classification returned invalid JSON, using defaults")
            return {"intent": "general_inquiry", "urgency": "medium", "_degraded": True}

    def _plan_workflow(self, message: str, intent: str, urgency: str) -> list[str]:
        prompt = f"Plan the workflow steps to resolve this {intent} ({urgency} urgency) ticket. Respond with JSON: {{\"workflow_steps\": [...]}}\n\nMessage: {message}"
        raw = self._call_adapter(self.adapters["workflow"], prompt)
        try:
            return json.loads(raw).get("workflow_steps", [])
        except json.JSONDecodeError:
            logger.warning("Workflow planning returned invalid JSON")
            return []

    def _invoke_tools(self, message: str, workflow_steps: list[str]) -> list[dict]:
        prompt = f"Determine which tools to call for these workflow steps: {workflow_steps}. Respond with JSON: {{\"tool_calls\": [{{\"name\": \"...\", \"parameters\": {{...}}}}]}}\n\nMessage: {message}"
        raw = self._call_adapter(self.adapters["tools"], prompt)
        try:
            return json.loads(raw).get("tool_calls", [])
        except json.JSONDecodeError:
            logger.warning("Tool invocation returned invalid JSON")
            return []

    def _generate_response(self, message: str, context: dict) -> str:
        prompt = f"Generate a professional, empathetic customer response.\n\nContext: {json.dumps(context)}\n\nCustomer message: {message}"
        return self._call_adapter(self.adapters["response"], prompt)

    def process_single_stage(self, message: str, adapter: str = "response-gen") -> str:
        """Single-stage inference for evaluation/testing."""
        return self._call_adapter(adapter, message)

    def _call_adapter(self, adapter: str, user_message: str) -> str:
        """Make a single call to a LoRA adapter via vLLM."""
        try:
            resp = self.client.chat.completions.create(
                model=adapter,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Adapter '%s' call failed: %s", adapter, exc)
            return ""
