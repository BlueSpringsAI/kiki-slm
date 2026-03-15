"""Multi-stage inference pipeline using OpenAI-compatible vLLM server.

Task 3.28: Intent -> Workflow -> Tools -> Response via LoRA adapters.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

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


class InferencePipeline:
    """Multi-stage inference pipeline via vLLM multi-LoRA server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "kiki-internal",
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.system_prompt = _SYSTEM_PROMPT

    def process_ticket(self, customer_message: str, channel: str = "email") -> dict:
        """Run full 4-stage pipeline with timing."""
        result: dict[str, Any] = {"input": customer_message, "channel": channel, "stages": {}}
        total_start = time.time()

        # Stage 1: Intent Classification
        t0 = time.time()
        intent_result = self._classify_intent(customer_message)
        result["stages"]["intent"] = {**intent_result, "latency_ms": round((time.time() - t0) * 1000)}

        intent = intent_result.get("intent", "general_inquiry")
        urgency = intent_result.get("urgency", "medium")

        # Stage 2: Workflow Planning
        t0 = time.time()
        workflow_steps = self._plan_workflow(customer_message, intent, urgency)
        result["stages"]["workflow"] = {"steps": workflow_steps, "latency_ms": round((time.time() - t0) * 1000)}

        # Stage 3: Tool Invocation
        t0 = time.time()
        tool_calls = self._invoke_tools(customer_message, workflow_steps)
        result["stages"]["tools"] = {"calls": tool_calls, "latency_ms": round((time.time() - t0) * 1000)}

        # Stage 4: Response Generation
        t0 = time.time()
        context = {"intent": intent, "urgency": urgency, "workflow_steps": workflow_steps, "tool_calls": tool_calls}
        response = self._generate_response(customer_message, context)
        result["stages"]["response"] = {"text": response, "latency_ms": round((time.time() - t0) * 1000)}

        result["total_latency_ms"] = round((time.time() - total_start) * 1000)
        result["intent"] = intent
        result["urgency"] = urgency
        result["response"] = response

        return result

    def _classify_intent(self, message: str) -> dict:
        prompt = f"Classify the intent and urgency of this customer message. Respond with JSON: {{\"intent\": \"...\", \"urgency\": \"...\"}}\n\nMessage: {message}"
        raw = self._call_adapter("intent-classifier", prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"intent": "general_inquiry", "urgency": "medium"}

    def _plan_workflow(self, message: str, intent: str, urgency: str) -> list[str]:
        prompt = f"Plan the workflow steps to resolve this {intent} ({urgency} urgency) ticket. Respond with JSON: {{\"workflow_steps\": [...]}}\n\nMessage: {message}"
        raw = self._call_adapter("workflow-reasoner", prompt)
        try:
            return json.loads(raw).get("workflow_steps", [])
        except json.JSONDecodeError:
            return []

    def _invoke_tools(self, message: str, workflow_steps: list[str]) -> list[dict]:
        prompt = f"Determine which tools to call for these workflow steps: {workflow_steps}. Respond with JSON: {{\"tool_calls\": [{{\"name\": \"...\", \"parameters\": {{...}}}}]}}\n\nMessage: {message}"
        raw = self._call_adapter("tool-caller", prompt)
        try:
            return json.loads(raw).get("tool_calls", [])
        except json.JSONDecodeError:
            return []

    def _generate_response(self, message: str, context: dict) -> str:
        prompt = f"Generate a professional, empathetic customer response.\n\nContext: {json.dumps(context)}\n\nCustomer message: {message}"
        return self._call_adapter("response-gen", prompt)

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
