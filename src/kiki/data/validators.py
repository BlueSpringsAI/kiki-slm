"""Pydantic schemas for validating data at every pipeline stage.

Task 3.8: Defines the canonical data contracts used across the entire
Kiki SLM pipeline — from raw ticket ingestion through ChatML formatting,
preference pair construction, tool calling, and final SLM output validation.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

VALID_INTENTS = (
    "order_status",
    "refund_request",
    "billing_inquiry",
    "technical_support",
    "complaint",
    "shipping_issue",
    "cancellation",
    "return_request",
    "account_management",
    "product_inquiry",
    "payment_issue",
    "fraud_report",
    "general_inquiry",
)

VALID_URGENCY = ("critical", "high", "medium", "low")

VALID_TOOLS = (
    "order_lookup_api",
    "shipment_tracking_api",
    "customer_profile_api",
    "refund_processing_api",
    "payment_gateway_api",
    "invoice_verification_api",
    "warranty_check_api",
    "ticket_update_api",
    "notification_service",
    "policy_engine",
    "vision_api",
    "document_verification",
)

VALID_CHANNELS = ("email", "chat", "phone", "social")

VALID_ROLES = ("system", "user", "assistant", "tool")


# ---------------------------------------------------------------------------
# Raw data schemas
# ---------------------------------------------------------------------------


class RawTicket(BaseModel):
    """Minimal ticket as ingested from any data source."""

    customer_message: str = Field(min_length=5)
    agent_response: str = Field(min_length=5)


class AnnotatedTicket(RawTicket):
    """Ticket after LLM or human annotation."""

    intent: str
    urgency: Literal["critical", "high", "medium", "low"]
    workflow_steps: list[str] = Field(min_length=1)
    tools_required: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("intent")
    @classmethod
    def intent_is_known(cls, v: str) -> str:
        if v not in VALID_INTENTS:
            logger.warning("Unknown intent '%s' — not in canonical list", v)
        return v


class CanonicalTicket(AnnotatedTicket):
    """Full canonical ticket with all metadata fields."""

    id: str | None = None
    category: str | None = None
    key_entities: dict[str, Any] = Field(default_factory=dict)
    resolution: str | None = None
    escalated: bool = False
    channel: str | None = None
    language: str = "en"
    source_dataset: str | None = None
    annotation_confidence: float | None = None

    @field_validator("channel")
    @classmethod
    def channel_is_valid(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_CHANNELS:
            logger.warning("Unknown channel '%s'", v)
        return v


# ---------------------------------------------------------------------------
# ChatML / training schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a ChatML conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None
    tools: list[dict[str, Any]] | None = None


class ChatMLExample(BaseModel):
    """A complete ChatML training example (≥2 messages)."""

    messages: list[ChatMessage] = Field(min_length=2)

    @model_validator(mode="before")
    @classmethod
    def coerce_raw_dicts(cls, data: Any) -> Any:
        """Accept raw list[dict] in the messages field."""
        if isinstance(data, dict) and "messages" in data:
            msgs = data["messages"]
            if msgs and isinstance(msgs[0], dict) and not isinstance(msgs[0], ChatMessage):
                data["messages"] = [ChatMessage(**m) for m in msgs]
        return data

    @field_validator("messages")
    @classmethod
    def must_have_user_and_assistant(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        roles = {m.role for m in v}
        if "user" not in roles:
            raise ValueError("ChatML example must contain at least one 'user' message")
        if "assistant" not in roles:
            raise ValueError("ChatML example must contain at least one 'assistant' message")
        return v


# ---------------------------------------------------------------------------
# Preference pair schemas (DPO / SimPO / KTO)
# ---------------------------------------------------------------------------


class PreferencePair(BaseModel):
    """A preference pair for DPO/SimPO training."""

    prompt: list[dict[str, Any]] = Field(min_length=1)
    chosen: list[dict[str, Any]] = Field(min_length=1)
    rejected: list[dict[str, Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def chosen_differs_from_rejected(self) -> PreferencePair:
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected must be different")
        return self


class KTOExample(BaseModel):
    """A KTO training example with binary label."""

    prompt: list[dict[str, Any]] = Field(min_length=1)
    completion: str = Field(min_length=1)
    label: bool


# ---------------------------------------------------------------------------
# Tool calling schemas
# ---------------------------------------------------------------------------


class ToolParameter(BaseModel):
    """A single parameter in a tool call."""

    type: str = "string"
    description: str = ""


class ToolSchema(BaseModel):
    """JSON Schema for a tool's parameters."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """A tool available to the SLM."""

    type: str = "function"
    function: dict[str, Any]

    @field_validator("function")
    @classmethod
    def function_has_name(cls, v: dict) -> dict:
        if "name" not in v:
            raise ValueError("Tool function must have a 'name' field")
        return v


class ToolCall(BaseModel):
    """A tool invocation from model output."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Tool call name must not be empty")
        return v


# ---------------------------------------------------------------------------
# SLM output schema
# ---------------------------------------------------------------------------


class SLMOutput(BaseModel):
    """Structured output from the Kiki SLM."""

    intent: str
    urgency: str
    workflow_steps: list[str]
    tools_required: list[str]
    reasoning: str
    response: str

    @field_validator("intent")
    @classmethod
    def intent_known(cls, v: str) -> str:
        if v not in VALID_INTENTS:
            logger.warning("SLM produced unknown intent '%s'", v)
        return v

    @field_validator("urgency")
    @classmethod
    def urgency_known(cls, v: str) -> str:
        if v not in VALID_URGENCY:
            logger.warning("SLM produced unknown urgency '%s'", v)
        return v


# ---------------------------------------------------------------------------
# Quality scoring schemas
# ---------------------------------------------------------------------------


class QualityScore(BaseModel):
    """Quality score produced by an LLM judge."""

    helpfulness: float = Field(ge=1.0, le=5.0)
    correctness: float = Field(ge=1.0, le=5.0)
    professionalism: float = Field(ge=1.0, le=5.0)
    empathy: float = Field(ge=1.0, le=5.0)

    @property
    def average(self) -> float:
        return (self.helpfulness + self.correctness + self.professionalism + self.empathy) / 4.0


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


class DatasetMetadata(BaseModel):
    """Metadata returned by data loaders."""

    name: str
    source: str  # "huggingface", "csv", "jsonl", "database"
    num_examples: int = Field(ge=0)
    columns: list[str] = Field(default_factory=list)
    license: str | None = None
    converter: str | None = None


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------


class ValidationReport(BaseModel):
    """Report from validate_dataset()."""

    total: int
    valid: int
    invalid: int
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def valid_ratio(self) -> float:
        return self.valid / self.total if self.total > 0 else 0.0


def validate_dataset(
    dataset,
    schema: type[BaseModel],
    *,
    sample_size: int | None = None,
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate every row in a HuggingFace Dataset against a Pydantic schema.

    Args:
        dataset: A HuggingFace ``datasets.Dataset`` or list of dicts.
        schema: The Pydantic model class to validate against.
        sample_size: If set, validate only the first *sample_size* rows.
        raise_on_error: If True, raise on the first validation error.

    Returns:
        A ``ValidationReport`` with counts and error details.
    """
    errors: list[dict[str, Any]] = []
    items = dataset if isinstance(dataset, list) else dataset

    total = 0
    valid = 0
    for idx, item in enumerate(items):
        if sample_size is not None and idx >= sample_size:
            break
        total += 1
        row = dict(item) if not isinstance(item, dict) else item
        try:
            schema.model_validate(row)
            valid += 1
        except Exception as exc:
            error_entry = {"index": idx, "error": str(exc)}
            errors.append(error_entry)
            if raise_on_error:
                raise
            if len(errors) <= 5:
                logger.warning("Row %d failed validation: %s", idx, exc)

    report = ValidationReport(total=total, valid=valid, invalid=total - valid, errors=errors)
    logger.info(
        "Validation complete: %d/%d valid (%.1f%%), %d errors",
        report.valid,
        report.total,
        report.valid_ratio * 100,
        report.invalid,
    )
    return report


def parse_slm_output(text: str) -> SLMOutput | None:
    """Attempt to parse raw model text as a structured SLMOutput.

    Returns None if parsing fails.
    """
    try:
        data = json.loads(text)
        return SLMOutput.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.debug("Failed to parse SLM output: %s", exc)
        return None
