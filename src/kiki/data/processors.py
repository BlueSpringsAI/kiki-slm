"""ChatML converters for the Kiki SLM pipeline.

Task 3.2: Convert each dataset format to TRL's messages format (ChatML).
Each converter is a static method returning {"messages": [...]}.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import datasets

from kiki.data.validators import VALID_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

KIKI_SYSTEM_PROMPT_TEMPLATE = """You are Kiki, an AI customer service agent for {company_name}. You help customers by:
1. Understanding their request and classifying intent
2. Planning the resolution workflow
3. Invoking the correct enterprise tools
4. Generating professional, empathetic responses

Available tools: {tool_list}

Always respond in structured JSON with: intent, urgency, workflow_steps, tools_required, reasoning, response.
When you need to call a tool, use the function calling format.
If required data is missing, ask the customer for it.
If confidence is low (<0.7), recommend escalation to a human agent."""

DEFAULT_TOOL_LIST = ", ".join(VALID_TOOLS)
DEFAULT_SYSTEM_PROMPT = KIKI_SYSTEM_PROMPT_TEMPLATE.format(
    company_name="our company",
    tool_list=DEFAULT_TOOL_LIST,
)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# ChatMLConverter
# ---------------------------------------------------------------------------


class ChatMLConverter:
    """Static methods to convert various dataset formats to ChatML messages."""

    @staticmethod
    def from_bitext(example: dict) -> dict:
        """Bitext customer-support datasets: instruction + response, with intent/category."""
        user_text = example.get("instruction", example.get("customer_message", ""))
        assistant_text = example.get("response", example.get("agent_response", ""))
        intent = example.get("intent", example.get("category", ""))

        system = DEFAULT_SYSTEM_PROMPT
        if intent:
            system += f"\n\nThis ticket has been classified as: {intent}"

        return {"messages": [_msg("system", system), _msg("user", user_text), _msg("assistant", assistant_text)]}

    @staticmethod
    def from_ticket(example: dict) -> dict:
        """Customer-support-tickets dataset with department/priority metadata."""
        user_text = example.get("customer_message", example.get("body", ""))
        assistant_text = example.get("agent_response", example.get("response", example.get("answer", "")))

        system = DEFAULT_SYSTEM_PROMPT
        meta_parts = []
        if example.get("department"):
            meta_parts.append(f"Department: {example['department']}")
        if example.get("priority"):
            meta_parts.append(f"Priority: {example['priority']}")
        if example.get("ticket_type"):
            meta_parts.append(f"Type: {example['ticket_type']}")
        if meta_parts:
            system += "\n\nTicket metadata:\n" + "\n".join(meta_parts)

        return {"messages": [_msg("system", system), _msg("user", user_text), _msg("assistant", assistant_text)]}

    @staticmethod
    def from_glaive_function_calling(example: dict) -> dict:
        """Parse SYSTEM/USER/ASSISTANT/FUNCTION_CALL/FUNCTION_RESPONSE format."""
        raw = example.get("text", example.get("conversations", ""))
        if isinstance(raw, list):
            # Already structured
            return {"messages": [_msg(m.get("role", "user"), m.get("content", "")) for m in raw]}

        messages: list[dict[str, str]] = []
        # Split on role markers
        parts = re.split(r"(SYSTEM:|USER:|ASSISTANT:|FUNCTION_CALL:|FUNCTION_RESPONSE:)", str(raw))

        current_role = None
        for part in parts:
            part = part.strip()
            if not part:
                continue
            role_map = {
                "SYSTEM:": "system",
                "USER:": "user",
                "ASSISTANT:": "assistant",
                "FUNCTION_CALL:": "assistant",
                "FUNCTION_RESPONSE:": "tool",
            }
            if part in role_map:
                current_role = role_map[part]
            elif current_role:
                messages.append(_msg(current_role, part))

        if len(messages) < 2:
            messages = [_msg("system", DEFAULT_SYSTEM_PROMPT), _msg("user", str(raw)), _msg("assistant", "")]

        return {"messages": messages}

    @staticmethod
    def from_xlam(example: dict) -> dict:
        """Parse xLAM's query/tools/answers format."""
        query = example.get("query", "")
        tools_raw = example.get("tools", "[]")
        answers_raw = example.get("answers", "[]")

        if isinstance(tools_raw, str):
            try:
                tools = json.loads(tools_raw)
            except json.JSONDecodeError:
                tools = []
        else:
            tools = tools_raw

        if isinstance(answers_raw, str):
            try:
                answers = json.loads(answers_raw)
            except json.JSONDecodeError:
                answers = []
        else:
            answers = answers_raw

        system = DEFAULT_SYSTEM_PROMPT
        if tools:
            system += "\n\nAvailable function schemas:\n" + json.dumps(tools, indent=2)

        assistant_content = json.dumps(answers, indent=2) if answers else "No tool call needed."

        return {"messages": [_msg("system", system), _msg("user", query), _msg("assistant", assistant_content)]}

    @staticmethod
    def from_hermes(example: dict) -> dict:
        """Parse NousResearch Hermes XML-style tool calling."""
        conversations = example.get("conversations", [])
        if not conversations:
            text = example.get("text", "")
            return {"messages": [_msg("system", DEFAULT_SYSTEM_PROMPT), _msg("user", text), _msg("assistant", "")]}

        messages = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))
            # Normalize role names
            role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
            role = role_map.get(role, role)
            messages.append(_msg(role, content))

        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, _msg("system", DEFAULT_SYSTEM_PROMPT))

        return {"messages": messages}

    @staticmethod
    def from_toolbench(example: dict) -> dict:
        """Parse Thought→Action→Observation chains (ToolACE style)."""
        conversations = example.get("conversations", [])
        if not conversations:
            return {
                "messages": [
                    _msg("system", DEFAULT_SYSTEM_PROMPT),
                    _msg("user", example.get("query", str(example))),
                    _msg("assistant", ""),
                ]
            }

        messages = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))
            role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool", "observation": "tool"}
            role = role_map.get(role, role)
            messages.append(_msg(role, content))

        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, _msg("system", DEFAULT_SYSTEM_PROMPT))

        return {"messages": messages}

    @staticmethod
    def from_banking77(example: dict) -> dict:
        """Convert Banking77 to intent classification format."""
        text = example.get("text", example.get("customer_message", ""))
        label = example.get("label", "")

        # Banking77 uses integer labels; map to string if available
        if isinstance(label, int):
            label_str = f"banking_intent_{label}"
        else:
            label_str = str(label)

        assistant_output = json.dumps(
            {
                "intent": label_str,
                "urgency": "medium",
                "workflow_steps": ["classify_intent", "route_to_department"],
                "tools_required": [],
                "reasoning": f"Customer intent classified as {label_str}.",
                "response": "Thank you for reaching out. Let me help you with that.",
            },
            indent=2,
        )

        return {
            "messages": [
                _msg("system", DEFAULT_SYSTEM_PROMPT),
                _msg("user", text),
                _msg("assistant", assistant_output),
            ]
        }

    @staticmethod
    def from_clinc(example: dict) -> dict:
        """Convert CLINC OOS to intent classification with out-of-scope handling."""
        text = example.get("text", "")
        intent = example.get("intent", example.get("label", ""))

        # CLINC uses integer labels; 42 = oos in the plus split
        if isinstance(intent, int):
            if intent == 42:
                intent_str = "out_of_scope"
            else:
                intent_str = f"clinc_intent_{intent}"
        else:
            intent_str = str(intent)

        is_oos = intent_str == "out_of_scope" or intent_str == "oos"

        if is_oos:
            assistant_output = json.dumps(
                {
                    "intent": "general_inquiry",
                    "urgency": "low",
                    "workflow_steps": ["escalate_to_supervisor"],
                    "tools_required": [],
                    "reasoning": "This request is outside my area of expertise. Escalating to a human agent.",
                    "response": "I'm not sure I can help with that specific request. Let me connect you with a specialist who can assist you better.",
                },
                indent=2,
            )
        else:
            assistant_output = json.dumps(
                {
                    "intent": intent_str,
                    "urgency": "medium",
                    "workflow_steps": ["classify_intent", "route_to_department"],
                    "tools_required": [],
                    "reasoning": f"Customer intent classified as {intent_str}.",
                    "response": "Thank you for reaching out. Let me help you with that.",
                },
                indent=2,
            )

        return {
            "messages": [
                _msg("system", DEFAULT_SYSTEM_PROMPT),
                _msg("user", text),
                _msg("assistant", assistant_output),
            ]
        }

    @staticmethod
    def from_kiki_annotated(example: dict) -> dict:
        """Convert our custom annotated format from the POC annotator."""
        assistant_output = json.dumps(
            {
                "intent": example.get("intent", "general_inquiry"),
                "urgency": example.get("urgency", "medium"),
                "workflow_steps": example.get("workflow_steps", []),
                "tools_required": example.get("tools_required", []),
                "reasoning": f"Customer intent is {example.get('intent', 'unknown')} with "
                f"{example.get('urgency', 'medium')} urgency. "
                f"Required workflow: {' → '.join(example.get('workflow_steps', []))}.",
                "response": example.get("agent_response", ""),
            },
            indent=2,
        )

        return {
            "messages": [
                _msg("system", DEFAULT_SYSTEM_PROMPT),
                _msg("user", example.get("customer_message", "")),
                _msg("assistant", assistant_output),
            ]
        }

    @staticmethod
    def from_preference_pair(example: dict) -> dict:
        """Convert chosen/rejected format for DPO — passes through as-is."""
        return {
            "prompt": example.get("prompt", []),
            "chosen": example.get("chosen", []),
            "rejected": example.get("rejected", []),
        }

    # ------------------------------------------------------------------
    # Dataset-level processing
    # ------------------------------------------------------------------

    # Converter registry
    _CONVERTERS: dict[str, Any] = {}

    @classmethod
    def _init_converters(cls) -> None:
        if cls._CONVERTERS:
            return
        cls._CONVERTERS = {
            "bitext": cls.from_bitext,
            "ticket": cls.from_ticket,
            "glaive": cls.from_glaive_function_calling,
            "xlam": cls.from_xlam,
            "hermes": cls.from_hermes,
            "toolbench": cls.from_toolbench,
            "toolace": cls.from_toolbench,  # Same format
            "banking77": cls.from_banking77,
            "clinc": cls.from_clinc,
            "kiki_annotated": cls.from_kiki_annotated,
            "preference": cls.from_preference_pair,
            "arcee_agent": cls.from_hermes,  # Similar conversation format
            "t1": cls.from_hermes,  # Similar multi-turn format
            "safety": cls.from_bitext,  # instruction/response format
        }

    @classmethod
    def get_converter(cls, name: str) -> Any:
        cls._init_converters()
        if name not in cls._CONVERTERS:
            raise ValueError(f"Unknown converter '{name}'. Available: {list(cls._CONVERTERS.keys())}")
        return cls._CONVERTERS[name]

    @classmethod
    def process_dataset(cls, dataset: datasets.Dataset, converter_name: str) -> datasets.Dataset:
        """Apply the named converter to every row, keeping only the 'messages' column."""
        converter = cls.get_converter(converter_name)
        logger.info("Converting %d examples with '%s' converter", len(dataset), converter_name)

        def _convert(example: dict) -> dict:
            try:
                return converter(example)
            except Exception as exc:
                logger.warning("Conversion failed: %s", exc)
                return {
                    "messages": [
                        _msg("system", DEFAULT_SYSTEM_PROMPT),
                        _msg("user", str(example)),
                        _msg("assistant", ""),
                    ]
                }

        converted = dataset.map(_convert, remove_columns=dataset.column_names, desc=f"Converting ({converter_name})")
        return converted
