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

    # Banking77 label names from the ClassLabel feature
    _BANKING77_LABELS = [
        "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
        "automatic_top_up", "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
        "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
        "card_delivery_estimate", "card_linking", "card_not_working",
        "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
        "contactless_not_working", "country_support", "declined_card_payment",
        "declined_cash_withdrawal", "declined_transfer",
        "direct_debit_payment_not_recognised", "disposable_card_limits",
        "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
        "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
        "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
        "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
        "order_physical_card", "passcode_forgotten", "pending_card_payment",
        "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
        "receiving_money", "refund_not_showing_up", "request_refund",
        "reverted_card_payment", "supported_cards_and_currencies", "terminate_account",
        "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
        "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
        "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
        "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
        "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
        "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
        "why_verify_identity", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ]

    # Map banking77 intents to Kiki urgency/workflow/tools
    _BANKING77_URGENCY = {
        "compromised_card": "critical", "lost_or_stolen_card": "critical",
        "lost_or_stolen_phone": "critical", "pin_blocked": "high",
        "card_not_working": "high", "declined_card_payment": "high",
        "failed_transfer": "high", "transaction_charged_twice": "high",
        "card_swallowed": "high",
    }

    @staticmethod
    def from_banking77(example: dict) -> dict:
        """Convert Banking77 to intent classification with real labels and responses."""
        text = example.get("text", example.get("customer_message", ""))
        label = example.get("label", "")

        # Resolve integer label to real name
        if isinstance(label, int) and 0 <= label < len(ChatMLConverter._BANKING77_LABELS):
            intent = ChatMLConverter._BANKING77_LABELS[label]
        else:
            intent = str(label).lower().replace(" ", "_")

        # Map to Kiki intent categories
        kiki_intent = "account_management"
        if any(kw in intent for kw in ("refund", "charge", "fee", "exchange")):
            kiki_intent = "billing_inquiry"
        elif any(kw in intent for kw in ("card", "pin", "contactless", "visa", "mastercard")):
            kiki_intent = "account_management"
        elif any(kw in intent for kw in ("transfer", "top_up", "receiving", "balance")):
            kiki_intent = "payment_issue"
        elif any(kw in intent for kw in ("lost", "stolen", "compromised")):
            kiki_intent = "fraud_report"
        elif any(kw in intent for kw in ("verify", "identity")):
            kiki_intent = "account_management"
        elif any(kw in intent for kw in ("terminate", "cancel")):
            kiki_intent = "cancellation"

        urgency = ChatMLConverter._BANKING77_URGENCY.get(intent, "medium")
        human_intent = intent.replace("_", " ")

        assistant_output = json.dumps(
            {
                "intent": kiki_intent,
                "urgency": urgency,
                "workflow_steps": ["verify_identity", f"investigate_{intent}", "resolve_issue", "notify_customer"],
                "tools_required": ["customer_profile_api", "payment_gateway_api"],
                "reasoning": f"Customer is asking about {human_intent}. This is a {kiki_intent} case with {urgency} urgency.",
                "response": f"I understand you're experiencing an issue with {human_intent}. Let me look into this for you right away. I'll pull up your account details and work on resolving this as quickly as possible.",
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

    # CLINC intent names from ClassLabel
    _CLINC_LABELS = [
        "restaurant_reviews", "nutrition_info", "account_blocked", "oil_change_how",
        "time", "weather", "redeem_rewards", "interest_rate", "gas_type",
        "accept_reservations", "smart_home", "user_name", "report_lost_card",
        "repeat", "whisper_mode", "what_are_your_hobbies", "order", "jump_start",
        "schedule_meeting", "meeting_schedule", "freeze_account", "what_song",
        "meaning_of_life", "restaurant_reservation", "traffic", "make_call", "text",
        "bill_balance", "improve_credit_score", "change_language", "no",
        "measurement_conversion", "timer", "flip_coin", "do_you_have_pets", "balance",
        "tell_joke", "last_maintenance", "exchange_rate", "uber", "car_rental",
        "credit_limit", "oos", "shopping_list", "expiration_date", "routing",
        "meal_suggestion", "tire_change", "todo_list", "card_declined",
        "rewards_balance", "change_accent", "vaccines", "reminder_update", "food_last",
        "change_ai_name", "bill_due", "who_do_you_work_for", "share_location",
        "international_visa", "calendar", "translate", "carry_on", "book_flight",
        "insurance_change", "todo_list_update", "timezone", "cancel_reservation",
        "transactions", "credit_score", "report_fraud", "spending_history", "directions",
        "spelling", "insurance", "what_is_your_name", "reminder", "where_are_you_from",
        "distance", "payday", "flight_status", "find_phone", "greeting", "alarm",
        "order_status", "confirm_reservation", "cook_time", "damaged_card",
        "reset_settings", "pin_change", "replacement_card_duration", "new_card",
        "roll_dice", "income", "taxes", "date", "who_made_you", "pto_request",
        "tire_pressure", "how_old_are_you", "rollover_401k", "pto_request_status",
        "how_busy", "application_status", "recipe", "calendar_update", "play_music",
        "yes", "direct_deposit", "credit_limit_change", "gas", "pay_bill",
        "ingredients_list", "lost_luggage", "goodbye", "what_can_i_ask_you",
        "book_hotel", "are_you_a_bot", "next_song", "change_speed", "plug_type",
        "maybe", "w2", "oil_change_when", "thank_you", "shopping_list_update",
        "pto_balance", "order_checks", "travel_alert", "fun_fact", "sync_device",
        "schedule_maintenance", "apr", "transfer", "ingredient_substitution",
        "calories", "current_location", "international_fees", "calculator",
        "definition", "next_holiday", "update_playlist", "mpg", "min_payment",
        "change_user_name", "restaurant_suggestion", "travel_notification", "cancel",
        "pto_used", "travel_suggestion", "change_volume",
    ]

    # Map CLINC intents to Kiki categories
    _CLINC_TO_KIKI = {
        "order_status": "order_status", "order": "order_status",
        "bill_balance": "billing_inquiry", "bill_due": "billing_inquiry",
        "pay_bill": "billing_inquiry", "min_payment": "billing_inquiry",
        "transactions": "billing_inquiry", "spending_history": "billing_inquiry",
        "report_fraud": "fraud_report", "report_lost_card": "fraud_report",
        "freeze_account": "fraud_report",
        "account_blocked": "account_management", "pin_change": "account_management",
        "reset_settings": "account_management", "change_user_name": "account_management",
        "cancel": "cancellation", "cancel_reservation": "cancellation",
        "damaged_card": "return_request", "new_card": "product_inquiry",
        "replacement_card_duration": "shipping_issue",
        "card_declined": "payment_issue", "transfer": "payment_issue",
        "balance": "billing_inquiry", "credit_score": "billing_inquiry",
        "credit_limit": "billing_inquiry", "credit_limit_change": "billing_inquiry",
        "interest_rate": "billing_inquiry", "apr": "billing_inquiry",
        "direct_deposit": "payment_issue", "redeem_rewards": "billing_inquiry",
        "rewards_balance": "billing_inquiry",
        "insurance": "product_inquiry", "insurance_change": "product_inquiry",
        "international_fees": "billing_inquiry", "exchange_rate": "billing_inquiry",
    }

    @staticmethod
    def from_clinc(example: dict) -> dict:
        """Convert CLINC OOS to intent classification with real labels and responses."""
        text = example.get("text", "")
        intent_raw = example.get("intent", example.get("label", ""))

        # Resolve integer label to real name
        if isinstance(intent_raw, int) and 0 <= intent_raw < len(ChatMLConverter._CLINC_LABELS):
            intent = ChatMLConverter._CLINC_LABELS[intent_raw]
        else:
            intent = str(intent_raw).lower().replace(" ", "_")

        is_oos = intent == "oos"

        if is_oos:
            assistant_output = json.dumps(
                {
                    "intent": "general_inquiry",
                    "urgency": "low",
                    "workflow_steps": ["assess_request", "escalate_to_supervisor"],
                    "tools_required": ["ticket_update_api"],
                    "reasoning": "This request is outside the standard service categories. Escalating to a human agent for proper handling.",
                    "response": "I appreciate you reaching out. This particular request falls outside my area of expertise, so let me connect you with a specialist who can assist you properly. I'll make sure your case is prioritized.",
                },
                indent=2,
            )
        else:
            # Map to Kiki intent or use general_inquiry
            kiki_intent = ChatMLConverter._CLINC_TO_KIKI.get(intent, "general_inquiry")
            human_intent = intent.replace("_", " ")

            assistant_output = json.dumps(
                {
                    "intent": kiki_intent,
                    "urgency": "medium",
                    "workflow_steps": ["verify_identity", f"handle_{intent}", "resolve_issue", "notify_customer"],
                    "tools_required": ["customer_profile_api"],
                    "reasoning": f"Customer is asking about {human_intent}. Classified as {kiki_intent}.",
                    "response": f"I'd be happy to help you with {human_intent}. Let me pull up your account information and get this sorted out for you right away.",
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
