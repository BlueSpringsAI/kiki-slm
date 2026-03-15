"""LLM-as-judge implementation for response quality scoring.

Task 3.23: Score responses on helpfulness, correctness,
professionalism, and empathy using GPT-4o or Claude.
"""

from __future__ import annotations

import json
import logging

from kiki.data.validators import QualityScore

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are an expert customer service quality evaluator. Score the AI agent's response on these dimensions (1-5 each):

1. **Helpfulness** (1-5): Does the response address the customer's actual need? Does it provide actionable information?
2. **Correctness** (1-5): Is the information accurate? Are the right tools/workflows suggested?
3. **Professionalism** (1-5): Is the tone appropriate for business communication? Is the formatting clean?
4. **Empathy** (1-5): Does the response acknowledge the customer's feelings? Is it warm without being insincere?

Respond with ONLY a JSON object:
{"helpfulness": <int>, "correctness": <int>, "professionalism": <int>, "empathy": <int>}"""

_PAIRWISE_SYSTEM_PROMPT = """You are an expert customer service evaluator. Compare two AI agent responses to the same customer message. Which response is better overall?

Respond with ONLY one of: "A", "B", or "tie"."""


class LLMJudge:
    """LLM-as-judge for response quality evaluation."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o") -> None:
        self.provider = provider
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI

                self._client = OpenAI()
            elif self.provider == "anthropic":
                from anthropic import Anthropic

                self._client = Anthropic()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def score_response(
        self,
        customer_message: str,
        response: str,
        context: dict | None = None,
    ) -> QualityScore:
        """Score a single response on 4 quality dimensions."""
        user_prompt = f"Customer message: {customer_message}\n\nAI Agent response: {response}"
        if context:
            user_prompt += f"\n\nAdditional context: {json.dumps(context)}"

        raw = self._call_judge(_JUDGE_SYSTEM_PROMPT, user_prompt)

        try:
            scores = json.loads(raw)
            return QualityScore(
                helpfulness=float(scores["helpfulness"]),
                correctness=float(scores["correctness"]),
                professionalism=float(scores["professionalism"]),
                empathy=float(scores["empathy"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse judge response: %s", exc)
            return QualityScore(helpfulness=3.0, correctness=3.0, professionalism=3.0, empathy=3.0)

    def batch_score(self, examples: list[dict]) -> list[QualityScore]:
        """Score multiple examples. Each dict should have 'customer_message' and 'response'."""
        scores = []
        for ex in examples:
            score = self.score_response(
                customer_message=ex["customer_message"],
                response=ex["response"],
                context=ex.get("context"),
            )
            scores.append(score)
        return scores

    def pairwise_compare(self, prompt: str, response_a: str, response_b: str) -> str:
        """Return 'A', 'B', or 'tie'."""
        user_prompt = (
            f"Customer message: {prompt}\n\n"
            f"Response A: {response_a}\n\n"
            f"Response B: {response_b}"
        )

        raw = self._call_judge(_PAIRWISE_SYSTEM_PROMPT, user_prompt).strip().upper()

        if raw in ("A", "B", "TIE"):
            return raw
        logger.warning("Unexpected pairwise result: '%s', defaulting to 'tie'", raw)
        return "tie"

    def _call_judge(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single LLM call."""
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            return resp.choices[0].message.content or ""
        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return resp.content[0].text
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
