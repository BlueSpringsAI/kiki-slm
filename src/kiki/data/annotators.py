"""Production LLM annotation for the Kiki SLM pipeline.

Task 3.4: Async batched annotation using GPT-4o-mini or Claude,
with structured output parsing, retry logic, quality scoring,
preference ranking, and checkpoint-based resume.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import datasets
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from kiki.data.validators import QualityScore

logger = logging.getLogger(__name__)


class LLMAnnotator:
    """Batched async annotation using OpenAI or Anthropic APIs."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        max_concurrent: int = 20,
        checkpoint_every: int = 100,
        checkpoint_dir: str = "data/.checkpoints",
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.max_concurrent = max_concurrent
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.temperature = temperature
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    # ------------------------------------------------------------------
    # Core async call with retry
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=30))
    async def _call_openai(self, user_content: str, response_format: dict | None = None) -> tuple[str, int, int]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=30))
    async def _call_anthropic(self, user_content: str) -> tuple[str, int, int]:
        client = self._get_client()
        resp = await client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        text = resp.content[0].text if resp.content else ""
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    async def _call_llm(self, user_content: str, response_format: dict | None = None) -> str:
        if self.provider == "openai":
            text, inp, out = await self._call_openai(user_content, response_format)
        else:
            text, inp, out = await self._call_anthropic(user_content)
        self._total_input_tokens += inp
        self._total_output_tokens += out
        return text

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, task_id: str) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / f"{task_id}.jsonl"

    def _load_checkpoint(self, task_id: str) -> list[dict]:
        path = self._checkpoint_path(task_id)
        if not path.exists():
            return []
        results = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        logger.info("Resumed %d items from checkpoint '%s'", len(results), path)
        return results

    def _save_checkpoint(self, task_id: str, results: list[dict]) -> None:
        path = self._checkpoint_path(task_id)
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # annotate_tickets
    # ------------------------------------------------------------------

    def annotate_tickets(
        self,
        dataset: datasets.Dataset,
        schema: type[BaseModel],
        task_id: str = "annotate",
    ) -> datasets.Dataset:
        """Annotate all tickets in *dataset* using the configured LLM.

        Returns a new Dataset with the original columns plus annotation fields.
        """
        return asyncio.run(self._annotate_tickets_async(dataset, schema, task_id))

    async def _annotate_tickets_async(
        self,
        dataset: datasets.Dataset,
        schema: type[BaseModel],
        task_id: str,
    ) -> datasets.Dataset:
        sem = asyncio.Semaphore(self.max_concurrent)
        existing = self._load_checkpoint(task_id)
        start_idx = len(existing)
        results = list(existing)

        async def _process(idx: int, item: dict) -> dict:
            async with sem:
                msg = item.get("customer_message", "")
                resp = item.get("agent_response", "")
                user_content = f"Customer message: {msg}\nAgent response: {resp}"

                text = await self._call_llm(
                    user_content,
                    response_format={"type": "json_object"} if self.provider == "openai" else None,
                )

                try:
                    parsed = json.loads(text)
                    schema.model_validate(parsed)
                except Exception as exc:
                    logger.warning("Row %d: annotation failed validation: %s", idx, exc)
                    parsed = {"_raw_response": text, "_error": str(exc)}

                merged = {**item, **parsed}
                return merged

        t0 = time.monotonic()
        pending = list(range(start_idx, len(dataset)))

        for batch_start in range(0, len(pending), self.checkpoint_every):
            batch_indices = pending[batch_start : batch_start + self.checkpoint_every]
            tasks = [_process(i, dict(dataset[i])) for i in batch_indices]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, res in zip(batch_indices, batch_results):
                if isinstance(res, Exception):
                    logger.error("Row %d failed: %s", i, res)
                    results.append({**dict(dataset[i]), "_error": str(res)})
                else:
                    results.append(res)

            self._save_checkpoint(task_id, results)
            elapsed = time.monotonic() - t0
            logger.info(
                "Progress: %d/%d (%.1fs elapsed, ~%.0f tokens used)",
                len(results),
                len(dataset),
                elapsed,
                self._total_input_tokens + self._total_output_tokens,
            )

        logger.info(
            "Annotation complete: %d items, %d input tokens, %d output tokens",
            len(results),
            self._total_input_tokens,
            self._total_output_tokens,
        )
        return datasets.Dataset.from_list(results)

    # ------------------------------------------------------------------
    # score_responses
    # ------------------------------------------------------------------

    def score_responses(self, pairs: list[dict]) -> list[QualityScore]:
        """Score response quality on helpfulness, correctness, professionalism, empathy."""
        return asyncio.run(self._score_responses_async(pairs))

    async def _score_responses_async(self, pairs: list[dict]) -> list[QualityScore]:
        sem = asyncio.Semaphore(self.max_concurrent)
        scoring_prompt = (
            "Rate the following customer service response on a scale of 1-5 for each dimension.\n"
            "Respond ONLY with valid JSON: {\"helpfulness\": N, \"correctness\": N, "
            "\"professionalism\": N, \"empathy\": N}\n\n"
        )

        async def _score_one(pair: dict) -> QualityScore:
            async with sem:
                user_content = (
                    f"{scoring_prompt}"
                    f"Customer message: {pair.get('customer_message', '')}\n"
                    f"Agent response: {pair.get('agent_response', '')}"
                )
                text = await self._call_llm(
                    user_content,
                    response_format={"type": "json_object"} if self.provider == "openai" else None,
                )
                data = json.loads(text)
                return QualityScore.model_validate(data)

        tasks = [_score_one(p) for p in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Scoring failed for pair %d: %s", i, r)
                scores.append(QualityScore(helpfulness=3.0, correctness=3.0, professionalism=3.0, empathy=3.0))
            else:
                scores.append(r)
        return scores

    # ------------------------------------------------------------------
    # rank_preferences
    # ------------------------------------------------------------------

    def rank_preferences(self, prompt: str, response_a: str, response_b: str) -> str:
        """Return 'a', 'b', or 'tie' indicating which response is better."""
        return asyncio.run(self._rank_async(prompt, response_a, response_b))

    async def _rank_async(self, prompt: str, response_a: str, response_b: str) -> str:
        user_content = (
            "Compare two customer service responses to the same query.\n"
            "Respond ONLY with JSON: {\"winner\": \"a\" or \"b\" or \"tie\", \"reason\": \"...\"}\n\n"
            f"Customer query: {prompt}\n\n"
            f"Response A: {response_a}\n\n"
            f"Response B: {response_b}"
        )
        text = await self._call_llm(
            user_content,
            response_format={"type": "json_object"} if self.provider == "openai" else None,
        )
        data = json.loads(text)
        winner = data.get("winner", "tie").lower()
        if winner not in ("a", "b", "tie"):
            winner = "tie"
        return winner

    # ------------------------------------------------------------------
    # generate_synthetic_examples
    # ------------------------------------------------------------------

    def generate_synthetic_examples(
        self,
        intent: str,
        count: int = 10,
        difficulty: str = "medium",
    ) -> list[dict]:
        """Generate synthetic customer service examples for a given intent."""
        return asyncio.run(self._generate_synthetic_async(intent, count, difficulty))

    async def _generate_synthetic_async(self, intent: str, count: int, difficulty: str) -> list[dict]:
        user_content = (
            f"Generate {count} realistic customer service tickets with intent '{intent}' "
            f"at {difficulty} difficulty level.\n"
            f"Each ticket should have: customer_message, agent_response, intent, urgency, "
            f"workflow_steps (list), tools_required (list), confidence (float 0-1).\n"
            f"Respond with a JSON array of objects. No other text."
        )
        text = await self._call_llm(
            user_content,
            response_format={"type": "json_object"} if self.provider == "openai" else None,
        )

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "examples" in data:
                data = data["examples"]
            if isinstance(data, dict) and "tickets" in data:
                data = data["tickets"]
            if not isinstance(data, list):
                data = [data]
            return data
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse synthetic examples: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    def get_cost_estimate(self) -> dict[str, Any]:
        """Return estimated cost based on token usage."""
        # Approximate pricing (per 1M tokens)
        pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        }
        rates = pricing.get(self.model, {"input": 1.0, "output": 3.0})
        input_cost = (self._total_input_tokens / 1_000_000) * rates["input"]
        output_cost = (self._total_output_tokens / 1_000_000) * rates["output"]
        return {
            "model": self.model,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
        }
