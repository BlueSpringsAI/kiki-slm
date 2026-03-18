# Qwen3-Specific Fine-Tuning Guidance

## Qwen3 Post-Training Pipeline (from Technical Report, arxiv 2505.09388)

Alibaba's official 4-stage pipeline for flagship models:

1. **Long-CoT Cold Start** — Curated math/code/logic/STEM dataset. Two-phase filtering (query + response filtering with QwQ-32B)
2. **Reasoning RL** — GRPO with 3,995 query-verifier pairs. Large batch size, high rollouts per query, off-policy training
3. **Thinking Mode Fusion** — Continual SFT combining thinking data (rejection sampling) with diverse non-thinking data
4. **General RL** — RL across 20+ task categories including instruction following and agent capabilities

**For smaller models (Qwen3-4B):** The team uses **strong-to-weak distillation** from Qwen3-32B/235B teachers instead of the full 4-stage pipeline. Achieves comparable quality at **1/10 GPU hours**.

## Qwen3-4B is the Best Small Model for Fine-Tuning

From a [DistilLabs benchmark of 12 small models across 8 tasks](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning):

- **Qwen3-4B-Instruct-2507 ranked #1** after fine-tuning, beating all models in the 1B-8B range including the larger Qwen3-8B
- Fine-tuned Qwen3-4B **matches or exceeds GPT-OSS-120B** (a 30x larger teacher) on 7/8 benchmarks
- On SQuAD 2.0, fine-tuned Qwen3-4B surpassed the teacher by **19 percentage points**
- Smaller models improve **dramatically more** from fine-tuning than larger ones
- Qwen3-4B base matches Qwen2.5-7B base in pre-training benchmarks

Benchmark config: 4 epochs, lr=5e-5, linear scheduler, LoRA rank 64, 10K synthetic examples.

---

## CRITICAL: Thinking Mode and `<think>` Tokens

Qwen3 has a unique thinking/non-thinking mode. This **directly affects SFT**.

### To disable thinking during SFT (Qwen team recommendation, [GitHub #1429](https://github.com/QwenLM/Qwen3/discussions/1429)):

Include an **empty `<think></think>` block** in assistant responses:

```
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
<think>

</think>

{content}<|im_end|>
```

**Do NOT use `/no_think` tokens in prompts** — produces inconsistent results including unexpected Chinese output and intermittent thinking behavior.

### To preserve reasoning capability:

Unsloth recommends a **75% reasoning / 25% non-reasoning** dataset mix. Fine-tuning with only non-reasoning data may degrade reasoning abilities.

### Important detail:

`<think>` and `</think>` are **not special tokenizer tokens** — they are regular XML-style text the model learned during training. The actual special tokens are `<|im_start|>` and `<|im_end|>`.

---

## CRITICAL: EOS Token Bug with LoRA

**The #1 reported issue** with Qwen3 fine-tuning: infinite generation after LoRA training.

### Root cause ([QwenLM/Qwen3 #1064](https://github.com/QwenLM/Qwen3/issues/1064), [LlamaFactory #7943](https://github.com/hiyouga/LlamaFactory/issues/7943)):

When fine-tuning with LoRA, the `embed_tokens` layer is frozen by default. The EOS token `<|im_end|>` shares identical uninitialized weights with many other tokens in the `lm_head`, making it impossible for the model to learn when to stop generating.

### Fix:

Add `embed_tokens` (and optionally `lm_head`) to LoRA targets:

```python
modules_to_save = ["embed_tokens", "lm_head"]
```

### Additional EOS gotcha:

Qwen3 base model EOS token changed from `<|im_end|>` to `<|endoftext|>` in a tokenizer update. The chat template still uses `<|im_end|>`. **Always explicitly set:**

```python
tokenizer.eos_token = "<|im_end|>"
```

And ensure **PAD token differs from EOS token** — if they're the same, EOS gets masked during training and the model never learns to stop.

---

## Chat Template (ChatML format)

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>
{reasoning or empty}
</think>

{response}<|im_end|>
```

### Gotchas ([HuggingFace deep dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)):

1. Dynamic context management — template strips earlier `<think>` blocks in multi-turn to save tokens, preserves only the most recent one
2. Tool arguments need careful serialization (checks if string vs object to avoid double-escaping)
3. **No default system prompt** (unlike Qwen2.5)
4. Use **right padding** — Qwen3 works best with right padding

---

## Function Calling / Tool Use

Qwen3 has **native function calling support** using Hermes-style format.

### Tool call format (assistant response):
```xml
<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
</tool_call>
```

### Tool result format:
```xml
<tool_response>
{"temperature": 22, "condition": "sunny"}
</tool_response>
```

Supports **parallel tool calling** and **multi-turn/multi-step tool calling**. The Qwen team recommends using **Qwen-Agent** for agentic applications. Fine-tuning is described as "the ultimate solution" for domain-specific tool use.

---

## Recommended LoRA Hyperparameters for Qwen3-4B

| Parameter | Conservative | Aggressive |
|-----------|-------------|------------|
| LoRA rank (r) | 16 | 64 |
| LoRA alpha | 16-32 (1-2x rank) | 64-128 |
| LoRA dropout | 0.05-0.1 | 0.0 |
| Learning rate | 1e-4 to 2e-4 | 5e-5 |
| Weight decay | 0.01 | 0.01 |
| Max grad norm | 0.3 | 0.3 |
| Scheduler | Cosine | Cosine |
| Warmup | 3-5% of steps | 3-5% |
| Epochs | 1-2 | 1-4 |
| Max seq length | 2048 | 8192-16384 |
| Quantization | 4-bit (QLoRA) | 4-bit (QLoRA) |

### Target modules:
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
modules_to_save = ["embed_tokens", "lm_head"]  # CRITICAL
```

---

## Community Experiences

### [nixiesearch — Fine-tuning Qwen3 at Home](https://nixiesearch.substack.com/p/fine-tuning-qwen3-at-home-to-respond)
- Fine-tuned Qwen3 1.7B-32B on 53K examples with QLoRA (r=16, alpha=32)
- Used **right padding** and **assistant-only loss** (train only on response, not prompt) — described as "crucial for conversational fine-tuning"
- 32B with QLoRA used only ~20GB VRAM on RTX 5090
- DPO post-training significantly improved quality over SFT alone
- Classical DPO caused crashes; **reference-free DPO** proved stable

### Known Issues Summary
| Issue | Cause | Fix |
|---|---|---|
| Infinite generation | EOS token embedding not trained | Add `embed_tokens` to `modules_to_save` |
| Chinese output after fine-tuning | Using `/no_think` tokens | Use empty `<think></think>` blocks instead |
| Generation never stops | PAD == EOS masking | Set different PAD and EOS tokens |
| Degraded reasoning | SFT with only non-thinking data | Mix 75% reasoning / 25% non-reasoning data |

---

## Implications for Kiki SLM

1. **Your current setup is missing `modules_to_save`** — you may hit the infinite generation bug. Add `embed_tokens` and `lm_head` to LoRA config.

2. **Your training data likely lacks `<think></think>` blocks** — the model may behave unpredictably at inference. Either add empty `<think></think>` to all training examples (non-thinking mode) or mix 75/25 thinking/non-thinking.

3. **`lora-r 16` is fine for Qwen3-4B on T4** — DistilLabs used r=64 but with more VRAM. r=16 with alpha=32 is the conservative sweet spot.

4. **Consider assistant-only loss** — training on the full prompt wastes compute on tokens the model already handles well.

5. **Qwen3-4B is confirmed the best choice** in its class — no need to switch models.

## Sources

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 Official Blog](https://qwenlm.github.io/blog/qwen3/)
- [HuggingFace: Qwen-3 Chat Template Deep Dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [Qwen3 GitHub Discussion #1429](https://github.com/QwenLM/Qwen3/discussions/1429)
- [QwenLM/Qwen3 Issue #1064](https://github.com/QwenLM/Qwen3/issues/1064)
- [LlamaFactory Issue #7943](https://github.com/hiyouga/LlamaFactory/issues/7943)
- [Qwen3 EOS Token Change Analysis](https://kaitchup.substack.com/p/qwen3-when-im_end-suddenly-becomes)
- [Unsloth: Qwen3 Fine-Tuning Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [Qwen Function Calling Docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [DistilLabs: 12 Small Models Benchmark](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning)
- [Fine-Tuning Qwen3 at Home (nixiesearch)](https://nixiesearch.substack.com/p/fine-tuning-qwen3-at-home-to-respond)
- [Fine-Tuning Qwen-3 with Hybrid Reasoning Guide](https://atalupadhyay.wordpress.com/2025/05/07/fine-tuning-qwen-3-with-hybrid-reasoning-a-comprehensive-guide/)
