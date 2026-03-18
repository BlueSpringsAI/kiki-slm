# SFT Fine-Tuning Research Findings — Summary

Research compiled from 30+ papers, guides, and practitioner reports (2023-2025).

## The Bottom Line

**270K examples × 3 epochs is 27-54x more data than needed and will likely hurt performance.**

## Actionable Recommendations for Kiki SLM

### 1. Cut dataset to 5-10K curated examples
- LIMA proved 1K curated examples can compete with 100K+ noisy ones
- Task-specific SFT (intent classification, JSON output, tool calling) needs 200-5K per skill
- Quality dominates quantity — always

### 2. Restructure dataset composition
- Drop or heavily downsample `arcee_agent` (28.5% of data, low relevance)
- Keep CS-core sources (bitext_cs, bitext_ecom, support_tickets) as ~60-70%
- Keep function calling data (xlam, hermes) at ~15-20%
- Use temperature sampling (T=2-5) to upweight smaller domains

### 3. Reduce epochs
- For 5-10K curated dataset: 2-3 epochs
- For current 270K: 1 epoch maximum
- Always use `load_best_model_at_end=True`

### 4. Filter by perplexity
- Score examples against Qwen3-4B base model
- Keep mid-range (30th-60th percentile)
- Use Cleanlab for automated quality scoring

### 5. Don't bother with curriculum ordering
- Complexity-stratified training > simple-to-hard ordering
- Score examples by entropy, apply CoT distillation only to hard tail

### 6. Fix Qwen3-specific LoRA issues (CRITICAL)
- Add `embed_tokens` and `lm_head` to `modules_to_save` — without this, EOS token is untrainable and the model generates infinitely
- Add empty `<think></think>` blocks to all training examples (for non-thinking mode) or mix 75/25 thinking/non-thinking
- Set `eos_token = "<|im_end|>"` explicitly — Qwen3 changed the default
- Ensure PAD token differs from EOS token
- Use **right padding** and **assistant-only loss**

## Files in This Folder

| File | Topic |
|---|---|
| `01_dataset_sizing.md` | Optimal SFT dataset sizes — evidence from LIMA, Alpaca, Unsloth, etc. |
| `02_quality_vs_quantity.md` | Why smaller curated datasets outperform larger noisy ones |
| `03_dataset_mixing.md` | Multi-source mixing strategies, temperature sampling, domain balancing |
| `04_epochs_and_overfitting.md` | Epoch recommendations by dataset size, overfitting evidence |
| `05_data_filtering_and_curation.md` | Perplexity filtering, Cleanlab, Argilla, token-level diversity |
| `06_tool_calling_and_structured_output.md` | Function calling and JSON output fine-tuning guidance |
| `07_curriculum_learning.md` | Complexity stratification vs. traditional curriculum learning |
| `08_qwen3_specific.md` | Qwen3 EOS bug, thinking modes, chat template, LoRA config, community findings |

## Key Sources

- [LIMA: Less Is More for Alignment (Meta, 2023)](https://arxiv.org/abs/2305.11206)
- [Unveiling the Secret Recipe: Fine-Tuning Small LLMs (Dec 2024)](https://arxiv.org/abs/2412.13337)
- [How Abilities in LLMs are Affected by SFT Data Composition (ACL 2024)](https://arxiv.org/abs/2310.05492)
- [Massive Supervised Fine-tuning Experiments (EMNLP 2025)](https://arxiv.org/abs/2506.14681)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
- [The Best Instruction-Tuning Data are Those That Fit (GRAPE)](https://arxiv.org/html/2502.04194v2)
- [Data Mixing Optimization for SFT](https://arxiv.org/html/2508.11953v1)
- [From Macro to Micro: Dataset Diversity in Fine-Tuning](https://arxiv.org/abs/2505.24768)
- [Complexity-Aware Fine-Tuning](https://arxiv.org/html/2506.21220)
- [Sebastian Raschka: Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Google: Master Gemini SFT](https://cloud.google.com/blog/products/ai-machine-learning/master-gemini-sft)
- [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 GitHub Discussion #1429: Disabling Thinking During SFT](https://github.com/QwenLM/Qwen3/discussions/1429)
- [QwenLM/Qwen3 Issue #1064: EOS Token Bug](https://github.com/QwenLM/Qwen3/issues/1064)
- [DistilLabs: 12 Small Models Benchmark — Qwen3-4B #1](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning)
- [HuggingFace: Qwen-3 Chat Template Deep Dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
