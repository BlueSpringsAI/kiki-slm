# Optimal SFT Dataset Size for 3-4B Models

## Key Finding: You need far less data than 270K

SFT teaches **format and behavior**, not knowledge. The model already "knows" language from pretraining.

## Evidence from Papers & Projects

| Project / Paper | Model Size | SFT Dataset Size | Result |
|---|---|---|---|
| **LIMA** (Meta, 2023) | 65B | **1,000 examples** | Preferred over GPT-4 in 43% of human evals |
| **Alpaca** (Stanford) | 7B | **52,000 examples** | Viable instruction-following from synthetic data |
| **Vicuna** | 7B | **~70,000 conversations** | ShareGPT data |
| **Zephyr-7B** (HuggingFace) | 7B | **~200K examples** | UltraChat-200k filtered |
| **Secret Recipe** (Dec 2024) | 3B-7B | Systematic study | Stacked training is simpler and more sample-efficient than phased |

## LIMA's "Superficial Alignment Hypothesis"

Almost all knowledge is learned during pretraining. SFT's role is to teach the **format** of interaction, not new knowledge:
- **Format/style alignment**: 1,000-10,000 high-quality examples is sufficient
- **New domain skills** (tool calling, specific API schemas): more data per skill, but gains are logarithmic
- Math and code improve consistently with more data; general ability saturates around ~1,000 samples

## Unsloth's Guidance

- Minimum 100 rows for any signal
- 1,000+ rows recommended
- "500 high-quality examples beats 5,000 noisy ones"

## Task-Specific Benchmarks

| Task Type | Examples Needed | Evidence |
|-----------|----------------|----------|
| Classification | 100-300 per category | Llama 3.1 8B hit 92% accuracy with 150 examples (LoRA) |
| Structured output (JSON) | 200-500 | Sufficient for JSON/HTML generation |
| Content generation | 500-2,000 | Broader response variety needed |
| Complex domain adaptation | 1,000-5,000 | Customer service, multi-turn reasoning |
| Full general-purpose SFT | 5,000-50,000 | Depends on domain breadth |

## Real-World Case Studies

- **Qwen 2.5 7B for customer support**: 400 examples across English, Spanish, Mandarin
- **Logistics company (Llama-based)**: 280 curated examples, 94% accuracy, trained in an afternoon
- **Granite 3B/7B, Llama 3.2 3B, Mistral 7B** (IBM): Taxonomy-driven approach across three phases — instruction following (308K), foundational knowledge (231K), complex skills (286K)

## Recommendation for Kiki SLM

**Cut from 270K to 5K-10K curated examples.** Current dataset is 27-54x larger than needed. Invest the saved compute time in data curation instead.

## Sources

- [LIMA: Less Is More for Alignment (Meta, 2023)](https://arxiv.org/abs/2305.11206)
- [Unveiling the Secret Recipe: Fine-Tuning Small LLMs (Dec 2024)](https://arxiv.org/abs/2412.13337)
- [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Unsloth Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- [How Much Data to Fine-Tune an LLM (Particula)](https://particula.tech/blog/how-much-data-fine-tune-llm)
