# Data Filtering and Curation Techniques

## Key Finding: Perplexity-based filtering is the most reliable quality signal

## Perplexity-Based Filtering

Research from "Massive Supervised Fine-tuning Experiments" (EMNLP 2025):
- Datasets with **lower pre-SFT model perplexity** consistently yield greater downstream performance
- Perplexity surpasses semantic similarity (Pearson R = 0.112, non-significant) and response length as a predictor

### Practical Technique
1. Compute perplexity of each example against your base model
2. Keep the **mid-range (30th-60th percentile)**
3. Discard very low perplexity (too easy, model already knows it)
4. Discard very high perplexity (too far out of distribution)

## Selective Token Masking (STM)

Goes further than example-level filtering:
- Compute **token-level perplexity** with an instruction-tuned model
- Mask ground-truth tokens exceeding a threshold during training
- The model only learns from tokens it doesn't already know well
- Result: more efficient learning, less redundant gradient updates

## Decontamination

Use **8-gram matching** to remove any training examples overlapping with evaluation benchmarks.

## Tools for Practitioners

### Cleanlab
Uses a Trustworthy Language Model (TLM) to score each (prompt, response) pair:
- Detects: factual inaccuracy, PII exposure, toxic language, non-English text, informal writing, incomplete responses
- Produces a weighted geometric mean across issue types as composite quality score

```python
from cleanlab_tlm import TLM
tlm = TLM()
scores = tlm.get_trustworthiness_score(prompts, responses)
```

### Argilla
Best for human-in-the-loop curation:
- Configure feedback workflow where labelers rate prompt/response quality
- Aggregate ratings to select highest-quality examples
- Integrates with distilabel for synthetic data generation
- Published a [synthetic SFT customer support dataset](https://huggingface.co/datasets/argilla/synthetic-sft-customer-support-single-turn) as reference

### Lilac (now part of Databricks)
Runs on-device with UI and Python API:
- Semantic clustering
- Duplicate removal
- PII detection
- Topic-level exploration
- Useful for understanding what your dataset actually contains before filtering

## Token-Level Diversity Analysis

From "From Macro to Micro" — a three-level framework:

| Level | Method | What it measures |
|---|---|---|
| **Macroscopic** (instruction-level) | BERTopic + HDBSCAN clustering | Topic coverage |
| **Mesoscopic** (tag-level) | LLM-based tagging | Intent, topic, sentiment distribution |
| **Microscopic** (token-level) | Token frequency bands | Semantic richness |

### Token Bands
- **High-band** (>500 occurrences): Common tokens, low information
- **Mid-band** (10-500): **Most semantic meaning — prioritize these**
- **Low-band** (<10): Rare tokens, potential noise

### Key Metrics (correlation with downstream performance)
| Metric | Correlation |
|---|---|
| Sequence Length mean | 0.73 |
| Information Entropy of tokens | 0.58 |
| Self-BLEU, N-gram Ratio | Weak |

**Critical**: Use your **model's tokenizer**, not word-based tokenization. Word-based tokenization showed no clear positive correlation with performance.

## Recommendation for Kiki SLM

1. Run perplexity scoring against Qwen3-4B on all 270K examples
2. Use Cleanlab to flag low-quality responses
3. Filter to mid-perplexity range (30th-60th percentile)
4. Check token diversity using microscopic analysis
5. Target 5-10K high-quality examples after filtering

## Sources

- [Massive Supervised Fine-tuning Experiments (EMNLP 2025)](https://arxiv.org/abs/2506.14681)
- [Cleanlab: Discover Bad Data in Instruction-Tuning Datasets](https://help.cleanlab.ai/tlm/use-cases/instruction_tuning_data/)
- [Argilla SFT Data Collection Guide](https://docs.v1.argilla.io/en/latest/conceptual_guides/llm/sft.html)
- [Lilac (GitHub)](https://github.com/databricks/lilac)
- [From Macro to Micro: Dataset Diversity in Fine-Tuning](https://arxiv.org/abs/2505.24768)
- [Mitigating Forgetting via Low-Perplexity Token Learning](https://arxiv.org/html/2501.14315v2)
- [Entropic Distribution Matching in SFT](https://arxiv.org/html/2408.16673v1)
