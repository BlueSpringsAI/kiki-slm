# Epoch Count and Overfitting in SFT

## Key Finding: 3 epochs on 270K examples is almost certainly overfitting

## Evidence

| Source | Recommendation |
|---|---|
| **Unsloth docs** | 1-3 epochs for instruction datasets; >3 epochs has diminishing returns |
| **Sebastian Raschka** | Multi-epoch training on static datasets "might not be beneficial and often deteriorates results, probably due to overfitting" |
| **Google (Gemini SFT)** | Start with 3-5 epochs; more epochs risk overfitting; with more data you need *fewer* epochs |
| **Scaling Data-Constrained LMs** | Up to 4 epochs of repeated data yields negligible loss difference vs. unique data — beyond that, diminishing returns |
| **Practitioner consensus** | For datasets >10K examples: 1-3 epochs. For 100K+: 1 epoch is often optimal |

## The Math on Current Setup

3 epochs × 270K examples = **810K effective training passes**

This is well beyond where research shows saturation for a 3-4B model on task-specific SFT.

## Raschka's LoRA Experiment

When he doubled iterations on a 50K Alpaca dataset (equivalent to 2 epochs), performance **declined**. Explicitly recommends: "If overfitting, decrease LoRA rank r or increase dataset size" as first remedies.

## Recommendation

| Dataset Size | Recommended Epochs |
|---|---|
| <1K examples | 3-5 epochs |
| 1K-10K examples | 2-3 epochs |
| 10K-50K examples | 1-2 epochs |
| 50K+ examples | 1 epoch |

**For Kiki SLM:**
- If you keep 270K: **1 epoch only**
- If you cut to 5-10K curated: **2-3 epochs** is appropriate
- Always monitor validation loss — stop when it plateaus or increases
- Use `load_best_model_at_end=True` for automatic checkpoint selection

## Sources

- [Sebastian Raschka: Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Google: Master Gemini SFT](https://cloud.google.com/blog/products/ai-machine-learning/master-gemini-sft)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
- [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
