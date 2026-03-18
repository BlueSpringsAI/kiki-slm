# Data Quality vs Quantity for SFT

## Key Finding: Smaller curated datasets consistently outperform larger noisy ones

## Evidence

### LIMA (Meta, 2023)
1,000 curated examples matched or beat models trained on orders of magnitude more data with RLHF.

### "Long Is More for Alignment" (2024)
Fine-tuning on only the **longest 1,000 instructions** from Alpaca (52K total) produced a stronger model than training on all 52K. Length serves as a proxy for response quality and complexity.

### Sebastian Raschka (LoRA experiments)
Doubling iterations on a 50K Alpaca dataset (equivalent to 2 epochs) caused performance to **decline** due to overfitting. Recommends: "If overfitting, decrease LoRA rank r or increase dataset size."

### Google (Gemini SFT guide)
> "A smaller, refined and representative dataset often outperforms a large, noisy one."

Start with ~100 examples to validate, then scale to cover corner cases.

### Parlance Labs (function calling)
> "Reducing the amount of training data while preparing higher quality data yields better results than tweaking hyperparameters."

Minimum effective dataset for function calling: **100 samples**.

### GRAPE Method (2025)
For each instruction, gather responses from multiple LLMs, then select the one with highest probability under the target base model. Achieved **6.1% improvement over a 4.5x larger dataset** on Llama 3.1-8B, and outperformed training on 1.58M examples with only 350K on Qwen 2.5-7B.

## Counterintuitive Finding

"Rethinking Data Selection for SFT" found that simply selecting the **1K examples with the longest responses** from a 52K dataset outperformed:
- Quality-based selection (using ChatGPT ratings)
- Diversity-based selection (K-Means clustering)

Quality-based selection was inconsistent and sometimes underperformed random selection. Length is a surprisingly strong quality proxy.

## Practical Implication

Deduplicate aggressively. Google's guide identifies deduplication as "one of the most crucial pre-processing steps" — duplicates lead to memorization and wasted compute.

## Sources

- [LIMA (Meta, 2023)](https://arxiv.org/abs/2305.11206)
- [Long Is More for Alignment (2024)](https://arxiv.org/html/2402.04833v2)
- [Sebastian Raschka: Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Google: Master Gemini SFT](https://cloud.google.com/blog/products/ai-machine-learning/master-gemini-sft)
- [Parlance Labs: Fine Tuning LLMs for Function Calling](https://parlance-labs.com/education/fine_tuning/pawel.html)
- [The Best Instruction-Tuning Data are Those That Fit (GRAPE)](https://arxiv.org/html/2502.04194v2)
- [Rethinking Data Selection for SFT](https://arxiv.org/html/2402.06094v1)
