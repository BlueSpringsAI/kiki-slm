# Curriculum Learning for SFT

## Key Finding: Skip ordering, use complexity stratification instead

## Traditional Curriculum Learning (Simple → Hard) is Fragile

Research from "Complexity-Aware Fine-Tuning" tested on Qwen 3B and Phi4-mini:
- Qwen 3B **plateaued at lower accuracy** with more epochs
- Phi4-mini **quickly overfit**
- The approach is extremely sensitive to hyperparameters (epoch counts, learning rates)

## What Works Instead: Complexity-Stratified Training

Rather than ordering data by difficulty, **split by difficulty and apply different techniques**:

| Difficulty Band | Training Approach |
|---|---|
| Easy/Medium examples | Standard SFT is sufficient |
| Hard examples | Chain-of-thought distillation from a large LLM |

### Results
- MMLU-Pro accuracy: **0.52 vs 0.39** for standard SFT (Qwen 3B)
- GSM8K accuracy: **0.82 vs 0.13** for standard SFT (Llama 3B)
- Used **81% less data** than full distillation

## Measuring Example Complexity

**Single-token answer entropy** proved most effective (ROC AUC 0.73):
1. Prompt the model for a direct answer
2. Calculate entropy across the token probability distribution
3. High entropy = hard example, low entropy = easy example

This outperformed model-as-judge approaches (ROC AUC ~0.55).

## Recommendation for Kiki SLM

1. Don't bother with simple-to-hard ordering of training data
2. Score examples by single-token entropy against Qwen3-4B base
3. Apply standard SFT to easy/medium examples
4. For hard examples (edge cases, complex multi-tool scenarios), consider generating chain-of-thought reasoning from a larger model (e.g., Claude or GPT-4) and including the reasoning in training data

## Sources

- [Complexity-Aware Fine-Tuning](https://arxiv.org/html/2506.21220)
