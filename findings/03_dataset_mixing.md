# Multi-Source Dataset Mixing for SFT

## Key Finding: Absolute per-domain quantity matters more than ratios

## Critical Research: "How Abilities in LLMs are Affected by SFT Data Composition" (Dong et al., ACL 2024)

- At **low data volumes**, mixing domains is synergistic — all abilities improve together
- At **high data volumes**, mixing causes **ability conflicts** — gains in one domain come at the expense of another
- The absolute amount of data per domain matters more than the ratio between domains
- General human-alignment ability saturates at ~1,000 samples
- Math reasoning and code generation improve consistently with more data

## The Cocktail Effect

Even for domain-specific models, mixing in general data helps:
- A medical QA model performed best with **67.7% general instruction data + 32.3% medical data**
- This beat 100% medical data

## Dual-Stage Mixed Fine-tuning (DMT)

Recommended approach:
1. **Stage 1**: Train on domain-specific data (customer service, tool calling, intent classification)
2. **Stage 2**: Train on general capability data mixed with a small proportion (~0.4%) of domain-specific data as rehearsal to prevent catastrophic forgetting

## Temperature Sampling for Mixing

Given K source datasets with sizes n_1, ..., n_K:
- Sampling probability for source i ∝ n_i^(1/T) where T is temperature
- **T=1**: Sample proportionally to dataset size (large datasets dominate)
- **T→∞**: Sample uniformly across sources
- **T<1**: Largest dataset dominates even more
- **Recommended T=2-5** to upweight smaller domains without making them fully equal

## What Happens When One Source Dominates

Tulu3 research assigned ~50% to math and found this **hurt math performance** while raising overall loss. Without optimization, naive proportional sampling lets the largest source dominate training.

## Practical Optimization Workflow

1. Sample equal proportions from each domain
2. Train models with 5 perturbation ratios per domain (small-scale experiments)
3. Fit loss function parameters using the scaling law L(D) = C * |D|^(-β) + E
4. Optimize domain weights using SLSQP

## Current Kiki Dataset Problem

| Source | Current % | Relevance | Recommended Action |
|--------|-----------|-----------|-------------------|
| arcee_agent | 28.5% | Low (generic agent) | Heavy downsample or drop |
| xlam_60k | 15.6% | Medium (function calling) | Keep 1-3K examples |
| bitext_ecom | 15.0% | High | Core — keep 2-3K |
| bitext_insurance | 10.0% | Medium | Keep 1-2K |
| bitext_cs | 9.0% | High | Core — keep 2-3K |
| bitext_banking | 8.5% | Medium | Keep 1-2K |
| support_tickets | 8.0% | High | Core — keep 2-3K |
| banking77 | 3.3% | Medium | Keep 500-1K |
| clinc_oos | 1.5% | Low-Medium | Drop or keep 200-500 |
| hermes_fc | 0.6% | Medium (tool calling) | Keep all 1,687 |

## Recommendation for Kiki SLM

Restructure to ~8-15K total examples with domain-aware weighting:
- **Core CS data** (bitext_cs, bitext_ecom, support_tickets): ~60-70%
- **Tool/function calling** (xlam, hermes_fc): ~15-20%
- **Secondary domains** (banking, insurance): ~10-15%
- **General agent data** (arcee): drop or keep <500 examples

## Sources

- [How Abilities in LLMs are Affected by SFT Data Composition (ACL 2024)](https://arxiv.org/abs/2310.05492)
- [Data Mixing Optimization for SFT](https://arxiv.org/html/2508.11953v1)
- [Massive Supervised Fine-tuning Experiments (EMNLP 2025)](https://arxiv.org/abs/2506.14681)
- [Improved SFT to Mitigate Catastrophic Forgetting (2025)](https://arxiv.org/abs/2506.09428)
