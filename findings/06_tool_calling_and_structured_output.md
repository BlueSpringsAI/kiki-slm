# Fine-Tuning for Tool Calling and Structured JSON Output

## Key Finding: 100-3K examples is sufficient for function calling with LoRA

## Tool / Function Calling

### Parlance Labs & W&B Guidance
- Start with **100 examples minimum** for basic function calling
- Use the **Instruct** version of the base model (not Base) when mixing general chat with function calling
- Introduce a special **function-call token** to the vocabulary to prefix function calls — enables easier parsing and mode-switching during streaming
- LoRA is recommended for function calling fine-tuning, especially in limited data regimes
- Focus on identifying examples where function calling succeeds and filtering out failures

### Recommended Dataset Construction
1. **Intent classification examples**: ~2,000-5,000 covering your full intent taxonomy with edge cases
2. **Tool/function calling examples**: ~1,000-3,000 showing correct tool selection, parameter extraction, and multi-step tool use
3. **General conversation/response examples**: ~1,000-2,000 for style, tone, and non-tool-use responses
4. **Error handling / refusal examples**: ~500-1,000 showing graceful handling of out-of-scope queries

**Total: 5,000-10,000 high-quality, deduplicated examples**

## Structured JSON Output

### Fine-Tuning Approach
- Train on examples that include the JSON structure in responses
- Model learns the schema implicitly through exposure

### Constrained Decoding (Inference-Time Alternative)
Tools like Outlines, vLLM, or SGLang guarantee 100% schema compliance at inference by masking invalid tokens.

**Warning (2025 research):** Restrictive output schemas can **degrade model performance** because the model diverts probability mass toward controlling syntax rather than task accuracy.

**Recommendation:** Fine-tune on JSON examples for the model to learn the pattern, but also consider constrained decoding at inference as a safety net rather than relying solely on the fine-tune.

## Intent Classification

- A fine-tuned Mistral model achieved **100% intent detection** on a customer support task
- 8% better than GPT-4o, 13% better than base model
- Achieved with **~2,000 labeled examples** mapped to intents
- For a 3B model (e.g., Llama 3.2 3B), similar results are achievable

## Sources

- [Parlance Labs: Fine Tuning LLMs for Function Calling](https://parlance-labs.com/education/fine_tuning/pawel.html)
- [W&B: Fine-tuning LLMs for Function Calling](https://wandb.ai/wandb/function-calling-finetuning/reports/Fine-tuning-LLMs-for-function-calling--VmlldzoxMjgxMTgxMg)
- [Baseten: How to Build Function Calling for Open-Source LLMs](https://www.baseten.co/blog/how-to-build-function-calling-and-json-mode-for-open-source-and-fine-tuned-llms/)
- [Intent Classification for Bank Chatbots through LLM Fine-Tuning](https://arxiv.org/html/2410.04925)
- [Predibase: LLM Fine-tuning for Customer Service](https://predibase.com/customer-service-automation)
