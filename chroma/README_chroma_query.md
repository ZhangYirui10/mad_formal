# Chroma Intent Enhanced Query

This script supports intent-enhanced retrieval using different models.

## Supported Models

1. **Qwen2.5 7B Instruct** (default)
2. **Llama3 8B Instruct** 
3. **GPT models** (requires OpenAI API key)

## Usage

### 1. Using Qwen2.5 7B Instruct (default)

```bash
python chroma/chroma_intent_enhanced_query.py
```

Or explicitly specify:

```bash
python chroma/chroma_intent_enhanced_query.py --model qwen
```

### 2. Using Llama3 8B Instruct

Using default path:
```bash
python chroma/chroma_intent_enhanced_query.py --model llama
```

Using custom path:
```bash
python chroma/chroma_intent_enhanced_query.py --model llama --model_path /path/to/your/llama3-8b-instruct
```

### 3. Using GPT models

```bash
python chroma/chroma_intent_enhanced_query.py --model gpt --api_key your_openai_api_key
```

Using different GPT models:
```bash
python chroma/chroma_intent_enhanced_query.py --model gpt --api_key your_openai_api_key --gpt_model_name gpt-4
```

## Output Files

The script generates different output files based on the selected model type:

- Qwen: `retrieved_evidence_bgebase_intent_enhanced_qwen.json`
- Llama: `retrieved_evidence_bgebase_intent_enhanced_llama.json`
- GPT: `retrieved_evidence_bgebase_intent_enhanced_gpt.json`

## Dependencies

### For Local Models (Qwen/Llama)
```bash
pip install transformers torch chromadb tqdm
```

### For GPT Models
```bash
pip install openai transformers torch chromadb tqdm
```

## Hardware Requirements

- **Qwen2.5 7B**: Requires approximately 14GB GPU memory
- **Llama3 8B**: Requires approximately 16GB GPU memory
- **GPT**: No local GPU required, but needs internet connection and OpenAI API access

## Examples

```bash
# Using Qwen model
python chroma/chroma_intent_enhanced_query.py --model qwen

# Using local Llama model
python chroma/chroma_intent_enhanced_query.py --model llama --model_path ./models/llama3-8b-instruct

# Using GPT-4
python chroma/chroma_intent_enhanced_query.py --model gpt --api_key sk-... --gpt_model_name gpt-4
``` 