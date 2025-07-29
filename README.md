# Multi-Agent Debate System

This is a claim verification system based on multi-agent debate using the Llama3-8B-Instruct model for reasoning and debate.

```
.
├── README.md
├── requirements.txt
├── main.py
├── data/                    # Output directory for results
├── eval/                    # Evaluation scripts
│   └── eval.py             # Main evaluation script
├── chroma/
│   ├── chroma_add.py
│   ├── chroma_query.py
│   └── chroma_intent_enhanced_score_ranked.py
├── agents/
├── model/
└── prompts/
```

## Installation Steps

### 0. Set Up Python Environment

First, ensure you have Python 3.13.2 installed. We recommend using conda:

```bash
# Create a new conda environment with Python 3.13.2
conda create --name mad_debate python=3.13.2 -y

# Activate the environment
conda activate mad_debate
```

### 1. Download Llama3-8B-Instruct Model

First, you need to download the Llama3-8B-Instruct model. 

```bash
pip install huggingface-hub
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./model/llama3-8b-instruct
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage Steps

### 1. Add Data to ChromaDB

First, run `chroma_add.py` to add evidence data to the vector database:

```bash
cd chroma
python chroma_add.py
```

This script will:
- Read evidence data from the `test.json` file
- Perform deduplication for each evidence sentence
- Add unique evidence sentences to the ChromaDB vector database

### 2. Query Relevant Evidence

The system provides two evidence search methods:

#### Method 1: Basic Vector Search
Run `chroma_query.py` for basic vector similarity search:

```bash
python chroma_query.py
```

This method:
- Uses the original claim directly for vector similarity search
- Retrieves the top 20 most relevant evidence for each claim
- Output file: `retrieved_evidence_bgebase.json`

#### Method 2: Intent-Enhanced Search
Run `chroma_intent_enhanced_query.py` for intent-enhanced search:

```bash
python chroma_intent_enhanced_query.py
```

This method:
- First infers the intent of the claim
- Reformulates the claim into pro (supporting) and con (opposing) versions
- Searches separately with pro and con versions, then merges results
- Sorts by similarity score and deduplicates to get top 20
- Output file: `retrieved_evidence_bgebase_intent_enhanced.json`

#### Method 3: Groundtruth Evidence Search
Use the groundtruth evidence directly from the dataset:

```bash
# No additional query step needed - use full_evidence.json directly
```

This method:
- Uses the complete groundtruth evidence from the original dataset
- Provides all available evidence for each claim without retrieval limitations
- Input file: `data/full_evidence.json`
- No preprocessing or retrieval step required

### 3. Run Main Program

Finally, run `main.py` for claim verification:

```bash
python main.py --mode single --input_file /path/to/your/data.json
```

**Required Parameters:**
- `--input_file`: Path to the input JSON file containing claims and evidence data
- `--mode`: Choose inference mode (optional, defaults to "single")

**Available Mode Options:**
- `single`: Single agent mode
- `multi`: Multi-agent debate mode (3 rounds)
- `multi_role`: Role-based multi-agent mode
- `multi_3p`: Three-party debate mode

**Search Method Integration:**

The system supports both search methods with different modes:

**For Basic Vector Search Results:**
```bash
# Single agent mode with basic search results
python main.py --mode single --input_file data/retrieved_evidence_bgebase.json

# Multi-agent debate mode with basic search results  
python main.py --mode multi --input_file data/retrieved_evidence_bgebase.json
```

**For Intent-Enhanced Search Results:**
```bash
# Single agent mode with intent-enhanced search results
python main.py --mode single --input_file data/retrieved_evidence_bgebase_intent_enhanced.json

# Multi-agent debate mode with intent-enhanced search results
python main.py --mode multi --input_file data/retrieved_evidence_bgebase_intent_enhanced.json
```

**For Groundtruth Evidence Results:**
```bash
# Single agent mode with groundtruth evidence
python main.py --mode single --input_file data/full_evidence.json

# Multi-agent debate mode with groundtruth evidence
python main.py --mode multi --input_file data/full_evidence.json

```

## Output Results

After the program completes, it will generate:
- `data/{input_filename}_answer_map_{mode}.json`: JSON file containing all verification results
- Console output: Shows processing progress, debate process, and output file location

**Search Method Comparison:**

| Search Method | Input File | Output Pattern | Description |
|---------------|------------|----------------|-------------|
| Basic Vector | `retrieved_evidence_bgebase.json` | `{filename}_answer_map_{mode}.json` | Direct claim-to-evidence matching |
| Intent-Enhanced | `retrieved_evidence_bgebase_intent_enhanced.json` | `{filename}_answer_map_{mode}.json` | Intent-aware pro/con evidence retrieval |
| Groundtruth Evidence | `full_evidence.json` | `{filename}_answer_map_{mode}.json` | Complete groundtruth evidence from dataset |

**Output File Naming Convention:**
- The output filename is automatically generated based on the input filename
- Format: `{input_filename}_answer_map_{mode}.json`
- Example: If input file is `my_data.json` and mode is `single`, output will be `data/my_data_answer_map_single.json`

**Console Logs:**
- Input file loading confirmation
- Output file location
- Number of examples being processed
- Processing progress with tqdm
- Final save confirmation with file path and processed count

## Evaluation

After running the main program and generating prediction results, you can evaluate the performance using the evaluation script.

### Running Evaluation

The evaluation script compares your prediction results against the groundtruth data:

```bash
# Evaluate a single prediction file
python eval/eval.py --prediction /path/to/your/prediction.json

# Evaluate multiple prediction files at once
python eval/eval.py --prediction /path/to/pred1.json /path/to/pred2.json /path/to/pred3.json
```

### Evaluation Metrics

The script provides comprehensive evaluation metrics:

- **Overall Accuracy**: Percentage of correct predictions across all examples
- **Class-wise Accuracy**: Accuracy for each class (TRUE, HALF-TRUE, FALSE)
- **F1 Scores**: Precision, Recall, and F1 score for each class
- **Macro-F1**: Average F1 score across all classes

### Example Output

```
Groundtruth file: data/GT_test_all.json
Prediction files: ['/path/to/prediction.json']
================================================================================
File: /path/to/prediction.json
  Mode: single
  Total examples compared: 400
  Correct predictions: 320
  Overall Accuracy: 80.00%

  Class-wise Accuracy:
    TRUE: 85.00% (85/100)
    HALF-TRUE: 75.00% (75/100)
    FALSE: 80.00% (160/200)

  F1 Scores:
    TRUE - Precision: 82.00%, Recall: 85.00%, F1: 83.47%
    HALF-TRUE - Precision: 78.00%, Recall: 75.00%, F1: 76.47%
    FALSE - Precision: 81.00%, Recall: 80.00%, F1: 80.50%
  Macro-F1: 80.15%
--------------------------------------------------------------------------------
```

### Supported Prediction Formats

The evaluation script supports two prediction formats:

1. **Single Format**: List of verdict strings
   ```json
   {
     "example_id_1": ["[VERDICT]: TRUE"],
     "example_id_2": ["[VERDICT]: FALSE"]
   }
   ```

2. **Multi Format**: Dictionary with final_verdict field
   ```json
   {
     "example_id_1": {"final_verdict": "[VERDICT]: TRUE"},
     "example_id_2": {"final_verdict": "[VERDICT]: FALSE"}
   }
   ```

The script automatically detects the format and extracts verdicts accordingly.

## System Requirements

1. **Python Version**: Python 3.13.2 (required)
2. **Model Download**: Ensure sufficient disk space for Llama3-8B-Instruct model (approximately 16GB)
3. **Memory Requirements**: At least 16GB RAM to run the model
4. **GPU Support**: GPU acceleration recommended for inference, requires CUDA version of PyTorch
5. **Data Paths**: Please modify file paths in the code according to your actual file locations

## Troubleshooting

If you encounter issues:

1. **Python Version Issues**: Ensure you're using Python 3.13.2 exactly
2. **Model Download Failure**: Check network connection or use mirror sources
3. **Insufficient Memory**: Consider using model quantization or reducing batch size
4. **ChromaDB Errors**: Ensure ChromaDB service is running properly
5. **Dependency Conflicts**: Consider using a virtual environment with the specified Python version

## License

Please ensure compliance with Llama3 model usage terms and license requirements. 