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
- Searches separately with pro and con versions, each retrieving top 10 results
- Merges pro and con results and deduplicates to get final evidence set
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
- `multi_people`: Multi-agent debate mode with politician vs scientist roles
- `multi_people_3`: Multi-agent debate mode with journalist, politician, and scientist (3 agents)
- `multi_role`: Role-based multi-agent mode with intent inference
- `multi_role_3`: Role-based multi-agent mode with journalist, pro, and con agents (3 agents)
- `pcj_3`: Pro-Con-Journalist debate with intent inference and claim reformulation
- `four_agents`: Four-agent debate mode (2 pro vs 2 con agents)
- `four_agents_people`: Four-agent debate mode with politician, scientist, journalist, and domain specialist

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

# Role-based multi-agent mode with groundtruth evidence
python main.py --mode multi_role --input_file data/full_evidence.json

# Multi-agent people mode with groundtruth evidence
python main.py --mode multi_people --input_file data/full_evidence.json

# Multi-agent people 3 mode with groundtruth evidence
python main.py --mode multi_people_3 --input_file data/full_evidence.json

# Multi-agent role 3 mode with groundtruth evidence
python main.py --mode multi_role_3 --input_file data/full_evidence.json

# PCJ-3 mode with groundtruth evidence
python main.py --mode pcj_3 --input_file data/full_evidence.json

# Four agents mode with groundtruth evidence
python main.py --mode four_agents --input_file data/full_evidence.json

# Four agents people mode with groundtruth evidence
python main.py --mode four_agents_people --input_file data/full_evidence.json
```

## Mode Descriptions

### Single Agent Mode (`single`)
- Uses a single agent to verify claims
- Output: List containing the verification result
- Fastest execution time

### Multi-Agent Debate Mode (`multi`)
- Implements a 3-round debate between Pro and Con agents
- Rounds: Opening statements → Rebuttals → Closing statements → Final verdict
- Output: Complete debate transcript with final verdict

### Role-Based Multi-Agent Mode (`multi_role`)
- **Step 1**: Infers the intent of the claim and determines appropriate roles for supporting and opposing agents
- **Step 2-4**: Conducts 3-round debate with role-specific agents
- **Step 5**: Final verdict by judge
- Output: Intent inference, role assignments, complete debate transcript, and final verdict
- Features: Dynamic role assignment based on claim intent

### Multi-Agent People Mode (`multi_people`)
- Debate between a Politician and a Scientist with distinct perspectives
- Politician: Focuses on public opinion, policy implications, and practical considerations
- Scientist: Emphasizes empirical evidence, methodology, and academic rigor
- Output: Complete debate transcript with final verdict

### Multi-Agent People 3 Mode (`multi_people_3`)
- Three-agent debate system with Journalist, Politician, and Scientist
- **Journalist**: Provides neutral analysis and fact-checking
- **Politician**: Focuses on public opinion, policy implications, and practical considerations
- **Scientist**: Emphasizes empirical evidence, methodology, and academic rigor
- **Debate Flow**: Journalist → Politician → Scientist in each round
- Output: Complete debate transcript with final verdict

### Multi-Agent Role 3 Mode (`multi_role_3`)
- Enhanced role-based debate with three agents: Journalist, Pro, and Con
- **Step 1**: Infers the intent of the claim and determines appropriate roles for supporting and opposing agents
- **Step 2-4**: Conducts 3-round debate with Journalist moderating and Pro/Con agents with specific roles
- **Journalist**: Provides neutral analysis and moderates the debate
- **Pro/Con Agents**: Assigned specific roles based on claim intent
- **Step 5**: Final verdict by judge
- Output: Intent inference, role assignments, complete debate transcript, and final verdict

### PCJ-3 Mode (`pcj_3`)
- Pro-Con-Journalist debate system with advanced features
- **Intent Inference**: Analyzes claim intent and reformulates for better understanding
- **Claim Reformulation**: Creates pro and con versions of the claim
- **Journalist Moderation**: Provides neutral analysis and fact-checking
- **Structured Debate**: Three-round debate with opening, rebuttal, and closing statements
- Output: Complete debate process including intent analysis, claim reformulation, and final verdict

### Four Agents Mode (`four_agents`)
- Four-agent debate system with 2 Pro agents vs 2 Con agents
- **Pro Agents**: Two agents supporting the claim from different perspectives
- **Con Agents**: Two agents opposing the claim from different perspectives
- **Debate Structure**: Opening statements → Rebuttals → Closing statements → Final verdict
- Each agent provides unique arguments and perspectives
- Output: Complete debate transcript with final verdict

### Four Agents People Mode (`four_agents_people`)
- Four-agent debate system with distinct professional roles
- **Politician**: Focuses on public opinion, policy implications, and practical considerations
- **Scientist**: Emphasizes empirical evidence, methodology, and academic rigor
- **Journalist**: Provides neutral analysis, fact-checking, and public interest perspective
- **Domain Specialist**: Expert in the specific domain of the claim (automatically inferred)
- **Dynamic Domain Assignment**: System automatically determines the most appropriate domain specialist for each claim
- **Debate Structure**: Opening statements → Rebuttals → Closing statements → Final verdict
- Output: Domain specialist assignment, complete debate transcript, and final verdict

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

**Output Format by Mode:**

**Single Mode:**
```json
{
  "example_id": ["[VERDICT]: TRUE"]
}
```

**Multi Mode:**
```json
{
  "example_id": {
    "pro_opening": "...",
    "con_opening": "...",
    "pro_rebuttal": "...",
    "con_rebuttal": "...",
    "pro_closing": "...",
    "con_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

**Multi-Role Mode:**
```json
{
  "example_id": {
    "intent": "The claim is about...",
    "support_role": "Expert",
    "oppose_role": "Skeptic",
    "pro_opening": "...",
    "con_opening": "...",
    "pro_rebuttal": "...",
    "con_rebuttal": "...",
    "pro_closing": "...",
    "con_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

**Multi-People Mode:**
```json
{
  "example_id": {
    "politician_opening": "...",
    "scientist_opening": "...",
    "politician_rebuttal": "...",
    "scientist_rebuttal": "...",
    "politician_closing": "...",
    "scientist_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

**Multi-People 3 Mode:**
```json
{
  "example_id": {
    "journalist_opening": "...",
    "politician_opening": "...",
    "scientist_opening": "...",
    "journalist_rebuttal": "...",
    "politician_rebuttal": "...",
    "scientist_rebuttal": "...",
    "journalist_closing": "...",
    "politician_closing": "...",
    "scientist_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

**Multi-Role 3 Mode:**
```json
{
  "example_id": {
    "intent": "The claim is about...",
    "support_role": "Expert",
    "oppose_role": "Skeptic",
    "journalist_opening": "...",
    "pro_opening": "...",
    "con_opening": "...",
    "journalist_rebuttal": "...",
    "pro_rebuttal": "...",
    "con_rebuttal": "...",
    "journalist_closing": "...",
    "pro_closing": "...",
    "con_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

**PCJ-3 Mode:**
```json
{
  "example_id": {
    "intent_analysis": "...",
    "claim_reformulation": {
      "pro_version": "...",
      "con_version": "..."
    },
    "journalist_opening": "...",
    "pro_opening": "...",
    "con_opening": "...",
    "journalist_rebuttal": "...",
    "pro_rebuttal": "...",
    "con_rebuttal": "...",
    "journalist_closing": "...",
    "pro_closing": "...",
    "con_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

**Four Agents Mode:**
```json
{
  "example_id": {
    "pro1_opening": "...",
    "pro2_opening": "...",
    "con1_opening": "...",
    "con2_opening": "...",
    "pro1_rebuttal": "...",
    "pro2_rebuttal": "...",
    "con1_rebuttal": "...",
    "con2_rebuttal": "...",
    "pro1_closing": "...",
    "pro2_closing": "...",
    "con1_closing": "...",
    "con2_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

**Four Agents People Mode:**
```json
{
  "example_id": {
    "domain_specialist": "Medical Expert",
    "politician_opening": "...",
    "scientist_opening": "...",
    "journalist_opening": "...",
    "domain_scientist_opening": "...",
    "politician_rebuttal": "...",
    "scientist_rebuttal": "...",
    "journalist_rebuttal": "...",
    "domain_scientist_rebuttal": "...",
    "politician_closing": "...",
    "scientist_closing": "...",
    "journalist_closing": "...",
    "domain_scientist_closing": "...",
    "final_verdict": "[VERDICT]: TRUE"
  }
}
```

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